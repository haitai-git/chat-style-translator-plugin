"""聊天风格转译插件：LLM 分析与语义过滤。

- analyze_messages：把收集到的对话交给 LLM，输出场景 + 成员风格 + 语句（含说话对象与含义）。
- merge_models：把多个模型（或多次窗口分析结果）融合成一个，相似场景与相似含义语句合并。
- compact_model：精简单个模型（纯本地去重），合并相似场景/近似语句/风格快照并限制语句总数。
- SemanticFilter：黑名单语义相似度过滤，优先用 embedding 余弦相似度，失败时退回 LLM 判定或字面相似度。
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("chat_style_translator.analyzer")

# 分析/融合请求可能耗时较长（大窗口 + 长 JSON 输出），覆盖 MaiBot 插件 RPC 默认 30 秒超时
LLM_RPC_TIMEOUT_MS = 180000

# 融合后的场景/快照合并阈值：相似的合并为一个（方便 AI 按场景直接检索），跨度大的错开保留
SCENE_MERGE_SIMILARITY = 0.8
SNAPSHOT_MERGE_SIMILARITY = 0.55

# 单模型精简：激进精简时，场景内字符二元组相似度达到该阈值的近似句合并为一条
STATEMENT_MERGE_SIMILARITY = 0.8

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[Any]:
    """从 LLM 回复中提取 JSON（支持对象与数组）。

    优先剥离 markdown 代码围栏，再尝试截取最外层花括号或方括号。
    """

    text = str(text or "").strip()
    if not text:
        return None
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end > array_start:
        try:
            return json.loads(text[array_start : array_end + 1])
        except json.JSONDecodeError:
            return None
    return None


class StyleAnalyzer:
    """基于 LLM 的群聊风格分析器。"""

    def __init__(self, ctx: Any) -> None:
        """初始化。

        Args:
            ctx: 插件上下文（用于 self.ctx.llm.generate）。
        """
        self._ctx = ctx
        self._model = ""
        self._max_statements_per_scene = 30

    def update_config(self, model: str = "", max_statements_per_scene: int = 30) -> None:
        """同步配置。"""

        self._model = str(model or "").strip()
        self._max_statements_per_scene = int(max_statements_per_scene or 30)

    async def _generate_json(self, system_prompt: str, user_prompt: str) -> Optional[Any]:
        """调用 LLM 并要求返回 JSON，解析失败时重试一次。"""

        result = await self._ctx.llm.generate(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self._model,
            temperature=0.3,
            rpc_timeout_ms=LLM_RPC_TIMEOUT_MS,
        )
        if not result.get("success"):
            logger.warning("LLM 生成失败: %s", result.get("error", "未知错误"))
            return None
        content = str(result.get("response") or "")
        parsed = extract_json(content)
        if parsed is not None:
            return parsed
        repair = await self._ctx.llm.generate(
            [
                {"role": "system", "content": "你只负责把用户给出的文本改写成合法的 JSON，不要添加任何多余内容。"},
                {"role": "user", "content": f"把下面文本改成 JSON 输出：\n{content}"},
            ],
            model=self._model,
            temperature=0.0,
            rpc_timeout_ms=LLM_RPC_TIMEOUT_MS,
        )
        return extract_json(str(repair.get("response") or ""))

    # ------------------------------------------------------------------
    # 对话转写
    # ------------------------------------------------------------------

    @staticmethod
    def build_transcript(messages: List[Dict[str, Any]]) -> str:
        """把收集的消息拼成适合 LLM 阅读的对话转写文本。"""

        lines = []
        for index, message in enumerate(messages, start=1):
            name = str(message.get("sender_name") or "未知")
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            target = str(message.get("at_user_name") or message.get("reply_to_name") or "").strip()
            prefix = f"{index}. {name}"
            if target:
                prefix += f" 对「{target}」说"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 场景分析
    # ------------------------------------------------------------------

    def _analysis_system_prompt(self) -> str:
        return (
            "你是一名群聊风格分析师。你会收到一个 QQ 群的对话记录（每行格式：序号. 说话人 [对「目标」说]: 内容）。"
            "请完成以下分析：\n"
            "1. 判断群友们正在聊什么，涉及哪些场景（比如打招呼、闲聊、吐槽、提问求助、表情包互斗等），场景可以多个；\n"
            "2. 分析每一句话是跟谁说的（对群全体说、对某个人说）；\n"
            "3. 分析每位活跃群友个人的聊天风格（用词习惯、语气、长度、情绪、口头禅），"
            "并给出他的默认性格（性格底色、情绪倾向、与人相处的惯用方式）；\n"
            "4. 在每个场景下挑选最具代表性的原话作为语句，并注明说话人、说话对象、这句话在这个场景下的含义；\n"
            "5. 对整体群聊风格做总结。\n\n"
             "只输出 JSON，不要输出任何其他内容。JSON 结构：\n"
             "{\n"            '  "overall_style": "整体聊天风格总结",\n'
            '  "members": [{"user_id": "QQ号（必须与对话记录一致）", "name": "昵称", "style": "个人聊天风格描述",\n'
            '     "personality": "默认性格（性格底色、情绪倾向、相处方式）", "speaking_traits": ["特点1", "特点2"]}],\n'
            '  "scenes": [\n'
            '    {"name": "场景名", "description": "场景描述", "style": "该场景下的说话风格",\n'
            '     "note": "场景备注：这个场景在聊什么、什么语境下出现、群友们一般怎么参与",\n'
            '     "statements": [{"text": "原话", "speaker": "QQ号", "speaker_name": "昵称", "target": "说话对象或\\"大家\\"",\n'
            '       "meaning": "备注：这句话在这个场景下的意思（人怎么看懂这句话）",\n'
            '       "ai_note": "AI理解：AI 在回复时应当如何理解这句话（语气、言外之意、适合怎么用）",\n'
            '       "tags": ["标签"]}]}\n'
            "  ]\n"
            "}\n"
            "注意：语句必须是对话中出现过的原话，不要编造；每个场景的 statements 不要超过 10 条；"
            "meaning 和 ai_note 都要写清楚，方便使用者审核。\n"
            "最后强调：不要输出思考过程、解释或任何多余文字，直接从 { 开始输出完整 JSON，一次性输出，不要截断。"
        )

    async def analyze_window(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """分析一段对话窗口，返回一个部分模型（场景/成员/语句）。"""

        transcript = self.build_transcript(messages)
        if not transcript.strip():
            return None
        prompt = (
            f"以下是这个群最近的一段对话（共 {len(messages)} 条）：\n\n{transcript}\n\n"
            "请按系统要求输出分析结果 JSON。"
        )
        parsed = await self._generate_json(self._analysis_system_prompt(), prompt)
        if not isinstance(parsed, dict):
            logger.warning("场景分析返回了非法 JSON")
            return None
        if not (parsed.get("scenes") or parsed.get("members")):
            logger.warning("场景分析返回了空结果（无场景与成员），丢弃该窗口")
            return None
        parsed.setdefault("overall_style", "")
        parsed.setdefault("members", [])
        parsed.setdefault("scenes", [])
        return parsed

    # ------------------------------------------------------------------
    # 融合（窗口融合 / 模型合并）
    # ------------------------------------------------------------------

    def _merge_system_prompt(self, options: Optional[Dict[str, Any]] = None) -> str:
        """构造模型融合提示词。

        Args:
            options: 融合选项，可包含：
                - ratio: (来源A占比, 来源B占比) 元组，遇到含义相同/高度相似的重复语句时，
                  按该比例分配优先保留哪一方的表述（仅两个来源时生效）；
                - dedup_level: "标准" / "激进精简" / "不去重"；
                - max_total_statements: 融合后所有场景合计的语句总数上限（0 表示不限制）。
        """

        options = options or {}
        ratio = options.get("ratio")
        dedup_level = str(options.get("dedup_level") or "标准")
        max_total = int(options.get("max_total_statements") or 0)

        rules = [
            "合并所有成员档案：成员按 QQ 号（user_id）判断是否同一个人，同一 QQ 号只保留一份档案，"
            "风格与默认性格（personality）描述取更详尽、更准确的；",
            "场景按内容相似度合并：内容相似或重复的场景必须融合为一个（无论来自哪份来源、"
            "哪个时段或批次，都不能因为来源/时段不同而分开保留），场景名取更贴切的，"
            "风格描述取综合描述，note 备注取更完整的；",
            "整体风格（overall_style）不合并：overall_style 输出空字符串，系统会分别保留各来源的整体风格，"
            "你不需要总结或合并整体风格；",
        ]
        if dedup_level == "不去重":
            rules.append("语句不去重：两个来源的语句都保留，即使完全重复也不删除；")
        elif dedup_level == "激进精简":
            rules.append(
                "语句激进精简：完全重复、含义高度相似、以及含义相近（表述不同但意思接近）的语句都合并为一条，"
                "保留更完整、更典型的那条；内容相近的场景也合并为一个；"
            )
        else:
            rules.append(
                "语句要去重：完全重复或含义高度相似的语句只保留一条，可以保留更完整的那句；"
                "语句的 speaker 必须使用真实 QQ 号（user_id）；"
            )
        if ratio is not None and len(ratio) == 2:
            ratio_a, ratio_b = int(ratio[0]), int(ratio[1])
            rules.append(
                f"融合比例（仅两个来源时生效）：当两个来源出现含义相同或高度相似的重复语句时，只保留一条，"
                f"并按 来源A:来源B = {ratio_a}:{ratio_b} 的比例决定保留哪一方的表述；"
                "请尽量按此比例均衡分配，不要全部偏向一边；"
            )
        rules.append(
            "只有场景本身确实不同时，才允许不同场景保留意思相近的语句（在 meaning 里说明各自场景差异）；"
            "场景内容相似或重复的必须合并为一个，不能因为来源或时段不同而分开保留；"
        )
        rules.append("融合时保留每条语句的 meaning（这句话的意思）与 ai_note（AI 理解），含义不清时补全；")
        rules.append(f"每个场景最终保留不超过 {self._max_statements_per_scene} 条最有代表性的语句；")
        if max_total > 0:
            rules.append(
                f"整个模型所有场景合计的语句总数不超过 {max_total} 条；"
                "超出时优先保留更典型、更完整的语句；"
            )
        numbered_rules = "".join(f"{index}. {rule}\n" for index, rule in enumerate(rules, start=1))
        return (
            "你负责把多份「群聊风格分析结果」融合成一份完整模型。规则：\n"
            f"{numbered_rules}"
            "只输出 JSON，不要输出任何其他内容。输出结构与输入结构一致：\n"
            "{\"overall_style\": \"\", \"members\": [{\"user_id\": \"\", \"name\": \"\", \"style\": \"\", "
            "\"personality\": \"\", \"speaking_traits\": []}], "
            "\"scenes\": [{\"name\": \"\", \"description\": \"\", \"style\": \"\", \"note\": \"\", "
            "\"statements\": [{\"text\": \"\", \"speaker\": \"\", \"speaker_name\": \"\", \"target\": \"\", \"meaning\": \"\", \"ai_note\": \"\", \"tags\": []}]}]}"
            "\n最后强调：不要输出思考过程、解释或任何多余文字，直接从 { 开始输出完整 JSON，一次性输出，不要截断。"
        )

    async def merge_partials(
        self,
        partials: List[Dict[str, Any]],
        name: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """融合多份窗口分析结果（或用户指定的多个模型）为一份完整模型。

        Args:
            partials: 待融合的分析结果列表。
            name: 融合后模型名。
            options: 融合选项（比例 / 去重强度 / 语句总数上限 / 合并阈值），见 :meth:`_merge_system_prompt`。
                - ``scene_merge_threshold``：场景合并相似度阈值（0.5~0.95，默认 0.8）；
                - ``snapshot_merge_threshold``：整体风格快照合并相似度阈值（0.3~0.85，默认 0.55）。

        整体风格不合并：各来源（不同分析批次）的整体风格分别存入 ``style_snapshots`` 列表，
        ``overall_style`` 字段兼容保留第一条（旧模型无 style_snapshots 时保持原值）。
        """

        if not partials:
            return None
        snapshots = self._collect_style_snapshots(partials)
        scene_threshold = float((options or {}).get("scene_merge_threshold") or SCENE_MERGE_SIMILARITY)
        snapshot_threshold = float((options or {}).get("snapshot_merge_threshold") or SNAPSHOT_MERGE_SIMILARITY)
        if len(partials) == 1:
            merged = json.loads(json.dumps(partials[0]))
        else:
            payload = json.dumps(partials, ensure_ascii=False, indent=2)
            prompt = (
                f"以下是需要融合的 {len(partials)} 份群聊风格分析结果：\n{payload}\n\n"
                "请按系统要求输出融合后的 JSON。"
            )
            merged = await self._generate_json(self._merge_system_prompt(options), prompt)
            if not isinstance(merged, dict) or not (
                merged.get("scenes") or merged.get("members") or str(merged.get("overall_style") or "").strip()
            ):
                logger.warning("融合结果非法或为空，改用简单拼接兜底（保留全部窗口内容）")
                merged = {
                    "overall_style": "",
                    "members": [],
                    "scenes": [],
                }
                seen_user_ids: set[str] = set()
                for partial in partials:
                    for member in partial.get("members") or []:
                        user_id = str(member.get("user_id") or "").strip()
                        if user_id and user_id in seen_user_ids:
                            continue
                        if user_id:
                            seen_user_ids.add(user_id)
                        merged["members"].append(member)
                    merged["scenes"].extend(partial.get("scenes") or [])
        merged.setdefault("overall_style", "")
        merged.setdefault("members", [])
        merged.setdefault("scenes", [])
        merged["style_snapshots"] = snapshots
        if not snapshots and str(merged.get("overall_style") or "").strip():
            merged["style_snapshots"] = [str(merged.get("overall_style") or "").strip()]
        merged["name"] = name or "群聊风格模型"
        normalized = self._normalize_model(merged)
        # 相似的场景合并为一个（方便 AI 按场景直接检索），跨度大的场景错开保留
        normalized["scenes"] = self._merge_similar_scenes(normalized.get("scenes") or [], threshold=scene_threshold)
        # 整体风格快照同样按相似度合并，差异大的错开放
        normalized["style_snapshots"] = self._merge_similar_snapshots(
            normalized.get("style_snapshots") or [], threshold=snapshot_threshold
        )
        return normalized

    @staticmethod
    def _collect_style_snapshots(partials: List[Dict[str, Any]]) -> List[str]:
        """收集各来源的整体风格快照（不合并，保留来源先后顺序，去空去重）。

        优先取来源已有的 ``style_snapshots`` 列表；没有时用其 ``overall_style`` 单条。
        """

        snapshots: List[str] = []
        for partial in partials:
            if not isinstance(partial, dict):
                continue
            existing = partial.get("style_snapshots")
            if isinstance(existing, list):
                items = [str(item).strip() for item in existing if str(item or "").strip()]
            else:
                items = [str(partial.get("overall_style") or "").strip()]
            for item in items:
                if item and item not in snapshots:
                    snapshots.append(item)
        return snapshots

    # ------------------------------------------------------------------
    # 相似合并（场景 / 整体风格快照）
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_similar_snapshots(snapshots: List[str], threshold: float = SNAPSHOT_MERGE_SIMILARITY) -> List[str]:
        """整体风格快照按相似度合并：相似的风格合并为一条，差异大的错开放。

        依次比较字符二元组相似度，与已保留任一条相似的并入（保留更长更完整的一条），
        否则作为独立快照追加（保持来源先后顺序）。
        """

        merged: List[str] = []
        for snapshot in snapshots:
            snapshot = str(snapshot or "").strip()
            if not snapshot:
                continue
            hit_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if StyleAnalyzer._text_similarity(snapshot, existing) >= threshold
                ),
                None,
            )
            if hit_index is None:
                merged.append(snapshot)
            elif len(snapshot) > len(merged[hit_index]):
                merged[hit_index] = snapshot
        return merged

    @staticmethod
    def _merge_similar_scenes(scenes: List[Dict[str, Any]], threshold: float = SCENE_MERGE_SIMILARITY) -> List[Dict[str, Any]]:
        """场景按相似度合并：内容相似或重复的场景合并为一个（语句、风格描述合并），差异大的错开保留。

        场景名去掉标点/空白后相同（如「监狱保释梗」与「监狱/保释梗」）或互相包含（如「闲聊」
        与「闲聊吹水」）、语句原文大面积重复时必定合并（与场景来自哪个来源/时段无关）；
        其余按 场景名 + 风格 + 备注 + 语句原文 的字符二元组相似度与阈值比较；
        合并时语句去重（按原文），场景名与描述取更长更完整的。
        """

        merged: List[Dict[str, Any]] = []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            hit_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if StyleAnalyzer._scenes_should_merge(existing, scene, threshold)
                ),
                None,
            )
            if hit_index is None:
                merged.append(json.loads(json.dumps(scene)))
                continue
            target = merged[hit_index]
            if len(str(scene.get("name") or "")) > len(str(target.get("name") or "")):
                target["name"] = scene.get("name")
            for key in ("description", "style", "note"):
                src_text = str(scene.get(key) or "").strip()
                if src_text and len(src_text) > len(str(target.get(key) or "")):
                    target[key] = src_text
            seen_texts = {str(st.get("text") or "").strip() for st in target.get("statements") or []}
            for statement in scene.get("statements") or []:
                if not isinstance(statement, dict):
                    continue
                text = str(statement.get("text") or "").strip()
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                target.setdefault("statements", []).append(json.loads(json.dumps(statement)))
        return merged

    @staticmethod
    def _scenes_should_merge(scene_a: Dict[str, Any], scene_b: Dict[str, Any], threshold: float) -> bool:
        """判断两个场景是否应合并。

        规范化场景名相同或互相包含、语句原文大面积重叠，或整体文本相似度达到阈值时合并
        （与场景来自哪个来源/时段无关）。
        """

        name_a = StyleAnalyzer._normalize_key(str(scene_a.get("name") or ""))
        name_b = StyleAnalyzer._normalize_key(str(scene_b.get("name") or ""))
        if name_a and name_b:
            if name_a == name_b:
                return True
            if len(name_a) >= 2 and len(name_b) >= 2 and (name_a in name_b or name_b in name_a):
                return True
        if StyleAnalyzer._scene_similarity(scene_a, scene_b) >= threshold:
            return True
        texts_a = {str(st.get("text") or "").strip() for st in scene_a.get("statements") or [] if isinstance(st, dict)}
        texts_b = {str(st.get("text") or "").strip() for st in scene_b.get("statements") or [] if isinstance(st, dict)}
        texts_a.discard("")
        texts_b.discard("")
        if texts_a and texts_b:
            overlap = len(texts_a & texts_b)
            if overlap / len(texts_a | texts_b) > 0.5:
                return True
        return False

    @staticmethod
    def _normalize_key(text: str) -> str:
        """规范化用于名称比较的文本：去掉空白、常见标点与引号。"""

        cleaned = re.sub(r"[\s/、，。！？；：（）()\-_·]+", "", str(text or ""))
        return cleaned.replace('"', "").replace("'", "")

    @staticmethod
    def _scene_similarity(scene_a: Dict[str, Any], scene_b: Dict[str, Any]) -> float:
        """计算两个场景的相似度（场景名 + 风格 + 备注 + 语句原文）。"""

        text_a = StyleAnalyzer._scene_text(scene_a)
        text_b = StyleAnalyzer._scene_text(scene_b)
        return StyleAnalyzer._text_similarity(text_a, text_b)

    @staticmethod
    def _scene_text(scene: Dict[str, Any]) -> str:
        """把场景渲染成用于相似度比较的文本。"""

        parts = [str(scene.get("name") or ""), str(scene.get("style") or ""), str(scene.get("note") or "")]
        for statement in scene.get("statements") or []:
            if isinstance(statement, dict):
                parts.append(str(statement.get("text") or ""))
        return "".join(parts)

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """字符二元组 Jaccard 相似度（用于场景/快照合并与字面兜底）。"""

        def bigrams(text: str) -> set[str]:
            text = re.sub(r"\s+", "", text)
            return {text[index : index + 2] for index in range(len(text) - 1)}

        grams_a = bigrams(text_a)
        grams_b = bigrams(text_b)
        if not grams_a or not grams_b:
            return 0.0
        return len(grams_a & grams_b) / len(grams_a | grams_b)

    # ------------------------------------------------------------------
    # 单模型精简（纯本地去重，不调用 LLM）
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_statements(
        statements: List[Dict[str, Any]], aggressive: bool
    ) -> Tuple[List[Dict[str, Any]], int]:
        """场景内语句去重：完全重复必去；激进精简时近似句也合并。

        近似句 = 去空白后字符二元组相似度达到 STATEMENT_MERGE_SIMILARITY；
        合并时保留更长更完整的那条，缺失的 meaning/ai_note 从另一条补全。
        返回 (去重后的语句, 移除条数)。
        """

        merged: List[Dict[str, Any]] = []
        removed = 0
        for statement in statements:
            if not isinstance(statement, dict):
                removed += 1
                continue
            text = str(statement.get("text") or "").strip()
            if not text:
                removed += 1
                continue
            hit_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if str(existing.get("text") or "").strip() == text
                    or (
                        aggressive
                        and StyleAnalyzer._text_similarity(
                            text, str(existing.get("text") or "")
                        )
                        >= STATEMENT_MERGE_SIMILARITY
                    )
                ),
                None,
            )
            if hit_index is None:
                merged.append(json.loads(json.dumps(statement)))
                continue
            removed += 1
            target = merged[hit_index]
            src = statement
            if len(text) > len(str(target.get("text") or "")):
                target, src = src, target
            for key in ("meaning", "ai_note"):
                if not str(target.get(key) or "").strip() and str(src.get(key) or "").strip():
                    target[key] = src.get(key)
            merged[hit_index] = json.loads(json.dumps(target))
        return merged, removed

    @staticmethod
    def _cap_total_statements(
        scenes: List[Dict[str, Any]], limit: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """限制模型语句总数：超限时优先保留更长更完整的语句，保留场景内先后顺序。

        每个场景按当前占比分配预算（至少保留 1 条），仍超限时全局砍最短。
        返回 (处理后的场景, 移除条数)。
        """

        if limit <= 0:
            return scenes, 0
        total = sum(len(scene.get("statements") or []) for scene in scenes)
        if total <= limit:
            return scenes, 0
        kept_by_scene: List[List[Dict[str, Any]]] = []
        for scene in scenes:
            statements = scene.get("statements") or []
            share = max(1, round(len(statements) * limit / total))
            ordered = sorted(
                enumerate(statements),
                key=lambda pair: (-len(str(pair[1].get("text") or "")), pair[0]),
            )
            chosen = sorted(pair[0] for pair in ordered[:share])
            kept_by_scene.append([statements[index] for index in chosen])
        removed = total - sum(len(kept) for kept in kept_by_scene)
        overflow = sum(len(kept) for kept in kept_by_scene) - limit
        while overflow > 0:
            candidates = [
                (len(str(statement.get("text") or "")), scene_index, statement_index)
                for scene_index, kept in enumerate(kept_by_scene)
                for statement_index, statement in enumerate(kept)
            ]
            if not candidates:
                break
            _, scene_index, statement_index = min(candidates, key=lambda item: (item[0], -item[1]))
            del kept_by_scene[scene_index][statement_index]
            overflow -= 1
            removed += 1
        for scene, kept in zip(scenes, kept_by_scene):
            scene["statements"] = kept
        return scenes, removed

    def compact_model(
        self,
        model: Dict[str, Any],
        name: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """精简单个风格模型（纯本地处理，不调用 LLM，不改变语句内容，只减少重复/相似内容）。

        相似或重复的场景必定合并为一条（场景名相同/互相包含或语句大面积重复时直接合并，
        与其原来自哪个时段/批次无关）。

        处理顺序：
        1. 规范化模型结构（去空、清洗字段）；
        2. 场景内语句去重：完全重复必去；激进精简时近似句合并；不去重则跳过；
        3. 相似场景合并（scene_merge_threshold）；
        4. 整体风格快照合并（snapshot_merge_threshold）；
        5. 语句总数上限（max_total_statements，保留更长更完整的语句）。

        Args:
            model: 待精简的模型。
            name: 精简后模型名。
            options: 精简选项，可包含：
                - dedup_level: "标准" / "激进精简" / "不去重"；
                - max_total_statements: 语句总数上限（0 不限制）；
                - scene_merge_threshold: 场景合并相似度阈值（0.5~0.95，默认 0.8）；
                - snapshot_merge_threshold: 风格快照合并相似度阈值（0.3~0.85，默认 0.55）。

        Returns:
            (精简后的模型, 统计信息 {statement_removed, scene_merged})。
        """

        options = options or {}
        dedup_level = str(options.get("dedup_level") or "标准")
        scene_threshold = float(options.get("scene_merge_threshold") or SCENE_MERGE_SIMILARITY)
        snapshot_threshold = float(options.get("snapshot_merge_threshold") or SNAPSHOT_MERGE_SIMILARITY)
        max_total = int(options.get("max_total_statements") or 0)

        compacted = self._normalize_model(json.loads(json.dumps(model)))
        compacted["style_snapshots"] = self._merge_similar_snapshots(
            compacted.get("style_snapshots") or [], threshold=snapshot_threshold
        )

        statement_removed = 0
        aggressive = dedup_level == "激进精简"
        skip_dedup = dedup_level == "不去重"
        scenes = compacted.get("scenes") or []
        for scene in scenes:
            if not skip_dedup:
                scene["statements"], removed = self._dedup_statements(
                    scene.get("statements") or [], aggressive
                )
                statement_removed += removed
        scene_count_before = len(scenes)
        compacted["scenes"] = self._merge_similar_scenes(scenes, threshold=scene_threshold)
        scene_merged = scene_count_before - len(compacted["scenes"])

        compacted["scenes"], capped = self._cap_total_statements(compacted["scenes"], max_total)
        statement_removed += capped

        compacted["name"] = name or str(model.get("name") or "群聊风格模型")
        return compacted, {"statement_removed": statement_removed, "scene_merged": scene_merged}

    @staticmethod
    def _normalize_model(model: Dict[str, Any]) -> Dict[str, Any]:
        """清洗并规范化模型结构。"""

        model["overall_style"] = str(model.get("overall_style") or "").strip()
        snapshots = []
        for snapshot in model.get("style_snapshots") or []:
            text = str(snapshot).strip()
            if text and text not in snapshots:
                snapshots.append(text)
        model["style_snapshots"] = snapshots
        if not snapshots and model["overall_style"]:
            model["style_snapshots"] = [model["overall_style"]]
        members = model.get("members") or []
        model["members"] = [
            {
                "user_id": str(m.get("user_id") or "").strip(),
                "name": str(m.get("name") or "未知"),
                "style": str(m.get("style") or "").strip(),
                "personality": str(m.get("personality") or "").strip(),
                "speaking_traits": [str(t) for t in (m.get("speaking_traits") or []) if str(t).strip()],
            }
            for m in members
            if isinstance(m, dict)
        ]
        scenes = []
        for scene in model.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            statements = []
            for index, statement in enumerate(scene.get("statements") or []):
                if not isinstance(statement, dict):
                    continue
                text = str(statement.get("text") or "").strip()
                if not text:
                    continue
                statements.append(
                    {
                        "id": str(statement.get("id") or f"st{index}_{len(statements)}"),
                        "text": text,
                        "speaker": str(statement.get("speaker") or "").strip(),
                        "speaker_name": str(statement.get("speaker_name") or "未知").strip(),
                        "target": str(statement.get("target") or "大家").strip(),
                        "meaning": str(statement.get("meaning") or "").strip(),
                        "ai_note": str(statement.get("ai_note") or "").strip(),
                        "tags": [str(t) for t in (statement.get("tags") or []) if str(t).strip()],
                    }
                )
            scenes.append(
                {
                    "name": str(scene.get("name") or "未命名场景").strip(),
                    "description": str(scene.get("description") or "").strip(),
                    "style": str(scene.get("style") or "").strip(),
                    "note": str(scene.get("note") or "").strip(),
                    "statements": statements,
                }
            )
        model["scenes"] = scenes
        return model

    # ------------------------------------------------------------------
    # 黑名单语义过滤
    # ------------------------------------------------------------------

    @staticmethod
    def _blacklist_applies(item: Dict[str, Any], model_id: str) -> bool:
        """判断黑名单条目是否作用于当前模型。

        规则：条目开了通用拉黑（apply_all）或未限定模型时对所有模型生效；
        否则只对限定的模型生效。
        """

        if bool(item.get("apply_all")):
            return True
        scoped_model = str(item.get("model_id") or "").strip()
        return (not scoped_model) or scoped_model == str(model_id or "").strip()

    @staticmethod
    def _blacklist_key(item: Dict[str, Any]) -> str:
        """黑名单条目的比较文本（语句 + 含义描述）。"""

        text = str(item.get("text") or "").strip()
        meaning = str(item.get("meaning") or "").strip()
        return meaning or text or ""

    @staticmethod
    def _statement_key(statement: Dict[str, Any]) -> str:
        """语句的比较文本（原话 + 备注含义）。"""

        text = str(statement.get("text") or "").strip()
        meaning = str(statement.get("meaning") or "").strip()
        return text if not meaning else f"{text}（{meaning}）"

    def _embedding_fallback_text_similarity(self, text_a: str, text_b: str) -> float:
        """字面相似度（字符二元组 Jaccard），用于无 embedding 时的兜底。"""

        return self._text_similarity(text_a, text_b)

    @staticmethod
    def _cosine(vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个向量的余弦相似度。"""

        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _embed_similarities(
        self, statement_keys: List[str], blacklist_keys: List[str]
    ) -> Optional[List[List[float]]]:
        """用 embedding 计算语句与黑名单的相似度矩阵；失败返回 None。"""

        try:
            black_result = await self._ctx.llm.embed(texts=blacklist_keys, max_concurrent=4)
            black_vectors = [
                entry.get("embedding")
                for entry in black_result.get("results", [])
                if isinstance(entry.get("embedding"), list)
            ]
            statement_result = await self._ctx.llm.embed(texts=statement_keys, max_concurrent=4)
            statement_vectors = [
                entry.get("embedding")
                for entry in statement_result.get("results", [])
                if isinstance(entry.get("embedding"), list)
            ]
            if not black_vectors or len(statement_vectors) != len(statement_keys):
                return None
            matrix = []
            for vector in statement_vectors:
                matrix.append([self._cosine(vector, black) for black in black_vectors])
            return matrix
        except Exception as exc:
            logger.warning("embedding 相似度计算失败，改用 LLM 判定: %s", exc)
            return None

    async def _llm_judge_similar(
        self, statement_keys: List[str], blacklist_keys: List[str]
    ) -> Optional[List[int]]:
        """用 LLM 批量判定哪些语句与黑名单意思高度相似，返回应删除的语句下标。"""

        if not blacklist_keys or not statement_keys:
            return None
        statement_lines = "\n".join(f"{index}. {key}" for index, key in enumerate(statement_keys))
        black_lines = "\n".join(f"- {key}" for key in blacklist_keys)
        prompt = (
            f"以下是黑名单语句（这些意思不允许出现）：\n{black_lines}\n\n"
            f"以下是候选语句：\n{statement_lines}\n\n"
            "请判断哪些候选语句表达的意思与黑名单语句高度相似（意思层面相似，不只是字面相同）。"
            "只输出 JSON 数组，例如 [0, 3]，没有则输出 []。"
        )
        try:
            result = await self._ctx.llm.generate(
                [
                    {"role": "system", "content": "你只输出 JSON 数组，不输出任何其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                model=self._model,
                temperature=0.0,
                rpc_timeout_ms=LLM_RPC_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.warning("LLM 意思判定失败，改用字面相似度: %s", exc)
            return None
        if not result.get("success"):
            return None
        parsed = extract_json(str(result.get("response") or ""))
        if not isinstance(parsed, list):
            return None
        return [int(index) for index in parsed if isinstance(index, (int, float)) and 0 <= int(index) < len(statement_keys)]

    async def filter_statements(
        self,
        statements: List[Dict[str, Any]],
        blacklist_items: List[Dict[str, Any]],
        threshold: float,
        model_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """过滤与黑名单语义高度相似的语句。

        黑名单条目可限定模型（model_id）或开启通用拉黑（apply_all）；
        文本完全相同时直接过滤。相似度优先级：embedding 余弦 → LLM 意思判定 → 字面相似度。
        返回 (保留的语句, 被过滤的语句)。
        """

        if not blacklist_items or not statements:
            return statements, []
        applicable = [item for item in blacklist_items if self._blacklist_applies(item, model_id)]
        if not applicable:
            return statements, []
        statement_keys = [self._statement_key(statement) for statement in statements]
        blacklist_keys = [self._blacklist_key(item) for item in applicable]

        exact_dropped: set[int] = set()
        black_texts = {str(item.get("text") or "").strip() for item in applicable}
        for index, statement in enumerate(statements):
            if str(statement.get("text") or "").strip() in black_texts:
                exact_dropped.add(index)
        if len(exact_dropped) == len(statements):
            dropped = [s for index, s in enumerate(statements) if index in exact_dropped]
            kept = [s for index, s in enumerate(statements) if index not in exact_dropped]
            return kept, dropped

        matrix = await self._embed_similarities(statement_keys, blacklist_keys)
        if matrix is not None:
            kept, dropped = [], []
            for index, statement in enumerate(statements):
                blocked = index in exact_dropped
                if not blocked and matrix[index]:
                    blocked = max(matrix[index]) >= threshold
                if blocked:
                    dropped.append(statement)
                else:
                    kept.append(statement)
            logger.info(
                "黑名单过滤(embedding): 模型=%s 保留 %d 条，过滤 %d 条", model_id, len(kept), len(dropped)
            )
            return kept, dropped
        dropped_indexes = await self._llm_judge_similar(statement_keys, blacklist_keys)
        if dropped_indexes is not None:
            dropped_set = set(dropped_indexes) | exact_dropped
            kept = [s for index, s in enumerate(statements) if index not in dropped_set]
            dropped = [s for index, s in enumerate(statements) if index in dropped_set]
            logger.info("黑名单过滤(LLM): 模型=%s 保留 %d 条，过滤 %d 条", model_id, len(kept), len(dropped))
            return kept, dropped
        kept, dropped = [], []
        for index, statement in enumerate(statements):
            blocked = index in exact_dropped
            if not blocked:
                key = statement_keys[index]
                best = max(self._embedding_fallback_text_similarity(key, black) for black in blacklist_keys)
                blocked = best >= threshold
            if blocked:
                dropped.append(statement)
            else:
                kept.append(statement)
        logger.info("黑名单过滤(字面): 模型=%s 保留 %d 条，过滤 %d 条", model_id, len(kept), len(dropped))
        return kept, dropped
