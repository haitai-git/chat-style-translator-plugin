"""聊天风格转译插件全量测试（分析器与注入器）。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from chat_style_translator_plugin.modules import analyzer as analyzer_mod
from chat_style_translator_plugin.modules.analyzer import StyleAnalyzer
from chat_style_translator_plugin.modules.injector import (
    _member_profiles,
    _speaker_profile,
    build_rewrite_reference,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class FakeLLM:
    """LLM 能力替身：按脚本返回结果。"""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = list(responses or [])

    def set_responses(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        if self.responses:
            return self.responses.pop(0)
        return {"success": True, "response": ""}

    async def embed(self, texts: List[str], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


class TestExtractJson:
    def test_variants(self) -> None:
        assert analyzer_mod.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
        assert analyzer_mod.extract_json('```text\n{"a": 1}\n```') == {"a": 1}
        assert analyzer_mod.extract_json('前缀 {"a": 1} 后缀') == {"a": 1}
        assert analyzer_mod.extract_json('{"a": 1') is None
        assert analyzer_mod.extract_json("") is None
        assert analyzer_mod.extract_json("没有JSON") is None
        # 数组（LLM 黑名单判定返回 [0, 3] 这类数组）
        assert analyzer_mod.extract_json("[0, 3]") == [0, 3]
        assert analyzer_mod.extract_json('前缀 [0, 3] 后缀') == [0, 3]
        assert analyzer_mod.extract_json('```json\n[0, 3]\n```') == [0, 3]
        assert analyzer_mod.extract_json("不是数组") is None
        assert analyzer_mod.extract_json('{"a": 1} [0]') == {"a": 1}  # 对象优先


class TestTranscript:
    def test_format(self) -> None:
        messages = [
            {"sender_name": "小明", "text": "你好", "at_user_name": "", "reply_to_name": ""},
            {"sender_name": "小红", "text": "你好呀", "at_user_name": "", "reply_to_name": "小明"},
            {"sender_name": "", "text": "  ", "at_user_name": "", "reply_to_name": ""},
            {"sender_name": "小刚", "text": "在吗", "at_user_name": "小红", "reply_to_name": ""},
        ]
        transcript = StyleAnalyzer.build_transcript(messages)
        assert "1. 小明: 你好" in transcript
        assert "2. 小红 对「小明」说: 你好呀" in transcript
        assert "小刚 对「小红」说: 在吗" in transcript
        assert "未知" not in transcript


class TestMergePrompt:
    def test_options(self) -> None:
        analyzer = StyleAnalyzer(None)
        analyzer.update_config(max_statements_per_scene=20)
        prompt = analyzer._merge_system_prompt(
            {"ratio": (70, 30), "dedup_level": "激进精简", "max_total_statements": 120}
        )
        assert "来源A:来源B = 70:30" in prompt
        assert "激进精简" in prompt
        assert "不超过 120 条" in prompt
        assert "每个场景最终保留不超过" not in prompt
        analysis = analyzer._merge_system_prompt({"analysis_scene_limit": True})
        assert "不超过 20 条最有代表性的语句" in analysis
        std = analyzer._merge_system_prompt(None)
        assert "来源A:来源B" not in std and "激进精简" not in std and "不超过 120 条" not in std
        assert "语句要去重：完全重复或含义高度相似的语句只保留一条" in std
        assert "整体风格（overall_style）不合并" in std  # 不合并整体风格，各来源分别存放
        off = analyzer._merge_system_prompt({"dedup_level": "不去重"})
        assert "语句不去重" in off

    def test_merge_partials_single(self) -> None:
        analyzer = StyleAnalyzer(None)
        partial = {"overall_style": "s", "members": [], "scenes": []}
        merged = run(analyzer.merge_partials([partial], name="单模型"))
        assert merged is not None
        assert merged["name"] == "单模型"
        assert merged is not partial  # 深拷贝
        assert merged["style_snapshots"] == ["s"]

    def test_merge_partials_multi_and_fallback(self) -> None:
        llm = FakeLLM([
            {"success": True, "response": '{"overall_style": "合并", "members": [], "scenes": []}'}
        ])
        analyzer = StyleAnalyzer(type("Ctx", (), {"llm": llm})())
        partial_a = {"overall_style": "a", "members": [], "scenes": []}
        partial_b = {"overall_style": "b", "members": [], "scenes": []}
        merged = run(analyzer.merge_partials([partial_a, partial_b], name="AB"))
        # 整体风格不合并：按来源分别存放
        assert merged["overall_style"] == "合并" and merged["name"] == "AB"
        assert merged["style_snapshots"] == ["a", "b"]
        llm.set_responses([{"success": True, "response": "不是JSON"}])
        merged2 = run(analyzer.merge_partials([partial_a, partial_b], name="AB"))
        # 简单拼接兜底：保留全部窗口内容，整体风格仍分开存放
        assert merged2["overall_style"] == ""
        assert merged2["style_snapshots"] == ["a", "b"]
        llm.set_responses([{"success": False, "error": "boom"}])
        merged3 = run(analyzer.merge_partials([partial_a, partial_b]))
        assert merged3["style_snapshots"] == ["a", "b"]

    def test_collect_style_snapshots(self) -> None:
        """快照去重保序；带 style_snapshots 的旧模型再融合时继续展开。"""
        partial_a = {"overall_style": "a", "members": [], "scenes": []}
        partial_b = {"overall_style": "a", "style_snapshots": ["a", "c"], "members": [], "scenes": []}
        snapshots = StyleAnalyzer._collect_style_snapshots([partial_a, partial_b])
        assert snapshots == ["a", "c"]
        assert StyleAnalyzer._collect_style_snapshots([{}, partial_a]) == ["a"]

    def test_merge_similar_snapshots(self) -> None:
        """整体风格快照：相似的合并为一条，差异大的错开放。"""
        snapshots = [
            "群聊气氛轻松，成员爱玩梗，常用表情包和网络黑话。",
            "群聊气氛轻松，成员爱玩梗，常用表情包和网络黑话，偶尔开车。",
            "深夜场话题非常严肃，全程讨论工作与学术问题。",
        ]
        merged = StyleAnalyzer._merge_similar_snapshots(snapshots, threshold=0.5)
        assert len(merged) == 2  # 前两条相似合并，第三条跨度大保留
        assert merged[0] == snapshots[1]  # 保留更长更完整的一条
        assert merged[1] == snapshots[2]
        # 阈值调高后都不合并
        assert len(StyleAnalyzer._merge_similar_snapshots(snapshots, threshold=0.99)) == 3

    def test_merge_similar_scenes(self) -> None:
        """相似场景合并（语句去重），跨度大的场景错开保留。"""
        scenes = [
            {"name": "监狱保释梗", "description": "", "style": "玩梗", "note": "",
             "statements": [{"text": "什么时候保释", "speaker": "1"}, {"text": "捞人", "speaker": "2"}]},
            {"name": "监狱/保释梗", "description": "", "style": "玩梗", "note": "",
             "statements": [{"text": "捞人", "speaker": "2"}, {"text": "保释金多少", "speaker": "3"}]},
            {"name": "开车", "description": "", "style": "直白", "note": "",
             "statements": [{"text": "这车开得飞快", "speaker": "4"}]},
        ]
        merged = StyleAnalyzer._merge_similar_scenes(scenes, threshold=0.6)
        assert len(merged) == 2  # 前两个相似场景合并，开车跨度大保留
        first = next(s for s in merged if "保释" in str(s.get("name") or ""))
        texts = {st["text"] for st in first["statements"]}
        assert texts == {"保释金多少", "什么时候保释", "捞人"}  # 语句合并去重
        assert any(s["name"] == "开车" for s in merged)

    def test_merge_similar_scenes_far_apart(self) -> None:
        """跨度大的场景不合并。"""
        scenes = [
            {"name": "表情包接龙", "statements": [{"text": "哈哈哈哈哈"}]},
            {"name": "游戏资讯爆料", "statements": [{"text": "新赛季爆料"}]},
        ]
        merged = StyleAnalyzer._merge_similar_scenes(scenes, threshold=0.6)
        assert len(merged) == 2

    def test_merge_similar_scenes_name_containment(self) -> None:
        """场景名互相包含时必定合并（与阈值、来源/时段无关），如「闲聊」与「闲聊吹水」。"""
        scenes = [
            {"name": "闲聊", "statements": [{"text": "今天天气不错"}]},
            {"name": "闲聊吹水", "statements": [{"text": "今天吃什么"}]},
        ]
        merged = StyleAnalyzer._merge_similar_scenes(scenes, threshold=0.99)
        assert len(merged) == 1

    def test_merge_similar_scenes_statement_overlap(self) -> None:
        """语句原文大面积重复的场景必定合并（与阈值、来源/时段无关）。"""
        scenes = [
            {"name": "深夜emo", "statements": [{"text": "好烦", "meaning": "吐槽"}]},
            {"name": "深夜emo第二弹", "statements": [{"text": "好烦", "meaning": "吐槽"}]},
        ]
        merged = StyleAnalyzer._merge_similar_scenes(scenes, threshold=0.99)
        assert len(merged) == 1
        merged_scene = merged[0]
        assert "深夜emo" in str(merged_scene.get("name") or "")  # 保留更长更完整的场景名
        assert len(merged_scene["statements"]) == 1  # 语句去重

    def test_merge_similar_scenes_short_name_no_false_merge(self) -> None:
        """单字/不含包含关系的场景名不误合并。"""
        scenes = [
            {"name": "车", "statements": [{"text": "滴滴"}]},
            {"name": "开车", "statements": [{"text": "这车开得飞快"}]},
        ]
        merged = StyleAnalyzer._merge_similar_scenes(scenes, threshold=0.99)
        assert len(merged) == 2  # 单字名不参与包含判定

    def test_merge_similar_scenes_overlap_no_false_merge(self) -> None:
        """语句重叠不足半数不合并，防误合并。"""
        scenes = [
            {"name": "深夜畅聊", "statements": [{"text": "在吗"}, {"text": "好烦"}, {"text": "睡不着"}]},
            {"name": "打招呼", "statements": [{"text": "在吗"}]},
        ]
        merged = StyleAnalyzer._merge_similar_scenes(scenes, threshold=0.99)
        assert len(merged) == 2  # 1/3 重叠 = 33% < 50%，不合并
        scenes2 = [
            {"name": "深夜畅聊", "statements": [{"text": "在吗"}, {"text": "好烦"}]},
            {"name": "打招呼", "statements": [{"text": "在吗"}]},
        ]
        merged2 = StyleAnalyzer._merge_similar_scenes(scenes2, threshold=0.99)
        assert len(merged2) == 2  # 恰好 50%（1/2）也不合并，只合并过半的场景
        scenes3 = [
            {"name": "深夜畅聊", "statements": [{"text": "在吗"}, {"text": "好烦"}]},
            {"name": "打招呼", "statements": [{"text": "在吗"}, {"text": "好烦"}]},
        ]
        merged3 = StyleAnalyzer._merge_similar_scenes(scenes3, threshold=0.99)
        assert len(merged3) == 1  # 完全一致（100%）合并

    def test_normalize_model(self) -> None:
        model = {
            "overall_style": "  风格  ",
            "style_snapshots": ["  快照1  ", "", "快照2", "快照1"],
            "members": [
                {"user_id": "1", "name": "A", "style": "s", "personality": "p", "speaking_traits": ["t1", "  "]},
            ],
            "scenes": [
                {
                    "name": "场景", "description": "d", "style": "s", "note": "n",
                    "statements": [
                        {"text": " 原话 ", "speaker": "1", "speaker_name": "A", "target": "大家",
                         "meaning": "m", "ai_note": "n", "tags": ["t", "  "]},
                        {"text": "", "speaker": "1"},
                        "非法语句",
                    ],
                },
                "非法场景",
            ],
        }
        normalized = StyleAnalyzer._normalize_model(model)
        assert normalized["overall_style"] == "风格"
        assert normalized["style_snapshots"] == ["快照1", "快照2"]  # 去空去重
        assert normalized["members"][0]["speaking_traits"] == ["t1"]
        scenes = normalized["scenes"]
        assert len(scenes) == 1
        statement = scenes[0]["statements"][0]
        assert len(scenes[0]["statements"]) == 1
        assert statement["text"] == "原话"
        assert statement["id"].startswith("st")
        assert statement["tags"] == ["t"]


class TestCompactModel:
    def make_model(self) -> Dict[str, Any]:
        return {
            "name": "大模型",
            "overall_style": "群聊气氛轻松，成员爱玩梗。",
            "style_snapshots": [
                "群聊气氛轻松，成员爱玩梗。",
                "群聊气氛轻松，成员爱玩梗，偶尔开车。",
                "深夜场话题非常严肃。",
            ],
            "members": [],
            "scenes": [
                {
                    "name": "监狱保释梗",
                    "style": "玩梗",
                    "statements": [
                        {"text": "什么时候保释", "speaker": "1", "meaning": "问时间"},
                        {"text": "什么时候保释", "speaker": "2", "meaning": ""},
                    ],
                },
                {
                    "name": "监狱/保释梗",
                    "style": "玩梗",
                    "statements": [
                        {"text": "保释金多少", "speaker": "3"},
                        {"text": "保释金多少钱", "speaker": "4"},
                    ],
                },
                {
                    "name": "哈哈哈接龙",
                    "style": "搞笑",
                    "statements": [
                        {"text": "哈哈哈哈哈哈", "speaker": "5"},
                        {"text": "哈哈哈哈哈", "speaker": "6"},
                    ],
                },
                {
                    "name": "游戏开黑",
                    "style": "直白",
                    "statements": [{"text": "来一把吗", "speaker": "7"}],
                },
            ],
        }

    def test_standard_dedup(self) -> None:
        analyzer = StyleAnalyzer(None)
        compacted, stats = analyzer.compact_model(
            self.make_model(), name="精简版", options={"dedup_level": "标准"}
        )
        assert compacted["name"] == "精简版"
        assert stats["statement_removed"] == 1  # 只去完全重复
        assert stats["scene_merged"] == 1  # 监狱保释梗 ×2 场景合并
        assert len(compacted["scenes"]) == 3
        assert len(compacted["style_snapshots"]) == 2  # 相似快照合并，跨度大的保留
        # 完全重复只保留一条，近似句保留（标准不去近似）
        texts = {st["text"] for scene in compacted["scenes"] for st in scene["statements"]}
        assert "什么时候保释" in texts and "哈哈哈哈哈哈" in texts and "哈哈哈哈哈" in texts

    def test_aggressive_dedup(self) -> None:
        analyzer = StyleAnalyzer(None)
        compacted, stats = analyzer.compact_model(
            self.make_model(), options={"dedup_level": "激进精简"}
        )
        # 完全重复 1 条 + 近似句 2 条（保释金多少/多少钱、哈哈哈哈哈哈/哈哈哈哈哈）
        assert stats["statement_removed"] == 3
        texts = {st["text"] for scene in compacted["scenes"] for st in scene["statements"]}
        assert "哈哈哈哈哈" not in texts  # 近似句合并为一条

    def test_no_dedup_keeps_statements(self) -> None:
        analyzer = StyleAnalyzer(None)
        compacted, stats = analyzer.compact_model(
            self.make_model(), options={"dedup_level": "不去重"}
        )
        assert stats["statement_removed"] == 0
        assert stats["scene_merged"] == 1  # 相似场景仍合并
        assert len(compacted["scenes"]) == 3

    def test_cap_total_statements(self) -> None:
        analyzer = StyleAnalyzer(None)
        compacted, stats = analyzer.compact_model(
            self.make_model(), options={"dedup_level": "标准", "max_total_statements": 3}
        )
        total = sum(len(scene["statements"]) for scene in compacted["scenes"])
        assert total <= 3
        texts = {st["text"] for scene in compacted["scenes"] for st in scene["statements"]}
        assert "来一把吗" not in texts  # 最短的优先被砍
        assert stats["statement_removed"] >= 3

    def test_input_unchanged(self) -> None:
        analyzer = StyleAnalyzer(None)
        model = self.make_model()
        before = json.dumps(model, ensure_ascii=False)
        analyzer.compact_model(model)
        assert json.dumps(model, ensure_ascii=False) == before  # 原模型不被修改


class TestBlacklistFilter:
    def test_helpers(self) -> None:
        assert StyleAnalyzer._blacklist_applies({"apply_all": True, "model_id": ""}, "m1") is True
        assert StyleAnalyzer._blacklist_applies({"apply_all": False, "model_id": ""}, "m1") is True
        assert StyleAnalyzer._blacklist_applies({"apply_all": False, "model_id": "m1"}, "m1") is True
        assert StyleAnalyzer._blacklist_applies({"apply_all": False, "model_id": "m1"}, "m2") is False
        assert StyleAnalyzer._blacklist_key({"text": "原话", "meaning": "含义"}) == "含义"
        assert StyleAnalyzer._blacklist_key({"text": "原话", "meaning": ""}) == "原话"
        assert StyleAnalyzer._statement_key({"text": "原话", "meaning": "含义"}) == "原话（含义）"
        assert StyleAnalyzer._statement_key({"text": "原话", "meaning": ""}) == "原话"

    def test_similarity_scoring(self) -> None:
        analyzer = StyleAnalyzer(None)
        assert analyzer._embedding_fallback_text_similarity("修仙打坐", "修仙打坐") == 1.0
        assert analyzer._embedding_fallback_text_similarity("修仙", "打坐") < 0.5
        assert analyzer._embedding_fallback_text_similarity("a", "b") == 0.0
        assert StyleAnalyzer._cosine([1, 0], [1, 0]) == 1.0
        assert StyleAnalyzer._cosine([1, 0], [0, 1]) == 0.0
        assert StyleAnalyzer._cosine([], []) == 0.0

    def test_exact_match(self) -> None:
        analyzer = StyleAnalyzer(None)
        statements = [{"text": "你完了"}, {"text": "下次再约"}]
        blacklist = [{"text": "你完了", "meaning": "", "apply_all": True, "model_id": ""}]
        kept, dropped = run(analyzer.filter_statements(statements, blacklist, 0.85))
        assert [s["text"] for s in kept] == ["下次再约"]
        assert [s["text"] for s in dropped] == ["你完了"]
        assert run(analyzer.filter_statements(statements, [], 0.85)) == (statements, [])
        assert run(analyzer.filter_statements([], blacklist, 0.85)) == ([], [])
        scoped = [{"text": "你完了", "meaning": "", "apply_all": False, "model_id": "m9"}]
        assert run(analyzer.filter_statements(statements, scoped, 0.85, model_id="m1")) == (statements, [])

    def test_embedding_path(self) -> None:
        class EmbedLLM(FakeLLM):
            def __init__(self) -> None:
                super().__init__()
                self.embed_calls = 0

            async def embed(self, texts: List[str], **kwargs: Any) -> dict[str, Any]:
                self.embed_calls += 1
                if len(texts) == 1:  # 第一次调用：黑名单向量
                    return {"results": [{"embedding": [1.0, 0.0]}]}
                # 第二次调用：语句向量（第一条与黑名单相似，第二条不相似）
                return {
                    "results": [
                        {"embedding": [1.0, 0.0]},
                        {"embedding": [0.0, 1.0]},
                    ]
                }

        llm = EmbedLLM()
        analyzer = StyleAnalyzer(type("Ctx", (), {"llm": llm})())
        statements = [{"text": "你这个菜鸡"}, {"text": "哈哈哈哈哈"}]
        blacklist = [{"text": "骂人是菜", "meaning": "表达嘲讽辱骂", "apply_all": True, "model_id": ""}]
        kept, dropped = run(analyzer.filter_statements(statements, blacklist, 0.95))
        assert llm.embed_calls == 2
        assert [s["text"] for s in dropped] == ["你这个菜鸡"]
        assert [s["text"] for s in kept] == ["哈哈哈哈哈"]

    def test_llm_path(self) -> None:
        class JudgeLLM(FakeLLM):
            async def embed(self, texts: List[str], **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("embedding 不可用")

            async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
                return {"success": True, "response": "[0]"}

        analyzer = StyleAnalyzer(type("Ctx", (), {"llm": JudgeLLM()})())
        statements = [{"text": "A句"}, {"text": "B句"}, {"text": "C句"}]
        blacklist = [{"text": "黑句", "meaning": "", "apply_all": True, "model_id": ""}]
        kept, dropped = run(analyzer.filter_statements(statements, blacklist, 0.85))
        assert [s["text"] for s in dropped] == ["A句"]
        assert len(kept) == 2

    def test_text_fallback(self) -> None:
        class NoEmbedLLM(FakeLLM):
            async def embed(self, texts: List[str], **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("no embed")

            async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
                return {"success": True, "response": "不是数组"}

        analyzer = StyleAnalyzer(type("Ctx", (), {"llm": NoEmbedLLM()})())
        statements = [{"text": "完全相同的句子"}, {"text": "完全相同的句子"}, {"text": "完全不同的话"}]
        blacklist = [{"text": "完全相同的句子", "meaning": "", "apply_all": True, "model_id": ""}]
        kept, dropped = run(analyzer.filter_statements(statements, blacklist, 0.85))
        assert [s["text"] for s in dropped] == ["完全相同的句子", "完全相同的句子"]
        assert len(kept) == 1


class TestInjector:
    def make_model(self) -> dict[str, Any]:
        return {
            "name": "群A风格模型",
            "overall_style": "简洁直接",
            "members": [
                {"user_id": "100", "name": "小明", "style": "爱用语气词", "personality": "开朗", "speaking_traits": ["哈哈"]},
                {"user_id": "", "name": "无号群友", "style": "s", "personality": "", "speaking_traits": []},
            ],
            "scenes": [
                {
                    "name": "打招呼",
                    "description": "日常打招呼",
                    "style": "热情",
                    "note": "早上常用",
                    "statements": [
                        {"text": "早啊兄弟", "speaker": "100", "speaker_name": "小明", "target": "大家",
                         "meaning": "问好", "ai_note": "亲切", "tags": []},
                        {"text": "早", "speaker": "101", "speaker_name": "小红", "target": "大家",
                         "meaning": "问好", "ai_note": "简短", "tags": []},
                    ],
                }
            ],
        }

    def test_empty_model(self) -> None:
        assert build_rewrite_reference({"name": "x", "overall_style": "", "scenes": [], "members": []}) == ""

    def test_render_content(self) -> None:
        ref = build_rewrite_reference(self.make_model(), max_chars=2500)
        assert "【群聊风格：群A风格模型】" in ref
        assert "整体风格：简洁直接" in ref
        assert "小明" in ref and "爱用语气词" in ref and "哈哈" in ref
        assert "【场景：打招呼】" in ref and "场景备注：早上常用" in ref
        assert "早啊兄弟" in ref and "小明(QQ 100)对大家说" in ref

    def test_render_snapshots_separated(self) -> None:
        """整体风格快照分开渲染，不合并成一段。"""
        model = self.make_model()
        model["style_snapshots"] = ["第一段时间风格", "第二段时间风格"]
        ref = build_rewrite_reference(model, max_chars=2500)
        assert "整体风格快照1：第一段时间风格" in ref
        assert "整体风格快照2：第二段时间风格" in ref
        assert "整体风格：简洁直接" not in ref  # 有快照时不使用合并后的整体风格

    def test_speaker_matching(self) -> None:
        model = self.make_model()
        ref = build_rewrite_reference(model, max_chars=2500, speaker={"user_id": "100", "name": "小明"}, enable_matching=True)
        assert "当前说话人 QQ 100" in ref
        ref2 = build_rewrite_reference(model, max_chars=2500, speaker={"user_id": "999", "name": "路人"}, enable_matching=True)
        assert "模型档案中暂无此人的风格" in ref2
        ref3 = build_rewrite_reference(model, max_chars=2500, speaker={"user_id": "100", "name": "小明"}, enable_matching=False)
        assert "当前说话人" not in ref3

    def test_truncation(self) -> None:
        model = self.make_model()
        full = build_rewrite_reference(model, max_chars=2500)
        short = build_rewrite_reference(model, max_chars=60)
        # 小预算下：非空、不超预算、内容被压缩
        assert 0 < len(short) <= 60
        assert len(short) < len(full)
        # 极小的预算至少保留第一段（首段永远放得下）
        assert build_rewrite_reference(model, max_chars=1) != ""

    def test_profiles(self) -> None:
        members = [{"user_id": "1", "name": "A", "style": "s", "personality": "p", "speaking_traits": ["t"]}]
        assert "QQ 1（A）" in _member_profiles(members)
        assert "QQ 1（B）" in _speaker_profile({"user_id": "1", "name": "B"}, members)
        assert "暂无此人的风格" in _speaker_profile({"user_id": "9", "name": "X"}, members)
        assert "未知" in _speaker_profile({}, members)
