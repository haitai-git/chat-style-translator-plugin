"""聊天风格转译插件：把风格模型转成回复后轻度微调用的风格参考文本。

微调思路：
1. 不再把风格模型注入 AI 的提示词（避免改变 AI 自身的行为风格）；
2. 每次 AI 生成回复草稿后，把风格模型渲染成「参考文本」交给 LLM 做轻度微调：
   只改几个词，原意严格不变；
3. 参考文本包含模型整体风格、群友档案（按 QQ 号区分）与场景语句，不含任何行为规则。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional

from .retrieval import scene_terms as _scene_terms
from .style_profile import contextual_style, observed_group_style, observed_style


def _member_profiles(members: List[Dict[str, Any]]) -> str:
    """把群友档案渲染成带 QQ 号的文本。"""

    lines = []
    for member in members[:8]:
        user_id = str(member.get("user_id") or "").strip()
        name = str(member.get("name") or "未知")
        personality = str(member.get("personality") or "").strip()
        style = str(member.get("style") or "").strip()
        traits = [str(t) for t in (member.get("speaking_traits") or []) if str(t).strip()]
        details = []
        if personality:
            details.append(f"默认性格：{personality}")
        if style:
            details.append(f"聊天风格：{style}")
        if traits:
            details.append("特点：" + "、".join(traits))
        detail_text = "；".join(details)
        if user_id:
            line = f"- QQ {user_id}（{name}）"
        else:
            line = f"- {name}"
        if detail_text:
            line += f"：{detail_text}"
        lines.append(line)
    return "\n".join(lines)


def _speaker_profile(
    speaker: Optional[Dict[str, Any]], members: List[Dict[str, Any]]
) -> str:
    """生成当前说话人的风格档案块；模型里没有此人档案时给出提示。"""

    user_id = str((speaker or {}).get("user_id") or "").strip()
    sname = str((speaker or {}).get("name") or "未知")
    matched = None
    if user_id:
        matched = next(
            (m for m in members if str(m.get("user_id") or "").strip() == user_id),
            None,
        )
    if matched:
        bits = []
        style = str(matched.get("style") or "").strip()
        personality = str(matched.get("personality") or "").strip()
        if style:
            bits.append(f"个人聊天风格：{style}")
        if personality:
            bits.append(f"默认性格：{personality}")
        if bits:
            return f"当前说话人 QQ {user_id}（{sname}）的风格：{'；'.join(bits)}"
        return f"当前说话人 QQ {user_id}（{sname}）：（档案暂无具体风格描述）"
    if user_id:
        return f"当前说话人 QQ {user_id}（{sname}）：（模型档案中暂无此人的风格，可参考群整体风格）"
    return "当前说话人：（未知，可参考群整体风格）"


def _example_relevance(statement: Dict[str, Any], incoming_terms: set[str]) -> int:
    text = statement.get("context")
    return len(incoming_terms & _scene_terms(text)) if isinstance(text, str) and incoming_terms else 0


def _scene_relevance(
    scene: Dict[str, Any], context_terms: set[str], incoming_terms: set[str]
) -> int:
    topic = " ".join(str(scene.get(field) or "") for field in ("name", "description", "note"))
    statements = scene.get("statements") or []
    examples = " ".join(
        str(statement.get(field) or "")
        for statement in statements[:12]
        if isinstance(statement, dict)
        for field in ("text", "meaning")
    )
    matched_context = max(
        (_example_relevance(st, incoming_terms) for st in statements if isinstance(st, dict)),
        default=0,
    )
    return 3 * len(context_terms & _scene_terms(topic)) + len(context_terms & _scene_terms(examples)) + 4 * matched_context


def _needs_contextual_example(
    statement: Dict[str, Any], scene_description: str, incoming_terms: set[str]
) -> bool:
    prior = statement.get("context")
    if not isinstance(prior, str) or not prior.strip() or not statement.get("text"):
        return False
    prior_terms = _scene_terms(prior)
    overlap = len(incoming_terms & prior_terms)
    if overlap < 3 or overlap < 0.55 * max(len(incoming_terms), len(prior_terms)):
        return False
    usage = str(statement.get("ai_note") or "")
    return contextual_style(f"{scene_description} {usage}")


def _scene_style_lines(scene_style: str, statements: list[dict[str, Any]], limit: int | None) -> list[str]:
    lines = []
    if observed := observed_style(statements):
        lines.append(f"表达习惯：{observed}")
    if scene_style:
        lines.append(f"场景风格：{scene_style[:limit]}")
    return lines


def _best_contextual_example(
    statements: list[dict[str, Any]], signals: str, incoming_terms: set[str],
) -> dict[str, Any] | None:
    return max(
        (item for item in statements if _needs_contextual_example(item, signals, incoming_terms)),
        key=lambda item: _example_relevance(item, incoming_terms),
        default=None,
    )


def build_ranked_scene_reference(
    candidates: Iterable[tuple[int, str, dict[str, Any], str]],
    max_chars: int,
    *,
    incoming_text: str = "",
    random_fallback: bool = False,
    visible_examples: set[str],
) -> tuple[str, list[tuple[int, str, str]]]:
    parts: list[str] = []
    included: list[tuple[int, str, str]] = []
    used = 0
    incoming_terms = _scene_terms(incoming_text) if not random_fallback else set()
    for model_index, model_name, scene, label in candidates:
        room = max_chars - used - (2 if parts else 0)
        if room <= 0:
            break

        name = " ".join(str(scene.get("name") or "未命名场景").split())[:100]
        style = str(scene.get("style") or "").strip()
        statements = [item for item in scene.get("statements") or [] if isinstance(item, dict)]
        evidence = _scene_style_lines(style, statements, 240)
        if random_fallback and not evidence:
            continue
        if random_fallback:
            heading = f"【随机场景表达参考 {len(parts) + 1}】"
        else:
            heading = f"【场景：{name}】"
        if len(heading) > room:
            break
        block = "\n".join([heading, *evidence])
        if len(block) > room:
            if parts:
                break
            block = block[:room].rstrip()

        if not random_fallback and incoming_terms and not visible_examples:
            signals = f"{name} {style} {scene.get('description') or ''} {scene.get('note') or ''}"
            if example := _best_contextual_example(statements, signals, incoming_terms):
                prior = str(example["context"]).strip()
                text = str(example["text"]).strip()
                line = f"\n代表性语句：\n- 前文「{prior}」 → 群友说：“{text}”"
                if text and len(block) + len(line) <= room:
                    block += line
                    visible_examples.add(text)

        if not random_fallback and model_name:
            source = "\n风格模型：" + " ".join(model_name.split())[:100]
            if len(block) + len(source) <= room:
                block += source
        parts.append(block)
        included.append((model_index, name, label))
        used += len(block) + (2 if len(parts) > 1 else 0)
    return "\n\n".join(parts), included


def build_rewrite_reference(
    model: Dict[str, Any],
    max_chars: int = 2500,
    speaker: Optional[Dict[str, Any]] = None,
    enable_matching: bool = True,
    *,
    context: str = "",
    incoming_text: Optional[str] = None,
    visible_examples: Optional[set[str]] = None,
    style_only: bool = False,
    ranked_scenes: Optional[List[Dict[str, Any]]] = None,
    group_observations: Optional[str] = None,
) -> str:
    """把一份风格模型渲染成「轻度微调」的参考文本（纯风格资料，不含行为规则）。

    Args:
        model: 风格模型字典。
        max_chars: 参考文本最大字符数，超长时按场景/语句截断。
        speaker: 当前说话人信息（user_id/name），用于优先参考其个人风格。
        enable_matching: 是否优先参考当前说话人的个人风格。
        context: 用于匹配蒸馏场景的聊天上下文和回复草稿。
        incoming_text: 当前正在回复的原话，用于检索范例前文。
        visible_examples: 收集完整送入提示词的范例原话。
        style_only: 提供语言风格证据。语境高度贴合的特殊场景可附一条前文和原话。
        ranked_scenes: 检索已选出的场景，空列表表示参考整体风格。
        group_observations: 已计算的群聊表达习惯。

    Returns:
        str: 参考文本；模型没有可用内容时返回空字符串。
    """

    name = str(model.get("name") or "群聊风格模型")
    overall_style = str(model.get("overall_style") or "").strip()
    summary = str(model.get("style_summary") or "").strip()
    snapshots = [str(s).strip() for s in (model.get("style_snapshots") or []) if str(s or "").strip()]
    scenes = [s for s in model.get("scenes") or [] if isinstance(s, dict)]
    members = [m for m in model.get("members") or [] if isinstance(m, dict)]
    model_scenes = scenes
    if not (
        summary or overall_style or snapshots or scenes or model.get("scene_count")
        or group_observations or (enable_matching and members)
    ):
        return ""

    incoming_terms = _scene_terms(context if incoming_text is None else incoming_text) if ranked_scenes != [] else set()
    if ranked_scenes is not None:
        scenes = ranked_scenes
        matched = bool(scenes)
    else:
        context_terms = _scene_terms(context) | incoming_terms
        scored = [
            (_scene_relevance(scene, context_terms, incoming_terms), scene) for scene in scenes
        ] if context_terms else []
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            scenes = [scene for _, scene in scored]
        matched = bool(scored and scored[0][0] > 0)
        if style_only:
            scenes = [scene for score, scene in scored[:2] if score > 0]
    parts: List[str] = []
    parts.append(f"【群聊风格：{name}】")
    if summary:
        parts.append(f"整体风格摘要：{summary}")
    elif snapshots:
        style_lines = []
        for index, snapshot in enumerate(snapshots, start=1):
            style_lines.append(f"整体风格快照{index}：{snapshot}")
        if style_only:
            parts.extend(reversed(style_lines))
        else:
            parts.append("\n".join(style_lines))
    elif overall_style:
        parts.append(f"整体风格：{overall_style}")

    if style_only:
        if not scenes:
            observed = group_observations
            if observed is None:
                observed = observed_group_style(model_scenes)
            if observed:
                parts.insert(1, f"表达习惯：{observed}")
        if enable_matching and speaker:
            user_id = str(speaker.get("user_id") or "").strip()
            profile = next((member for member in members if user_id and str(member.get("user_id") or "") == user_id), None)
            if profile:
                traits = profile.get("speaking_traits") or []
                style = str(profile.get("style") or "").strip()
                details = "、".join(str(trait) for trait in traits[:4] if str(trait).strip())
                profile_style = "，".join(value for value in (style, details) if value)
                if profile_style:
                    parts.append(f"当前说话人的表达风格：{profile_style[:180]}")
    elif enable_matching:
        if speaker:
            parts.append(_speaker_profile(speaker, members))
        member_lines = _member_profiles(members)
        if member_lines:
            parts.append("群友档案（按 QQ 号区分）:\n" + member_lines)

    scene_sections: List[str] = []
    rendered_examples: set[str] = set()
    for scene_index, scene in enumerate(scenes):
        scene_name = str(scene.get("name") or "未命名场景")
        scene_style = str(scene.get("style") or "").strip()
        description = str(scene.get("description") or "").strip()
        note = str(scene.get("note") or "").strip()
        lines = [f"【场景：{scene_name}】"]
        details = []
        for label, text in (("场景风格", scene_style), ("场景备注", note), ("场景描述", description)):
            if text and (not style_only or label == "场景风格"):
                details.append(f"{label}：{text[:240] if ranked_scenes is not None else text}")
        if ranked_scenes is None and not style_only:
            lines.extend(details)
        statements = [s for s in scene.get("statements") or [] if isinstance(s, dict)]
        if incoming_terms and not style_only:
            statements.sort(key=lambda item: _example_relevance(item, incoming_terms), reverse=True)
        examples = statements[:4]
        if style_only:
            lines.extend(_scene_style_lines(scene_style, statements, 240 if ranked_scenes is not None else None))
            examples = []
            if matched and scene_index == 0 and not visible_examples:
                signals = f"{scene_name} {scene_style} {description} {note}"
                if example := _best_contextual_example(statements, signals, incoming_terms):
                    examples.append(example)
        if examples:
            statement_lines = []
            for statement in examples:
                text = str(statement.get("text") or "").strip()
                if not text:
                    continue
                speaker_name = str(statement.get("speaker_name") or "某群友")
                speaker_id = str(statement.get("speaker") or "").strip()
                target = str(statement.get("target") or "大家")
                speaker_label = f"{speaker_name}(QQ {speaker_id})" if speaker_id else speaker_name
                prefix = (
                    f"群友说：“{text}”" if style_only
                    else f"{speaker_label}对{target}说：“{text}”"
                )
                prior = statement.get("context")
                if isinstance(prior, str) and prior.strip():
                    line = f"- 前文「{prior.strip()}」 → {prefix}"
                else:
                    line = f"- {prefix}"
                rendered_examples.add(text)
                meaning = str(statement.get("meaning") or "").strip()
                usage = str(statement.get("ai_note") or "").strip()
                if not style_only:
                    if meaning:
                        line += f"（意思：{meaning}）"
                    if usage:
                        line += f"（用法：{usage}）"
                statement_lines.append(line)
            if statement_lines:
                lines.append("代表性语句：\n" + "\n".join(statement_lines))
        if ranked_scenes is not None and not style_only:
            lines.extend(details)
        scene_sections.append("\n".join(lines))

    if scene_sections:
        scene_sections[0] = "场景与语句：\n" + scene_sections[0]
        if matched:
            first = scene_sections.pop(0)
            parts.insert(1, first)
        parts.extend(scene_sections)

    selected = parts[:2] if matched else parts[:1]
    used = len("\n\n".join(selected))
    for part in parts[len(selected) :]:
        if used + 2 + len(part) <= max_chars:
            selected.append(part)
            used += 2 + len(part)
    result = "\n\n".join(selected)[:max_chars].rstrip()
    if visible_examples is not None:
        visible_examples.update(text for text in rendered_examples if f'说：“{text}”' in result)
    return result
