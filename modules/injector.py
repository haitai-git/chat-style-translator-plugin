"""聊天风格转译插件：把风格模型转成回复后轻度微调用的风格参考文本。

微调思路：
1. 不再把风格模型注入 AI 的提示词（避免改变 AI 自身的行为风格）；
2. 每次 AI 生成回复草稿后，把风格模型渲染成「参考文本」交给 LLM 做轻度微调：
   只改几个词，原意严格不变；
3. 参考文本包含模型整体风格、群友档案（按 QQ 号区分）与场景语句，不含任何行为规则。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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


def build_rewrite_reference(
    model: Dict[str, Any],
    max_chars: int = 2500,
    speaker: Optional[Dict[str, Any]] = None,
    enable_matching: bool = True,
) -> str:
    """把一份风格模型渲染成「轻度微调」的参考文本（纯风格资料，不含行为规则）。

    Args:
        model: 风格模型字典。
        max_chars: 参考文本最大字符数，超长时按场景/语句截断。
        speaker: 当前说话人信息（user_id/name），用于优先参考其个人风格。
        enable_matching: 是否优先参考当前说话人的个人风格；关闭时只按群整体风格。

    Returns:
        str: 参考文本；模型没有可用内容时返回空字符串。
    """

    name = str(model.get("name") or "群聊风格模型")
    overall_style = str(model.get("overall_style") or "").strip()
    snapshots = [str(s).strip() for s in (model.get("style_snapshots") or []) if str(s or "").strip()]
    scenes = [s for s in model.get("scenes") or [] if isinstance(s, dict)]
    members = [m for m in model.get("members") or [] if isinstance(m, dict)]
    if not overall_style and not snapshots and not scenes and not members:
        return ""

    parts: List[str] = []
    parts.append(f"【群聊风格：{name}】")
    if snapshots:
        style_lines = []
        for index, snapshot in enumerate(snapshots, start=1):
            style_lines.append(f"整体风格快照{index}：{snapshot}")
        parts.append("\n".join(style_lines))
    elif overall_style:
        parts.append(f"整体风格：{overall_style}")

    if enable_matching and speaker:
        speaker_line = _speaker_profile(speaker, members)
        if speaker_line:
            parts.append(speaker_line)

    if members:
        member_lines = _member_profiles(members)
        if member_lines:
            parts.append("群友档案（按 QQ 号区分）:\n" + member_lines)

    scene_sections: List[str] = []
    for scene in scenes:
        scene_name = str(scene.get("name") or "未命名场景")
        scene_style = str(scene.get("style") or "").strip()
        description = str(scene.get("description") or "").strip()
        note = str(scene.get("note") or "").strip()
        lines = [f"【场景：{scene_name}】"]
        if scene_style:
            lines.append(f"场景风格：{scene_style}")
        if note:
            lines.append(f"场景备注：{note}")
        if description:
            lines.append(f"场景描述：{description}")
        statements = [s for s in scene.get("statements") or [] if isinstance(s, dict)]
        if statements:
            statement_lines = []
            for statement in statements[:4]:
                text = str(statement.get("text") or "").strip()
                if not text:
                    continue
                speaker_name = str(statement.get("speaker_name") or "某群友")
                speaker_id = str(statement.get("speaker") or "").strip()
                target = str(statement.get("target") or "大家")
                speaker_label = f"{speaker_name}(QQ {speaker_id})" if speaker_id else speaker_name
                line = f"- {speaker_label}对{target}说：“{text}”"
                statement_lines.append(line)
            if statement_lines:
                lines.append("代表性语句：\n" + "\n".join(statement_lines))
        scene_sections.append("\n".join(lines))

    combined_scenes = "\n\n".join(scene_sections)
    if combined_scenes:
        parts.append("场景与语句：\n" + combined_scenes)

    text = "\n\n".join(part for part in parts if part).strip()
    if len(text) <= max_chars:
        return text

    truncated: List[str] = []
    for part in parts:
        if truncated and len("\n\n".join(truncated + [part])) > max_chars:
            continue
        truncated.append(part)
    text = "\n\n".join(truncated).strip()
    if not text:
        return ""
    return text[:max_chars].rstrip()
