"""回复乱码检测。

用于在 Maisaka 回复器生成回复后，检测模型把任务指令、内部标识符、
推理过程当成回复内容输出时留下的"异常残片"，以及编码错误产生的乱码字符。

纯函数实现，不依赖插件运行时，便于单元测试。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("chat_style_translator.garbage_guard")


@dataclass(frozen=True)
class GarbageRule:
    """一条乱码检测规则。"""

    pattern: str
    name: str
    description: str
    regex: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regex", re.compile(self.pattern, re.IGNORECASE))


@dataclass(frozen=True)
class GarbageMatch:
    """一次乱码命中的结果。"""

    rule: GarbageRule
    matched_text: str


DEFAULT_RULES: Tuple[GarbageRule, ...] = (
    # 指令片段泄漏：本次事故的"为起点按序取字"类残片
    GarbageRule(
        pattern=r"为起点按序取字|按序取字|若引用.*消息.*第\d+至\d+个字符|"
        r"对应原文第\d+至\d+个字|第二个字符\s*\d+\s*[:：]|命中该标签|"
        r"起始序号|元组\s*\(|需要判断目标消息是否命中",
        name="指令片段泄漏",
        description="回复中出现了疑似任务指令残片（取字/标签判断等）",
    ),
    # 内部标识符泄漏
    GarbageRule(
        pattern=r"\bmsg_id\b|\bmessage_id\b|\b(?:reply_)?session_id\b|stream_id\b|"
        r"\bretry_count\b|\bmax_retries\b|\bcompletion_tokens\b|\bprompt_tokens\b|"
        r"\btool_calls\b|\bmatched_regex\b|selected_expression|消息ID\s*[:=]|引用消息ID",
        name="内部标识符泄漏",
        description="回复中出现了系统内部标识符（msg_id/session_id 等）",
    ),
    # 消息模板 / 富媒体描述泄漏
    GarbageRule(
        pattern=r"<message |<system-reminder>|\[图片[:：]|\[表情[:：]|\[视频[:：]|"
        r"quote=\d+|group_card=",
        name="消息模板泄漏",
        description="回复中出现了消息模板或图片/引用描述残片",
    ),
    # JSON / 代码结构泄漏
    GarbageRule(
        pattern=r'"msg_id"\s*[:=]|"type"\s*:\s*"text"|"data"\s*:\s*\{|"content"\s*:\s*"',
        name="JSON 结构泄漏",
        description="回复中出现了 JSON 结构片段",
    ),
    # 编码乱码
    GarbageRule(
        pattern=r"\ufffd|锟斤拷|锘|烫烫|屯屯|�",
        name="编码乱码",
        description="回复中存在乱码字符（替换符/编码错误等）",
    ),
    # 推理过程泄漏
    GarbageRule(
        pattern=r"按之前的推算|我们需要判断|草稿.*取字|无法微调成.*风格而不改变意思|"
        r"原样输出|目标命中|得出结论\s*[:：]",
        name="推理内容泄漏",
        description="回复中出现了模型推理过程片段",
    ),
)


def _build_rules(
    extra_patterns: Optional[Iterable[str]] = None,
) -> List[GarbageRule]:
    """组装默认规则与用户自定义规则（无效正则跳过并告警）。"""

    rules: List[GarbageRule] = list(DEFAULT_RULES)
    for raw in extra_patterns or []:
        pattern = str(raw or "").strip()
        if not pattern:
            continue
        try:
            rules.append(
                GarbageRule(
                    pattern=pattern,
                    name=f"自定义正则 {pattern}",
                    description="用户自定义检测正则",
                )
            )
        except re.error:
            logger.warning("自定义乱码检测正则无效，已跳过: %r", pattern)
    return rules


def detect_garbage(
    text: str,
    extra_patterns: Optional[Iterable[str]] = None,
    rules: Optional[Sequence[GarbageRule]] = None,
) -> Optional[GarbageMatch]:
    """检测文本中是否包含乱码/指令残片。

    Args:
        text: 待检测的回复文本。
        extra_patterns: 用户自定义追加正则（按行分隔）。
        rules: 内置规则集合，默认使用 DEFAULT_RULES。

    Returns:
        Optional[GarbageMatch]: 命中时返回匹配信息，否则返回 None。
    """

    if not text:
        return None
    for rule in rules if rules is not None else _build_rules(extra_patterns):
        match = rule.regex.search(text)
        if match:
            return GarbageMatch(rule=rule, matched_text=match.group(0))
    return None
