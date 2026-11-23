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

_ID_FIELD = r"(?:msg_id|message_id|(?:reply_)?session_id|stream_id|(?:引用)?消息\s*ID)"
_ID_VALUE = r"(?:-?[0-9]{6,}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}|[a-f0-9]{12,})"
_DEBUG_FIELD = r"(?:retry_count|max_retries|completion_tokens|prompt_tokens)"
_CODE_SPANS = re.compile(r"```[\s\S]*?```|`[^`\r\n]+`")


@dataclass(frozen=True)
class GarbageRule:
    """一条乱码检测规则。"""

    pattern: str
    name: str
    description: str
    outside_code: bool = False
    regex: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regex", re.compile(self.pattern, re.IGNORECASE))


@dataclass(frozen=True)
class GarbageMatch:
    """一次乱码命中的结果。"""

    rule: GarbageRule
    matched_text: str


DEFAULT_RULES: Tuple[GarbageRule, ...] = (
    GarbageRule(
        pattern=r"为起点按序取字|若引用.*消息.*第\d+至\d+个字符|"
        r"对应原文第\d+至\d+个字|第二个字符\s*\d+\s*[:：]|需要判断目标消息是否(?:真的|确实)?命中",
        name="指令片段泄漏",
        description="回复中出现了疑似任务指令残片（取字/标签判断等）",
    ),
    GarbageRule(
        pattern=(
            rf"""(?<![\w"'“‘「])["']?{_ID_FIELD}["']?[ \t]*[:=：][ \t]*"""
            rf"""(?:["']{_ID_VALUE}["']|{_ID_VALUE})(?=$|[\s,，;；}}\]>。])|"""
            rf"""(?<!\w)"?{_DEBUG_FIELD}"?[ \t]*[:=][ \t]*\d+[\s,;]+"?{_DEBUG_FIELD}"?[ \t]*[:=][ \t]*\d+\b"""
        ),
        name="内部标识符泄漏",
        description="回复包含带具体值的内部 ID 或成组调试字段",
        outside_code=True,
    ),
    GarbageRule(
        pattern=r"<message |<system-reminder>|\[图片[:：]|\[表情[:：]|\[视频[:：]|"
        r"quote=\d+|group_card=",
        name="消息模板泄漏",
        description="回复中出现了消息模板或图片/引用描述残片",
    ),
    GarbageRule(
        pattern=r'(?m)^[ \t]*\{\s*"type"\s*:\s*"text"\s*,\s*"data"\s*:\s*\{\s*"text"\s*:',
        name="JSON 结构泄漏",
        description="回复像是直接输出了 OneBot 消息结构",
        outside_code=True,
    ),
    GarbageRule(
        pattern=r"\ufffd|锟斤拷|锘[縴锛銆]|(?<![\u3400-\u9fff])(?:烫{3,}|屯{3,})(?![\u3400-\u9fff])",
        name="编码乱码",
        description="回复中存在乱码字符（替换符/编码错误等）",
    ),
    GarbageRule(
        pattern=r"按之前的推算.{0,30}从\s*\d+\s*开始|草稿.*取字|无法微调成.*风格而不改变意思|"
        r"目标命中[\s，,：:]*(?:所以|应当|应该|直接).{0,12}(?:回复|输出)",
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
    prose = _CODE_SPANS.sub(lambda span: re.sub(r"[^\r\n]", " ", span[0]), text)
    for rule in rules if rules is not None else _build_rules(extra_patterns):
        match = rule.regex.search(prose if rule.outside_code else text)
        if match:
            return GarbageMatch(rule=rule, matched_text=match.group(0))
    return None
