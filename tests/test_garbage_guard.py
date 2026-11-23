"""回复乱码检测测试（检测逻辑 + after_response Hook 行为）。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from chat_style_translator_plugin.modules.garbage_guard import (
    DEFAULT_RULES,
    GarbageRule,
    detect_garbage,
)
from chat_style_translator_plugin.plugin import ChatStyleTranslatorPlugin

# 模拟的乱码回复
INCIDENT_TEXT = (
    "出门前记得带上那把蓝伞。”为起点按序取字。若引用的消息ID=1234567890，"
    "其内容“出门前记得带上那把蓝伞。”为第2至11个字符，对应原文第2至11个字"
    "“出门前记得带上那把蓝伞。”。 第二个字符2：2=“”，与原文字符“”?"
)

NORMAL_TEXTS = [
    "晚安啦",
    "深海地图掉材料快些，北边的空房偶尔也能捡到呗",
    "这把普通的弓又不贵",
    "哥几个都打了三把了，我先退出了",
    "试了下那把弓配二级箭",
    "我觉得应该把灯打开，检查的是灯泡，不是开关",
    "今天天气不错，一起出去玩吧",
    "这个接口的 message_id 从哪拿呀？",
    "有个 session_id 跟前两组的不一样",
    '回复内容包含 "msg_id": "123" 的 JSON',
    "这里的 retry_count=3 是失败次数吗",
    "这杯还烫烫的，等凉了再喝",
    "从起始序号开始遍历，然后原样输出。",
    "我们需要判断数组索引是否越界。",
    "字段可以写成 " + chr(96) + "message_id=1234567890" + chr(96) + " 作为例子。",
    "代码举例如下\n" + chr(96) * 3 + 'json\n{"type":"text","data":{"text":"你好"}}\n' + chr(96) * 3,
    "？",
]


class TestDetectGarbage:
    def test_incident_text_detected(self) -> None:
        match = detect_garbage(INCIDENT_TEXT)
        assert match is not None
        assert match.rule.name == "指令片段泄漏"

    def test_msg_id_leak_detected(self) -> None:
        for text in (
            "之前的回复出现了异常输出 msg_id=1234567890 需要重新发送",
            "消息ID=1234567890 的内容是第2至11个字符",
            '{"msg_id": "1234567890", "role": "assistant"}',
            "retry_count=2, max_retries=3",
        ):
            assert detect_garbage(text) is not None, text

    def test_message_template_leak_detected(self) -> None:
        for text in (
            "<message msg_id=\"123\" time=\"01:00\" user=\"测试用户\">\n晚安",
            "[图片：金发兽耳少女自拍]",
            "quote=9876543210 哪里有杯子",
        ):
            assert detect_garbage(text) is not None, text

    def test_json_leak_detected(self) -> None:
        assert detect_garbage('{"type": "text", "data": {"text": "你好"}}') is not None

    def test_mojibake_detected(self) -> None:
        for text in ("回复锟斤拷了", "这里有\uFFFD替换符", "烫烫烫"):
            assert detect_garbage(text) is not None, text

    def test_reasoning_leak_detected(self) -> None:
        for text in (
            "按之前的推算，二十四号从2开始",
            "我们需要判断目标消息是否真的命中该标签",
            "目标命中，所以回复这个",
        ):
            assert detect_garbage(text) is not None, text

    def test_normal_texts_not_detected(self) -> None:
        for text in NORMAL_TEXTS:
            assert detect_garbage(text) is None, text

    def test_empty_text(self) -> None:
        assert detect_garbage("") is None
        assert detect_garbage(None) is None

    def test_custom_patterns(self) -> None:
        assert detect_garbage("普通回复", extra_patterns=[r"普通回复"]) is not None
        assert detect_garbage("普通回复", extra_patterns=[r"\b不存在词\b"]) is None

    def test_invalid_custom_pattern_skipped(self) -> None:
        assert detect_garbage("完全正常的回复", extra_patterns=[r"[unclosed"]) is None

    def test_custom_rules_override(self) -> None:
        custom = [GarbageRule(pattern=r"只有我", name="自定义", description="测试")]
        assert detect_garbage("只有我命中", rules=custom) is not None
        assert detect_garbage(INCIDENT_TEXT, rules=custom) is None


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def make_plugin(**config_patch: Any) -> ChatStyleTranslatorPlugin:
    plugin = ChatStyleTranslatorPlugin()
    base: Dict[str, Any] = {"plugin": {"enabled": True, "config_version": "0.11.0"}}
    base.update(config_patch)
    plugin.set_plugin_config(base)
    plugin._sync_config()
    return plugin


class TestGuardHook:
    def test_garbage_triggers_retry(self) -> None:
        plugin = make_plugin()
        result = run(plugin.guard_garbage_response(
            response=INCIDENT_TEXT,
            session_id="sess-1",
        ))
        assert result["action"] == "continue"
        modified = result["modified_kwargs"]
        assert modified["retry"] is True
        assert "乱码" in modified["retry_reason"] or "残片" in modified["retry_reason"]
        assert modified["matched_regex"] == "指令片段泄漏"
        assert modified["matched_regex_pattern"]
        assert modified["matched_regex_description"]
        assert modified["response"] == INCIDENT_TEXT

    def test_normal_text_continues(self) -> None:
        plugin = make_plugin()
        for text in NORMAL_TEXTS:
            result = run(plugin.guard_garbage_response(response=text, session_id="sess-1"))
            assert result == {"action": "continue"}, text

    def test_disabled_no_intercept(self) -> None:
        plugin = make_plugin(garbage_guard={"enabled": False})
        result = run(plugin.guard_garbage_response(response=INCIDENT_TEXT, session_id="sess-1"))
        assert result == {"action": "continue"}

    def test_retry_disabled_log_only(self) -> None:
        plugin = make_plugin(garbage_guard={"enabled": True, "retry_on_detect": False})
        result = run(plugin.guard_garbage_response(response=INCIDENT_TEXT, session_id="sess-1", retry_count=3, max_retries=3))
        assert result == {"action": "continue"}

    def test_short_normal_text_continues(self) -> None:
        plugin = make_plugin()
        result = run(plugin.guard_garbage_response(response="原样", session_id="sess-1"))
        assert result == {"action": "continue"}


    def test_short_garble_retries_and_aborts_at_host_limit(self) -> None:
        plugin = make_plugin()
        first = run(plugin.guard_garbage_response(response="\uFFFD", retry_count=0, max_retries=3))
        assert first["modified_kwargs"]["retry"] is True
        final = run(plugin.guard_garbage_response(response="\uFFFD", retry_count=3, max_retries=3))
        assert final == {"action": "abort", "custom_result": {"reason": "回复重试后仍命中乱码拦截"}}
        for normal in ("好", "烫烫", "原样"):
            assert run(plugin.guard_garbage_response(
                response=normal, retry_count=3, max_retries=3,
            )) == {"action": "continue"}

    def test_custom_patterns_from_config(self) -> None:
        plugin = make_plugin(garbage_guard={"custom_patterns": [r"猪头"]})
        result = run(plugin.guard_garbage_response(response="你真是个猪头", session_id="sess-1"))
        assert result["action"] == "continue"
        assert result["modified_kwargs"]["retry"] is True
        assert result["modified_kwargs"]["matched_regex"].startswith("自定义正则")

    def test_default_rules_compiled(self) -> None:
        for rule in DEFAULT_RULES:
            assert rule.regex.pattern == rule.pattern
