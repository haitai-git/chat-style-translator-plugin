"""聊天风格转译插件全量测试（插件核心逻辑、收集、自动融合、分析、微调 Hook、生命周期）。"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from chat_style_translator_plugin.modules.storage import StyleStore
from chat_style_translator_plugin.plugin import ChatStyleTranslatorPlugin


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def make_plugin(**config_patch: Any) -> ChatStyleTranslatorPlugin:
    plugin = ChatStyleTranslatorPlugin()
    base: Dict[str, Any] = {"plugin": {"enabled": True, "config_version": "0.9.0"}}
    base.update(config_patch)
    plugin.set_plugin_config(base)
    plugin._sync_config()
    return plugin


def sample_message(
    sender: str = "20002",
    text: str = "今天天气不错",
    group_id: str = "88880001",
    self_id: str = "",
    is_command: bool = False,
    at: Optional[Dict[str, str]] = None,
    reply: Optional[Dict[str, str]] = None,
    session_id: str = "sess-1",
) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "message_id": "msg-1",
        "timestamp": "1.0",
        "platform": "qq",
        "session_id": session_id,
        "is_command": is_command,
        "is_emoji": False,
        "is_picture": False,
        "is_mentioned": False,
        "is_at": False,
        "is_notify": False,
        "processed_plain_text": text,
        "plain_text": text,
        "message_info": {
            "user_info": {"user_id": sender, "user_nickname": f"user{sender}", "user_cardname": f"card{sender}"},
            "group_info": {"group_id": group_id, "group_name": "测试群"},
            "additional_config": {"self_id": self_id} if self_id else {},
        },
        "raw_message": [],
    }
    if at:
        message["raw_message"].append({"type": "at", "data": at})
    if reply:
        message["raw_message"].append({"type": "reply", "data": reply})
    return message


class FakeCtx:
    def __init__(self, data_dir: Path) -> None:
        self.paths = type("Paths", (), {"data_dir": data_dir, "runtime_dir": data_dir})()
        self.chat = type("Chat", (), {"get_all_streams": self._streams})()
        self.llm = type("LLM", (), {"generate": self._gen, "embed": self._embed})()
        self.send = type("Send", (), {"text": self._send})()
        self.api = type("API", (), {"call": self._api_call})()
        self._streams_result: List[Dict[str, Any]] = []
        self._api_calls: List[Dict[str, Any]] = []

    async def _streams(self) -> Any:
        return {"streams": self._streams_result}

    async def _gen(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        return {"success": True, "response": ""}

    async def _embed(self, **kwargs: Any) -> Any:
        raise RuntimeError("not used")

    async def _send(self, text: str, stream_id: str) -> Any:
        return {"success": True}

    async def _api_call(self, method: str, **kwargs: Any) -> Any:
        self._api_calls.append({"method": method, **kwargs})
        return {"data": {"message_id": "1"}}


class FakeAnalyzer:
    analyze_result: Dict[str, Any] = {"overall_style": "风格", "members": [], "scenes": []}

    async def analyze_window(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # 回显收到的消息，让测试能验证分析前的过滤是否生效
        result = json.loads(json.dumps(self.analyze_result))
        result["scenes"] = [
            {"name": "S", "statements": [{"text": m["text"]} for m in messages]}
        ]
        return result

    async def merge_partials(
        self,
        partials: List[Dict[str, Any]],
        name: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        merged = json.loads(json.dumps(partials[0]))
        merged["name"] = name or "群聊风格模型"
        return merged

    async def filter_statements(
        self,
        statements: List[Dict[str, Any]],
        blacklist_items: List[Dict[str, Any]],
        threshold: float,
        model_id: str = "",
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return statements, []


class FakeLLM:
    def __init__(self) -> None:
        self.responses: List[Dict[str, Any]] = []

    def set_responses(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    async def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        if self.responses:
            return self.responses.pop(0)
        return {"success": True, "response": ""}


class TestExtractors:
    def test_group_and_sender(self) -> None:
        plugin = make_plugin()
        message = sample_message(sender="123", text="hi")
        assert plugin._extract_group_id(message) == "88880001"
        assert plugin._extract_sender_id(message) == "123"
        assert plugin._extract_group_id({}) == ""
        assert plugin._extract_sender_id({}) == ""
        assert plugin._extract_group_id({"message_info": {"group_info": None}}) == ""

    def test_sender_blocked(self) -> None:
        plugin = make_plugin(user_block={"qq": ["黑名单用户"], "bot_ids": ["888"], "block_qq_bots": True})
        assert plugin._is_sender_blocked(sample_message(sender="黑名单用户")) is True
        assert plugin._is_sender_blocked(sample_message(sender="888")) is True
        assert plugin._is_sender_blocked(sample_message(sender="285419631000")) is True
        assert plugin._is_sender_blocked(sample_message(sender="20002")) is False
        plugin2 = make_plugin(user_block={"block_qq_bots": False, "bot_ids": ["888"]})
        assert plugin2._is_sender_blocked(sample_message(sender="888")) is False

    def test_text_blocked_and_phrase(self) -> None:
        plugin = make_plugin(user_block={"block_slash": True})
        assert plugin._is_text_blocked("/风格列表") is True
        assert plugin._is_text_blocked("正常聊天") is False
        plugin2 = make_plugin(user_block={"block_slash": False})
        assert plugin2._is_text_blocked("/风格列表") is False
        plugin3 = make_plugin(phrase_blacklist={"items": ["修仙", "签到"]})
        assert plugin3._text_has_blocked_phrase("今天开始修仙了") is True
        assert plugin3._text_has_blocked_phrase("你修的是什么") is False
        assert plugin3._text_has_blocked_phrase("修") is False
        assert plugin3._text_has_blocked_phrase("每日签到打卡") is True
        assert plugin3._text_has_blocked_phrase("") is False

    def test_self_message_detection(self) -> None:
        plugin = make_plugin(user_block={"skip_self_messages": True, "self_qq": []})
        assert plugin._is_self_message(sample_message(sender="10001", text="x", self_id="10001")) is True
        assert "10001" in plugin._self_account_ids
        assert plugin._is_self_message(sample_message(sender="10001", text="x")) is True
        assert plugin._is_self_message(sample_message(sender="20002", text="x", self_id="10001")) is False
        plugin2 = make_plugin(user_block={"skip_self_messages": True, "self_qq": ["55555"]})
        assert plugin2._is_self_message(sample_message(sender="55555", text="x")) is True
        plugin3 = make_plugin(user_block={"skip_self_messages": False})
        assert plugin3._is_self_message(sample_message(sender="10001", text="x", self_id="10001")) is False

    def test_build_record(self) -> None:
        message = sample_message(
            sender="100", text="  你好  ",
            at={"target_user_id": "200", "target_user_nickname": "小红"},
            reply={"target_message_sender_id": "300", "target_message_sender_nickname": "小明"},
        )
        record = ChatStyleTranslatorPlugin._build_record(message)
        assert record["text"] == "你好"
        assert record["sender_id"] == "100"
        assert record["at_user_id"] == "200" and record["at_user_name"] == "小红"
        assert record["reply_to_id"] == "300" and record["reply_to_name"] == "小明"
        assert record["sender_name"] == "card100"
        assert ChatStyleTranslatorPlugin._build_record(sample_message(text="  ")) is None
        assert ChatStyleTranslatorPlugin._build_record(sample_message(text="命令", is_command=True)) is None

    def test_session_and_speaker_memory(self) -> None:
        plugin = make_plugin()
        plugin._remember_message_session(sample_message(sender="1", text="x"), "88880001")
        assert plugin._session_map["sess-1"] == "88880001"
        assert plugin._stream_map["88880001"] == "sess-1"
        assert plugin._group_stream_id("88880001") == "sess-1"
        assert plugin._group_stream_id("999") == ""
        plugin._remember_speaker(sample_message(sender="42", text="x"))
        assert plugin._speaker_map["sess-1"]["user_id"] == "42"

    def test_session_map_bounded(self, monkeypatch: Any, tmp_path: Path) -> None:
        monkeypatch.setattr("chat_style_translator_plugin.plugin._MAX_SESSION_MAP_ENTRIES", 3)
        plugin = make_plugin()
        for i in range(1, 6):
            plugin._remember_message_session(
                sample_message(sender=str(i), text="x", session_id=f"sess-{i}"), f"g{i}"
            )
        assert len(plugin._session_map) <= 3
        assert len(plugin._stream_map) <= 3
        assert plugin._session_map["sess-5"] == "g5"  # 最新保留
        assert "sess-1" not in plugin._session_map  # 最旧被淘汰
        assert "sess-2" not in plugin._session_map
        assert plugin._stream_map["g5"] == "sess-5"
        assert "g1" not in plugin._stream_map  # 反向映射同步清理

    def test_session_map_rebind_cleans_old_reverse(self) -> None:
        plugin = make_plugin()
        plugin._remember_message_session(sample_message(session_id="sess-1"), "g1")
        plugin._remember_message_session(sample_message(session_id="sess-1"), "g2")
        assert plugin._session_map["sess-1"] == "g2"
        assert plugin._stream_map["g2"] == "sess-1"
        assert "g1" not in plugin._stream_map

    def test_resolve_active_models(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.bind_group("111"))
        run(store.set_group_field("111", "active_models", ["m1"]))
        plugin = make_plugin()
        plugin._store = store
        plugin._global_models = []
        assert plugin._resolve_active_models("111") == ["m1"]
        assert plugin._resolve_active_models("999") == []
        plugin._global_models = ["g1"]
        assert plugin._resolve_active_models("111") == ["g1"]

    def test_strip_rewrite_wrappers(self) -> None:
        strip = ChatStyleTranslatorPlugin._strip_rewrite_wrappers
        assert strip("```text\n你好\n```") == "你好"
        assert strip("```\n你好\n```") == "你好"
        assert strip("微调后：你好") == "你好"
        assert strip("改写后：你好") == "你好"
        assert strip("「你好」") == "你好"
        assert strip("\"你好\"") == "你好"
        assert strip("你好") == "你好"
        assert strip("") == ""

    def test_rewrite_safety(self) -> None:
        safety = ChatStyleTranslatorPlugin._rewrite_passes_safety
        assert safety("你好世界", "你好世界！") is True
        assert safety("你好世界", "你好世界") is True
        assert safety("你好世界", "你") is False
        assert safety("你好世界", "你" * 20) is False
        assert safety("你好世界", "   ") is False

    def test_owner_permission(self) -> None:
        plugin = make_plugin(owner={"qq": ["123456789"]})
        assert plugin._is_owner({"platform": "qq", "user_id": "123456789"}) is True
        assert plugin._is_owner({"platform": "discord", "user_id": "123456789"}) is False
        assert plugin._is_owner({"user_id": "123456789"}) is False
        assert plugin._is_owner({"user_id": "别人"}) is False
        assert plugin._is_owner({"is_local_operator": True}) is True
        assert plugin._is_owner({}) is False

    def test_merge_options_from_config(self) -> None:
        plugin = make_plugin(merge={"ratio_a": 70, "dedup_level": "激进精简", "max_total_statements": 120})
        assert plugin._merge_options_from_config() == {
            "ratio": (70, 30), "dedup_level": "激进精简",
            "scene_merge_threshold": 0.8, "snapshot_merge_threshold": 0.55,
        }
        plugin2 = make_plugin(merge={"scene_merge_threshold": 0.9, "snapshot_merge_threshold": 0.4})
        assert plugin2._merge_options_from_config() == {
            "ratio": (50, 50), "dedup_level": "标准",
            "scene_merge_threshold": 0.9, "snapshot_merge_threshold": 0.4,
        }
        # 防御性钳制（正常配置模型已校验范围，这里直接注入非法值验证钳制逻辑）
        stub_merge = type("Merge", (), {"ratio_a": 200, "dedup_level": "乱填", "max_total_statements": -5})()
        plugin3 = make_plugin()
        plugin3._plugin_config_instance = type("Cfg", (), {"merge": stub_merge})()  # type: ignore[attr-defined]
        assert plugin3._merge_options_from_config() == {
            "ratio": (100, 0), "dedup_level": "标准",
            "scene_merge_threshold": 0.8, "snapshot_merge_threshold": 0.55,
        }


class TestCollectionFlow:
    def make_plugin_with_store(self, tmp_path: Path, **config_patch: Any) -> ChatStyleTranslatorPlugin:
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001"))
        plugin = make_plugin(**config_patch)
        plugin._store = store
        return plugin

    def test_collect_normal(self, tmp_path: Path) -> None:
        plugin = self.make_plugin_with_store(tmp_path)
        result = run(plugin.handle_message(sample_message(text="普通聊天内容")))
        assert result is None
        assert plugin._store.message_count("88880001") == 1
        record = plugin._store.messages_for("88880001")[0]
        assert record["text"] == "普通聊天内容"
        assert record["sender_id"] == "20002"
        assert plugin._session_map["sess-1"] == "88880001"
        assert plugin._speaker_map["sess-1"]["user_id"] == "20002"

    def test_collect_filters(self, tmp_path: Path) -> None:
        plugin = self.make_plugin_with_store(
            tmp_path,
            user_block={"qq": ["黑名单"], "block_slash": True},
            phrase_blacklist={"items": ["修仙"]},
        )
        collected: List[str] = []
        for message in [
            sample_message(sender="黑名单", text="你好"),
            sample_message(text="/风格列表"),
            sample_message(text="开始修仙了"),
            sample_message(text="只有修字"),
            sample_message(text="正常的话"),
        ]:
            run(plugin.handle_message(message))
        records = plugin._store.messages_for("88880001")
        assert [r["text"] for r in records] == ["只有修字", "正常的话"]

    def test_collect_trims_oldest(self, tmp_path: Path) -> None:
        """每群存储上限：收集超过上限后自动删除最旧消息。"""
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001"))
        plugin = make_plugin(collection={"max_stored_messages": 3})
        plugin._store = store
        for i in range(1, 6):
            run(plugin.handle_message(sample_message(sender=str(i), text=f"消息{i}", session_id=f"sess-{i}")))
        assert store.message_count("88880001") == 3
        assert [r["text"] for r in store.messages_for("88880001")] == ["消息3", "消息4", "消息5"]

    def test_collect_skip_self(self, tmp_path: Path) -> None:
        plugin = self.make_plugin_with_store(tmp_path, user_block={"skip_self_messages": True})
        run(plugin.handle_message(sample_message(sender="10001", text="机器人的话", self_id="10001")))
        assert plugin._store.message_count("88880001") == 0
        run(plugin.handle_message(sample_message(sender="10001", text="机器人的话2")))
        assert plugin._store.message_count("88880001") == 0
        assert "sess-1" not in plugin._speaker_map  # 本人消息不当说话人

    def test_collect_unbound_group(self, tmp_path: Path) -> None:
        plugin = self.make_plugin_with_store(tmp_path)
        run(plugin.handle_message(sample_message(group_id="999", text="不收集")))
        assert plugin._store.message_count("88880001") == 0

    def test_collect_trigger_auto_analyze(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001"))
        plugin = make_plugin(collection={"auto_analyze_count": 2})
        plugin._store = store
        triggered: List[tuple] = []

        async def fake_trigger(group_id: str, reason: str) -> None:
            triggered.append((group_id, reason))

        plugin._trigger_auto_analyze = fake_trigger  # type: ignore[method-assign]
        run(plugin.handle_message(sample_message(text="1")))
        run(plugin.handle_message(sample_message(text="2")))
        assert triggered == [("88880001", "消息条数达到阈值")]


class TestAutoMerge:
    def make_plugin(self, tmp_path: Path, **merge: Any) -> ChatStyleTranslatorPlugin:
        store = StyleStore(tmp_path)
        run(store.save_model({"name": "模型A", "overall_style": "a", "members": [], "scenes": []}))
        run(store.save_model({"name": "模型B", "overall_style": "b", "members": [], "scenes": []}))
        plugin = make_plugin(merge={"trigger": True, "model_a": "模型A", "model_b": "模型B", **merge})
        plugin._store = store
        plugin._analyzer = FakeAnalyzer()
        return plugin

    def test_runs_every_trigger(self, tmp_path: Path) -> None:
        plugin = self.make_plugin(tmp_path)
        results: List[str] = []

        async def fake_merge(a: Dict[str, Any], b: Dict[str, Any], name: str) -> str:
            results.append(name)
            return f"融合完成，已生成模型「{name}」"

        plugin.merge_models = fake_merge  # type: ignore[method-assign]
        run(plugin._maybe_auto_merge())
        run(plugin._maybe_auto_merge())
        assert results == ["模型A×模型B", "模型A×模型B"]  # 每次触发都执行，不限次数
        assert not (tmp_path / "merge_history.json").exists()

    def test_off(self, tmp_path: Path) -> None:
        plugin = make_plugin(merge={"trigger": False, "model_a": "模型A", "model_b": "模型B"})
        plugin._store = StyleStore(tmp_path)
        plugin._analyzer = FakeAnalyzer()
        called: List[str] = []

        async def fake_merge(a, b, name):
            called.append(name)

        plugin.merge_models = fake_merge  # type: ignore[method-assign]
        run(plugin._maybe_auto_merge())
        assert called == []

    def test_validation(self, tmp_path: Path) -> None:
        plugin = self.make_plugin(tmp_path, model_a="不存在")
        run(plugin._maybe_auto_merge())
        assert not (tmp_path / "merge_history.json").exists()
        plugin2 = make_plugin(merge={"trigger": True, "model_a": "模型A", "model_b": ""})
        plugin2._store = StyleStore(tmp_path)
        plugin2._analyzer = FakeAnalyzer()
        run(plugin2._maybe_auto_merge())

    def test_failure_logged_only(self, tmp_path: Path) -> None:
        plugin = self.make_plugin(tmp_path)

        async def failing_merge(a, b, name):
            return "融合失败: LLM 未返回有效结果。"

        plugin.merge_models = failing_merge  # type: ignore[method-assign]
        run(plugin._maybe_auto_merge())
        assert not (tmp_path / "merge_history.json").exists()

    def test_options_passthrough(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.save_model({"name": "A", "scenes": [], "members": []}))
        run(store.save_model({"name": "B", "scenes": [], "members": []}))
        plugin = make_plugin(merge={"trigger": True, "model_a": "A", "model_b": "B",
                                    "ratio_a": 80, "dedup_level": "激进精简", "max_total_statements": 100})
        plugin._store = store
        plugin._analyzer = FakeAnalyzer()
        received: Dict[str, Any] = {}

        async def fake_merge(a, b, name):
            received["options"] = plugin._merge_options_from_config()
            return "融合完成"

        plugin.merge_models = fake_merge  # type: ignore[method-assign]
        run(plugin._maybe_auto_merge())
        assert received["options"] == {
            "ratio": (80, 20),
            "dedup_level": "激进精简",
            "scene_merge_threshold": 0.8,
            "snapshot_merge_threshold": 0.55,
        }


class TestAutoCompact:
    def make_plugin(self, tmp_path: Path, **compact: Any) -> ChatStyleTranslatorPlugin:
        store = StyleStore(tmp_path)
        run(store.save_model({"name": "大模型", "model_id": "m1", "overall_style": "a", "members": [], "scenes": []}))
        plugin = make_plugin(compact={"trigger": True, "model": "大模型", **compact})
        plugin._store = store
        plugin._analyzer = FakeAnalyzer()
        return plugin

    def test_runs_every_trigger(self, tmp_path: Path) -> None:
        plugin = self.make_plugin(tmp_path)
        results: List[str] = []

        async def fake_compact(model: Dict[str, Any], new_name: str = "") -> str:
            results.append(new_name)
            return f"精简完成，已生成模型「{new_name or '大模型（精简版）'}」"

        plugin.compact_model = fake_compact  # type: ignore[method-assign]
        run(plugin._maybe_auto_compact())
        run(plugin._maybe_auto_compact())
        assert results == ["", ""]  # 每次触发都执行，不限次数
        assert not (tmp_path / "merge_history.json").exists()

    def test_off(self, tmp_path: Path) -> None:
        plugin = make_plugin(compact={"trigger": False, "model": "大模型"})
        plugin._store = StyleStore(tmp_path)
        plugin._analyzer = FakeAnalyzer()
        called: List[str] = []

        async def fake_compact(model: Dict[str, Any], new_name: str = "") -> str:
            called.append(new_name)

        plugin.compact_model = fake_compact  # type: ignore[method-assign]
        run(plugin._maybe_auto_compact())
        assert called == []

    def test_validation(self, tmp_path: Path) -> None:
        plugin = self.make_plugin(tmp_path, model="不存在")
        run(plugin._maybe_auto_compact())
        assert not (tmp_path / "merge_history.json").exists()
        plugin2 = make_plugin(compact={"trigger": True, "model": ""})
        plugin2._store = StyleStore(tmp_path)
        plugin2._analyzer = FakeAnalyzer()
        run(plugin2._maybe_auto_compact())

    def test_failure_logged_only(self, tmp_path: Path) -> None:
        plugin = self.make_plugin(tmp_path)

        async def failing_compact(model: Dict[str, Any], new_name: str = "") -> str:
            return "精简失败: 意外错误"

        plugin.compact_model = failing_compact  # type: ignore[method-assign]
        run(plugin._maybe_auto_compact())
        assert not (tmp_path / "merge_history.json").exists()

    def test_options_passthrough(self, tmp_path: Path) -> None:
        plugin = make_plugin(compact={
            "dedup_level": "激进精简", "max_total_statements": 100,
            "scene_merge_threshold": 0.9, "snapshot_merge_threshold": 0.4,
        })
        assert plugin._compact_options_from_config() == {
            "dedup_level": "激进精简",
            "scene_merge_threshold": 0.9,
            "snapshot_merge_threshold": 0.4,
        }
        # 防御性钳制（正常配置模型已校验范围，这里直接注入非法值验证钳制逻辑）
        stub_compact = type("Compact", (), {"dedup_level": "乱填", "max_total_statements": -5})()
        plugin2 = make_plugin()
        plugin2._plugin_config_instance = type("Cfg", (), {"compact": stub_compact})()  # type: ignore[attr-defined]
        assert plugin2._compact_options_from_config() == {
            "dedup_level": "标准",
            "scene_merge_threshold": 0.8,
            "snapshot_merge_threshold": 0.55,
        }


class TestAnalysis:
    def make_env(self, tmp_path: Path, **config_patch: Any) -> tuple[ChatStyleTranslatorPlugin, StyleStore]:
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001"))
        plugin = make_plugin(**config_patch)
        plugin._store = store
        analyzer = FakeAnalyzer()
        analyzer.analyze_result = {
            "overall_style": "风格", "members": [],
            "scenes": [{"name": "S", "statements": [{"text": "原话"}]}],
        }
        plugin._analyzer = analyzer
        return plugin, store

    def test_empty_messages(self, tmp_path: Path) -> None:
        plugin, _ = self.make_env(tmp_path)
        result = run(plugin.run_analysis("88880001"))
        assert "还没有收集到可分析的对话" in result

    def test_not_bound(self, tmp_path: Path) -> None:
        plugin, _ = self.make_env(tmp_path)
        result = run(plugin.run_analysis("999"))
        assert "尚未绑定" in result

    def test_success(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        run(store.append_message("88880001", {"id": "1", "text": "大家好", "sender_name": "小明"}))
        result = run(plugin.run_analysis("88880001"))
        assert "分析完成" in result
        assert store.message_count("88880001") == 0
        group = store.get_group("88880001")
        assert group["last_analyzed_at"] > 0
        assert group["analyzing"] is False
        models = store.list_models()
        assert len(models) == 1 and models[0]["status"] == "pending"

    def test_filters_phrase_and_self(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(
            tmp_path,
            phrase_blacklist={"items": ["修仙"]},
            user_block={"skip_self_messages": True},
        )
        plugin._self_account_ids.add("10001")
        run(store.append_message("88880001", {"id": "1", "text": "开始修仙了", "sender_name": "A", "sender_id": "5"}))
        run(store.append_message("88880001", {"id": "2", "text": "正常的话", "sender_name": "B", "sender_id": "6"}))
        run(store.append_message("88880001", {"id": "3", "text": "机器人发言", "sender_name": "bot", "sender_id": "10001"}))
        result = run(plugin.run_analysis("88880001"))
        assert "分析完成" in result
        model = store.list_models()[0]
        assert len(model["scenes"][0]["statements"]) == 1
        assert model["scenes"][0]["statements"][0]["text"] == "正常的话"

    def test_concurrency_guard(self, tmp_path: Path) -> None:
        plugin, _ = self.make_env(tmp_path)
        plugin._analyzing.add("88880001")
        result = run(plugin.run_analysis("88880001"))
        assert "正在分析中" in result

    def test_background_loop_check(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        run(store.append_message("88880001", {"id": "1", "text": "消息", "sender_name": "A"}))
        triggered: List[str] = []

        async def fake_trigger(group_id: str, reason: str) -> None:
            triggered.append(group_id)

        plugin._trigger_auto_analyze = fake_trigger  # type: ignore[method-assign]
        plugin._analyzing = set()
        run(plugin._check_scheduled_analyze())
        assert triggered == []  # auto_analyze_count 默认 100，未达阈值

    def _append_n(self, store: StyleStore, count: int) -> None:
        for i in range(count):
            run(store.append_message("88880001", {"id": str(i), "text": f"消息{i}", "sender_name": "A"}))

    def _enable_progress(self, plugin: ChatStyleTranslatorPlugin) -> List[str]:
        msgs: List[str] = []

        async def fake_notify(group_id: str, text: str) -> None:
            msgs.append(text)

        plugin._notify_analysis = fake_notify  # type: ignore[method-assign]
        return msgs

    def test_total_progress_every_batch(self, tmp_path: Path) -> None:
        """间隔为 0：每批完成都发送总进度（已处理 X/总 Y 条（Z%）），并发不改变结果。"""
        plugin, store = self.make_env(
            tmp_path,
            analysis={"model": "", "notify_qq": "", "max_statements_per_scene": 30,
                      "analyze_concurrency": 2, "progress_interval_seconds": 0},
            collection={"auto_analyze_count": 0, "daily_time": "", "max_messages": 300, "window_size": 60},
        )
        self._append_n(store, 180)  # 3 批
        msgs = self._enable_progress(plugin)
        result = run(plugin._run_analysis("88880001", "测试", notify=True))
        assert "分析完成" in result
        progress = [m for m in msgs if "分析进度" in m]
        assert len(progress) == 3
        assert progress[0].endswith("分析进度：已处理 60/180 条（33%）")
        assert progress[1].endswith("分析进度：已处理 120/180 条（66%）")
        assert progress[2].endswith("分析进度：已处理 180/180 条（100%）")
        assert any("分析完成" in m for m in msgs)  # 完成通知仍发送
        model = store.list_models()[0]
        assert len(model["scenes"][0]["statements"]) >= 3  # 并发不丢失各批内容

    def test_total_progress_throttle(self, tmp_path: Path) -> None:
        """间隔很大：只发首批完成后的一条进度。"""
        plugin, store = self.make_env(
            tmp_path,
            analysis={"model": "", "notify_qq": "", "max_statements_per_scene": 30,
                      "analyze_concurrency": 2, "progress_interval_seconds": 99999},
            collection={"auto_analyze_count": 0, "daily_time": "", "max_messages": 300, "window_size": 60},
        )
        self._append_n(store, 180)
        msgs = self._enable_progress(plugin)
        result = run(plugin._run_analysis("88880001", "测试", notify=True))
        assert "分析完成" in result
        progress = [m for m in msgs if "分析进度" in m]
        assert len(progress) == 1
        assert progress[0].endswith("分析进度：已处理 60/180 条（33%）")

    def test_batch_failure_tolerated(self, tmp_path: Path) -> None:
        """单批分析异常：跳过该批，其余批正常完成，分析不中断。"""
        plugin, store = self.make_env(
            tmp_path,
            analysis={"model": "", "notify_qq": "", "max_statements_per_scene": 30,
                      "analyze_concurrency": 2, "progress_interval_seconds": 0},
            collection={"auto_analyze_count": 0, "daily_time": "", "max_messages": 300, "window_size": 60},
        )
        self._append_n(store, 180)  # 3 批
        calls: List[int] = []

        async def flaky_analyze_window(window: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            calls.append(len(window))
            if len(calls) == 2:  # 第 2 批抛异常
                raise RuntimeError("模拟 LLM 网络错误")
            result = json.loads(json.dumps(FakeAnalyzer.analyze_result))
            result["scenes"] = [{"name": "S", "statements": [{"text": m["text"]} for m in window]}]
            return result

        plugin._analyzer.analyze_window = flaky_analyze_window  # type: ignore[method-assign]
        result = run(plugin._run_analysis("88880001", "测试", notify=True))
        assert "分析完成" in result  # 其余批正常融合完成
        model = store.list_models()[0]
        assert len(model["scenes"][0]["statements"]) >= 1  # 保留成功的批次内容

    def test_all_windows_fail_notifies(self, tmp_path: Path) -> None:
        """所有窗口分析都失败（如 LLM 余额不足）：notify 开启时私聊通知失败原因，不生成模型。"""
        plugin, store = self.make_env(
            tmp_path,
            analysis={"model": "", "notify_qq": "10086", "max_statements_per_scene": 30,
                      "analyze_concurrency": 2, "progress_interval_seconds": 0},
            collection={"auto_analyze_count": 0, "daily_time": "", "max_messages": 300, "window_size": 60},
        )
        self._append_n(store, 120)  # 2 批
        msgs = self._enable_progress(plugin)

        async def failing_analyze_window(window: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            return None  # 模拟 LLM 全部失败

        plugin._analyzer.analyze_window = failing_analyze_window  # type: ignore[method-assign]
        result = run(plugin._run_analysis("88880001", "测试", notify=True))
        assert "LLM 未返回有效结果" in result
        assert not store.list_models()  # 不生成模型
        assert store.get_group("88880001")["analyzing"] is False  # 状态复位
        assert any("分析失败" in m for m in msgs)  # 失败原因私聊通知

    def test_empty_messages_notifies(self, tmp_path: Path) -> None:
        """没有可分析对话：notify 开启时通知失败原因。"""
        plugin, _ = self.make_env(
            tmp_path,
            analysis={"model": "", "notify_qq": "10086", "max_statements_per_scene": 30},
        )
        msgs = self._enable_progress(plugin)
        result = run(plugin._run_analysis("88880001", "测试", notify=True))
        assert "还没有收集到可分析的对话" in result
        assert any("分析失败" in m for m in msgs)


class TestRewriteHook:
    def make_env(self, tmp_path: Path, **config_patch: Any) -> tuple[ChatStyleTranslatorPlugin, StyleStore, FakeLLM]:
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001"))
        run(store.save_model({
            "name": "群模型", "overall_style": "口语化", "members": [],
            "scenes": [{"name": "闲聊", "description": "", "style": "", "note": "",
                        "statements": [{"text": "咋回事捏", "speaker": "1", "speaker_name": "A",
                                        "target": "大家", "meaning": "疑问", "ai_note": "随便", "tags": []}]}],
        }))
        llm = FakeLLM()
        plugin = make_plugin(**config_patch)
        plugin._store = store
        plugin._session_map["sess-1"] = "88880001"

        async def fake_rewrite(
            reference: str,
            response: str,
            original_response: str = "",
            forbidden_words: Optional[List[str]] = None,
            *, incoming_text: str = "",
        ) -> Optional[str]:
            result = await llm.generate([{"role": "user", "content": response}])
            if not result.get("success"):
                return None
            text = str(result.get("response") or "").strip()
            return text or None

        plugin._rewrite_with_llm = fake_rewrite  # type: ignore[method-assign]
        return plugin, store, llm

    def mount(self, store: StyleStore) -> None:
        model_id = store.list_models()[0]["model_id"]
        run(store.set_group_field("88880001", "active_models", [model_id]))

    def test_no_inject_disabled(self, tmp_path: Path) -> None:
        plugin, store, _ = self.make_env(tmp_path, inject={"enabled": False})
        self.mount(store)
        result = run(plugin.rewrite_style_response(response="你好全世界", session_id="sess-1"))
        assert result == {"action": "continue"}

    def test_short_or_missing(self, tmp_path: Path) -> None:
        plugin, _, _ = self.make_env(tmp_path)
        assert run(plugin.rewrite_style_response(response="", session_id="sess-1")) == {"action": "continue"}
        assert run(plugin.rewrite_style_response(response="短", session_id="sess-1")) == {"action": "continue"}
        assert run(plugin.rewrite_style_response(response="你好全世界", session_id="")) == {"action": "continue"}
        assert run(plugin.rewrite_style_response(response="你好全世界", session_id="未知")) == {"action": "continue"}

    def test_no_active_models(self, tmp_path: Path) -> None:
        plugin, _, _ = self.make_env(tmp_path)
        assert run(plugin.rewrite_style_response(response="你好全世界", session_id="sess-1")) == {"action": "continue"}

    def test_rewrite_success(self, tmp_path: Path) -> None:
        plugin, store, llm = self.make_env(tmp_path)
        self.mount(store)
        llm.set_responses([{"success": True, "response": "你好全世界呀"}])
        result = run(plugin.rewrite_style_response(response="你好全世界", session_id="sess-1"))
        assert result["action"] == "continue"
        assert result["modified_kwargs"]["response"] == "你好全世界呀"

    def test_rewrite_same_text_no_change(self, tmp_path: Path) -> None:
        plugin, store, llm = self.make_env(tmp_path)
        self.mount(store)
        llm.set_responses([{"success": True, "response": "你好全世界"}])
        result = run(plugin.rewrite_style_response(response="你好全世界", session_id="sess-1"))
        assert "modified_kwargs" not in result

    def test_rewrite_failure_fallback(self, tmp_path: Path) -> None:
        plugin, store, llm = self.make_env(tmp_path)
        self.mount(store)
        llm.set_responses([{"success": False, "error": "boom"}])
        assert "modified_kwargs" not in run(
            plugin.rewrite_style_response(response="你好全世界", session_id="sess-1")
        )
        llm.set_responses([{"success": True, "response": "你"}])  # 长度比例 0.2 < 0.5，安全检查不过
        assert "modified_kwargs" not in run(
            plugin.rewrite_style_response(response="你好全世界", session_id="sess-1")
        )

    def test_rewrite_concurrent_guard(self, tmp_path: Path) -> None:
        plugin, store, _ = self.make_env(tmp_path)
        self.mount(store)
        plugin._rewriting.add("sess-1")
        result = run(plugin.rewrite_style_response(response="你好全世界", session_id="sess-1"))
        assert "modified_kwargs" not in result

    def test_rewriting_released_on_failure(self, tmp_path: Path) -> None:
        """微调执行体异常时，会话标记必须被 finally 释放，避免后续回复被永久跳过。"""
        plugin, store, _ = self.make_env(tmp_path)
        self.mount(store)

        async def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("boom")

        plugin._rewrite_with_llm = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            run(plugin.rewrite_style_response(response="你好全世界", session_id="sess-1"))
        assert "sess-1" not in plugin._rewriting

    def test_rewrite_timeout_falls_back(self, tmp_path: Path) -> None:
        plugin, store, _ = self.make_env(tmp_path, inject={"rewrite_timeout_seconds": 3.0})
        self.mount(store)
        plugin._set_context(FakeCtx(tmp_path))
        # 恢复真实的 _rewrite_with_llm（内部带超时）
        plugin._rewrite_with_llm = ChatStyleTranslatorPlugin._rewrite_with_llm.__get__(  # type: ignore[method-assign]
            plugin, ChatStyleTranslatorPlugin
        )

        async def slow_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            await asyncio.sleep(10)
            return {"success": True, "response": "你好全世界呀"}

        plugin.ctx.llm.generate = slow_generate  # type: ignore[attr-defined]
        result = run(
            asyncio.wait_for(
                plugin._rewrite_with_llm("参考文本", "你好全世界"),
                timeout=8,
            )
        )
        assert result is None  # 内部超时后返回 None

    def test_hook_timeout_falls_back(self, tmp_path: Path) -> None:
        """微调 Hook 整体超时（hook_timeout_seconds）生效：超时跳过微调返回原回复。"""
        plugin, store, _ = self.make_env(tmp_path, inject={"hook_timeout_seconds": 3.0})
        self.mount(store)
        plugin._set_context(FakeCtx(tmp_path))
        plugin._rewrite_with_llm = ChatStyleTranslatorPlugin._rewrite_with_llm.__get__(  # type: ignore[method-assign]
            plugin, ChatStyleTranslatorPlugin
        )

        async def slow_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            await asyncio.sleep(10)
            return {"success": True, "response": "你好全世界呀"}

        plugin.ctx.llm.generate = slow_generate  # type: ignore[attr-defined]
        start = time.monotonic()
        result = run(
            asyncio.wait_for(
                plugin.rewrite_style_response(response="你好全世界", session_id="sess-1"),
                timeout=8,
            )
        )
        elapsed = time.monotonic() - start
        assert result == {"action": "continue"}  # 整体超时后跳过微调、原回复不变
        assert elapsed < 8  # 没有被慢 LLM 拖到 10 秒，整体超时确实生效


class TestAutoEnableModels:
    """目录模式和手动选择模式。"""

    def make_env(self, tmp_path: Path, auto_enable: bool = True) -> tuple[ChatStyleTranslatorPlugin, StyleStore]:
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001"))
        run(store.save_model({
            "name": "自动模型", "status": "pending", "overall_style": "口语化",
            "members": [], "scenes": [],
        }))
        plugin = make_plugin(plugin={"enabled": True, "config_version": "0.18.0", "auto_enable_models": auto_enable})
        plugin._store = store
        return plugin, store

    def test_directory_includes_all_models_and_restores_manual_selection(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path, auto_enable=True)
        pending = store.list_models()[0]["model_id"]
        run(store.save_model({"model_id": "approved", "name": "通用模型", "status": "approved"}))
        run(store.save_model({
            "model_id": "scoped", "name": "本群模型", "status": "approved",
            "group_id": "88880001", "group_only": True, "global_disabled": True,
        }))
        run(store.set_group_field("88880001", "active_models", ["scoped"]))
        raw_config = {
            "plugin": {"config_version": "0.20.18", "auto_enable_models": True},
            "inject": {"global_models": ["approved"]},
            "groups": {"items": [{"group_id": "88880001", "prefer_group_models": True}]},
        }
        plugin.set_plugin_config(raw_config)
        plugin._sync_config()
        run(plugin._refresh_global_models())

        assert plugin._global_models == []
        for group in ("88880001", "任意未绑定群"):
            assert set(plugin._resolve_active_models(group)) == {"approved", "scoped", pending}
        assert store.get_model(pending)["status"] == "pending"
        status = run(plugin.get_config_widget_state())["active_models"].text
        assert "目录自动生效　3 个模型" in status
        assert "群聊通用模型和群挂载暂时停用" in status
        assert plugin._active_model_source("approved", "88880001") == "目录自动生效"

        raw_config["plugin"]["auto_enable_models"] = False
        plugin.set_plugin_config(raw_config)
        plugin._sync_config()
        run(plugin._refresh_global_models())
        assert plugin._resolve_active_models("88880001") == ["scoped"]
        assert plugin._resolve_active_models("任意未绑定群") == ["approved"]

    def test_directory_updates_on_added_changed_and_deleted_models(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path, auto_enable=True)
        original = store.list_model_ids()[0]
        path = store.model_dir / "imported.json"
        path.write_text("invalid", encoding="utf-8")
        assert plugin._resolve_active_models("88880001") == [original]
        path.write_text('{"overall_style":"口语短句"}', encoding="utf-8")
        assert set(plugin._resolve_active_models("88880001")) == {"imported", original}
        path.unlink()
        assert plugin._resolve_active_models("88880001") == [original]

    def test_disabled_keeps_pending_and_not_global(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path, auto_enable=False)
        run(plugin._refresh_global_models())
        model_id = store.list_models()[0]["model_id"]
        file_id = store.find_model_file(model_id)
        assert file_id not in plugin._global_models
        assert plugin._resolve_active_models("88880001") == []
        # 关闭时不自动改状态，仍保持待审核
        assert store.get_model(file_id).get("status") == "pending"


class TestPreRewriteBlacklist:
    """微调前黑名单：草稿中出现黑名单词时，先在微调前删除再微调；微调失败仍返回删词后的文本。"""

    def make_env(self, tmp_path: Path, **config_patch: Any) -> tuple[ChatStyleTranslatorPlugin, StyleStore]:
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001"))
        run(store.save_model({
            "name": "群模型", "overall_style": "口语化", "members": [],
            "scenes": [{"name": "闲聊", "description": "", "style": "", "note": "",
                        "statements": [{"text": "咋回事捏", "speaker": "1", "speaker_name": "A",
                                        "target": "大家", "meaning": "疑问", "ai_note": "随便", "tags": []}]}],
        }))
        plugin = make_plugin(**config_patch)
        plugin._store = store
        plugin._session_map["sess-1"] = "88880001"
        return plugin, store

    def mount(self, store: StyleStore) -> None:
        model_id = store.list_models()[0]["model_id"]
        run(store.set_group_field("88880001", "active_models", [model_id]))

    def test_strip_units(self) -> None:
        plugin = make_plugin(pre_rewrite_blacklist={"enabled": True, "items": ["涩涩", "笨蛋"]})
        strip = plugin._strip_pre_rewrite_blacklist
        assert strip("今晚一起涩涩吗") == ("今晚一起吗", True)
        assert strip("你这个笨蛋！") == ("你这个", True)
        assert strip("你好呀") == ("你好呀", False)
        assert strip("涩") == ("涩", False)  # 单字不误伤
        assert strip("涩涩") == ("", True)
        assert strip("涩涩，你好") == ("你好", True)
        assert strip("你好，涩涩，世界") == ("你好，世界", True)  # 重复标点合并
        # 未启用 / 空列表时不处理
        plugin2 = make_plugin(pre_rewrite_blacklist={"enabled": False, "items": ["涩涩"]})
        assert plugin2._strip_pre_rewrite_blacklist("一起涩涩吗") == ("一起涩涩吗", False)
        plugin3 = make_plugin(pre_rewrite_blacklist={"enabled": True, "items": []})
        assert plugin3._strip_pre_rewrite_blacklist("一起涩涩吗") == ("一起涩涩吗", False)
        assert plugin3._strip_pre_rewrite_blacklist("") == ("", False)
        # 仅支持中文词条：纯英文/数字/符号/emoji 词条无效
        plugin4 = make_plugin(pre_rewrite_blacklist={"enabled": True, "items": ["lol", "😀", "123", "!!"]})
        assert plugin4._strip_pre_rewrite_blacklist("哈哈哈 lol 笑死") == ("哈哈哈 lol 笑死", False)
        assert plugin4._strip_pre_rewrite_blacklist("😀") == ("😀", False)
        assert plugin4._strip_pre_rewrite_blacklist("混搭涩涩") == ("混搭涩涩", False)  # 无中文词条 → 不处理
        plugin5 = make_plugin(pre_rewrite_blacklist={"enabled": True, "items": ["lol", "涩涩"]})
        assert plugin5._strip_pre_rewrite_blacklist("混搭涩涩") == ("混搭", True)  # 含中文的词条生效
        assert ChatStyleTranslatorPlugin._contains_cjk("涩涩") is True
        assert ChatStyleTranslatorPlugin._contains_cjk("混搭a1") is True
        assert ChatStyleTranslatorPlugin._contains_cjk("lol") is False
        assert ChatStyleTranslatorPlugin._contains_cjk("😀") is False
        assert ChatStyleTranslatorPlugin._contains_cjk("123") is False

    def test_sync_config(self) -> None:
        plugin = make_plugin(pre_rewrite_blacklist={
            "enabled": True, "items": ["涩涩", " 笨蛋 "], "semantic_match": True,
            "rewrite_style": "温柔", "rewrite_style_custom": "像东北老铁唠嗑",
        })
        assert plugin._pre_rewrite_blacklist_enabled is True
        assert plugin._pre_rewrite_blacklist == ["涩涩", "笨蛋"]
        assert plugin._pre_rewrite_semantic_match is True
        assert plugin._pre_rewrite_style == "温柔"
        assert plugin._pre_rewrite_style_custom == "像东北老铁唠嗑"
        plugin2 = make_plugin()
        assert plugin2._pre_rewrite_blacklist_enabled is False
        assert plugin2._pre_rewrite_blacklist == []
        assert plugin2._pre_rewrite_semantic_match is False
        assert plugin2._pre_rewrite_style == "无"
        assert plugin2._pre_rewrite_style_custom == ""

    def test_style_guide_resolution(self) -> None:
        plugin = make_plugin(pre_rewrite_blacklist={"rewrite_style": "温柔"})
        assert "温柔" in plugin._pre_rewrite_style_guide()
        assert "网络梗词" in plugin._pre_rewrite_style_guide()  # 温柔避免网络梗词
        plugin2 = make_plugin(pre_rewrite_blacklist={"rewrite_style": "冷漠"})
        assert "冷淡" in plugin2._pre_rewrite_style_guide()
        assert "禁止网络梗词" in plugin2._pre_rewrite_style_guide()  # 冷漠禁用网络梗词
        plugin6 = make_plugin(pre_rewrite_blacklist={"rewrite_style": "沉稳"})
        assert "禁止网络梗词" in plugin6._pre_rewrite_style_guide()  # 沉稳禁用网络梗词
        plugin7 = make_plugin(pre_rewrite_blacklist={"rewrite_style": "活泼"})
        assert "适当玩梗" in plugin7._pre_rewrite_style_guide()  # 活泼允许玩梗
        plugin3 = make_plugin(pre_rewrite_blacklist={"rewrite_style": "无"})
        assert plugin3._pre_rewrite_style_guide() == ""
        # 自定义：有描述 → 返回描述文本；留空 → 不注入
        plugin4 = make_plugin(pre_rewrite_blacklist={
            "rewrite_style": "自定义", "rewrite_style_custom": "像东北老铁唠嗑",
        })
        assert "像东北老铁唠嗑" in plugin4._pre_rewrite_style_guide()
        plugin5 = make_plugin(pre_rewrite_blacklist={"rewrite_style": "自定义", "rewrite_style_custom": "  "})
        assert plugin5._pre_rewrite_style_guide() == ""

    def test_no_symbol_emotion_default_and_sync(self) -> None:
        """「不用符号表达情绪」开关默认开启，配置可关闭/打开。"""
        plugin = make_plugin()
        assert plugin._no_symbol_emotion is True  # 默认开启
        off = make_plugin(user_block={"no_symbol_emotion": False})
        assert off._no_symbol_emotion is False
        on = make_plugin(user_block={"no_symbol_emotion": True})
        assert on._no_symbol_emotion is True

    def test_rewrite_prompt_contains_no_symbol_rule(self, tmp_path: Path) -> None:
        """风格微调提示词默认注入「不用符号表达情绪」约束（默认开启）。"""
        plugin, _ = self.make_env(tmp_path)
        plugin._set_context(FakeCtx(tmp_path))
        plugin._rewrite_with_llm = ChatStyleTranslatorPlugin._rewrite_with_llm.__get__(  # type: ignore[method-assign]
            plugin, ChatStyleTranslatorPlugin
        )
        seen_system: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen_system.append(str(messages[0].get("content") or ""))
            return {"success": True, "response": "你好呀"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        assert run(plugin._rewrite_with_llm("参考文本", "你好世界")) == "你好呀"
        assert "波浪号" in seen_system[0] and "……" in seen_system[0]

    def test_rewrite_prompt_omits_no_symbol_rule_when_disabled(self, tmp_path: Path) -> None:
        """关闭「不用符号表达情绪」后，微调提示词不再注入该约束。"""
        plugin, _ = self.make_env(
            tmp_path,
            user_block={"no_symbol_emotion": False},
        )
        plugin._set_context(FakeCtx(tmp_path))
        plugin._rewrite_with_llm = ChatStyleTranslatorPlugin._rewrite_with_llm.__get__(  # type: ignore[method-assign]
            plugin, ChatStyleTranslatorPlugin
        )
        seen_system: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen_system.append(str(messages[0].get("content") or ""))
            return {"success": True, "response": "你好呀"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        assert run(plugin._rewrite_with_llm("参考文本", "你好世界")) == "你好呀"
        assert "波浪号" not in seen_system[0]

    def test_semantic_remove_prompt_contains_no_symbol_rule(self, tmp_path: Path) -> None:
        """删词补意提示词默认注入「不用符号表达情绪」约束（默认开启）。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["属实"]},
        )
        plugin._set_context(FakeCtx(tmp_path))
        seen_system: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen_system.append(str(messages[0].get("content") or ""))
            return {"success": True, "response": "这波是躺赢进世界赛了啊"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        run(plugin._semantic_remove_keep_meaning("这波属实躺赢进世界赛了啊", "这波躺赢进世界赛了啊", ["属实"]))
        assert "波浪号" in seen_system[0] and "……" in seen_system[0]

    def test_semantic_remove_prompt_omits_no_symbol_rule_when_disabled(self, tmp_path: Path) -> None:
        """关闭「不用符号表达情绪」后，删词补意提示词不再注入该约束。"""
        plugin, _ = self.make_env(
            tmp_path,
            user_block={"no_symbol_emotion": False},
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["属实"]},
        )
        plugin._set_context(FakeCtx(tmp_path))
        seen_system: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen_system.append(str(messages[0].get("content") or ""))
            return {"success": True, "response": "这波是躺赢进世界赛了啊"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        run(plugin._semantic_remove_keep_meaning("这波属实躺赢进世界赛了啊", "这波躺赢进世界赛了啊", ["属实"]))
        assert "波浪号" not in seen_system[0]

    def test_removed_before_rewrite(self, tmp_path: Path) -> None:
        """命中黑名单：先删词再微调，LLM 收到的草稿已不含黑名单词。"""
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "items": ["涩涩"]},
        )
        self.mount(store)
        seen: List[str] = []

        async def fake_rewrite(
            reference: str,
            response: str,
            original_response: str = "",
            forbidden_words: Optional[List[str]] = None,
            *, incoming_text: str = "",
        ) -> Optional[str]:
            seen.append(response)
            return "今晚一块儿吗"

        plugin._rewrite_with_llm = fake_rewrite  # type: ignore[method-assign]
        result = run(plugin.rewrite_style_response(response="今晚一起涩涩吗", session_id="sess-1"))
        assert seen == ["今晚一起吗"]
        assert result["modified_kwargs"]["response"] == "今晚一块儿吗"

    def test_failure_returns_cleaned(self, tmp_path: Path) -> None:
        """命中黑名单但微调失败：仍返回删词后的文本，黑名单词不出现。"""
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "items": ["涩涩"]},
        )
        self.mount(store)

        async def fail_rewrite(
            reference: str,
            response: str,
            original_response: str = "",
            forbidden_words: Optional[List[str]] = None,
            *, incoming_text: str = "",
        ) -> Optional[str]:
            return None

        plugin._rewrite_with_llm = fail_rewrite  # type: ignore[method-assign]
        result = run(plugin.rewrite_style_response(response="今晚一起涩涩吗", session_id="sess-1"))
        assert result["modified_kwargs"]["response"] == "今晚一起吗"

    def test_failure_without_hit_keeps_original(self, tmp_path: Path) -> None:
        """未命中黑名单且微调失败：退回原回复（与旧逻辑一致）。"""
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "items": ["涩涩"]},
        )
        self.mount(store)

        async def fail_rewrite(
            reference: str,
            response: str,
            original_response: str = "",
            forbidden_words: Optional[List[str]] = None,
            *, incoming_text: str = "",
        ) -> Optional[str]:
            return None

        plugin._rewrite_with_llm = fail_rewrite  # type: ignore[method-assign]
        result = run(plugin.rewrite_style_response(response="今晚一起玩吗", session_id="sess-1"))
        assert "modified_kwargs" not in result

    def test_no_active_models(self, tmp_path: Path) -> None:
        """未挂载模型时不微调，但命中黑名单仍返回删词后的文本。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "items": ["涩涩"]},
        )
        result = run(plugin.rewrite_style_response(response="今晚一起涩涩吗", session_id="sess-1"))
        assert result["modified_kwargs"]["response"] == "今晚一起吗"

    def test_disabled_keeps_word(self, tmp_path: Path) -> None:
        """开关关闭时黑名单不生效。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": False, "items": ["涩涩"]},
        )
        assert run(plugin.rewrite_style_response(response="今晚一起涩涩吗", session_id="sess-1")) == {
            "action": "continue"
        }

    def test_disabled_no_semantic_llm_call(self, tmp_path: Path) -> None:
        """开关关闭时，即使语义匹配开启也不调用 LLM（防止白耗调用/意外改词）。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": False, "semantic_match": True, "items": ["属实"]},
        )
        plugin._set_context(FakeCtx(tmp_path))
        called: List[bool] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            called.append(True)
            return {"success": True, "response": "今晚一起吗"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        result = run(plugin.rewrite_style_response(response="今晚一起属实吗", session_id="sess-1"))
        assert called == []  # 语义 LLM 未被调用
        assert "modified_kwargs" not in result  # 文本未被改动

    def test_strip_to_empty_aborts_response(self, tmp_path: Path) -> None:
        """删词后无内容可发，终止回复。"""
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "items": ["涩涩"]},
        )
        self.mount(store)
        result = run(plugin.rewrite_style_response(response="涩涩涩涩", session_id="sess-1"))
        assert result["action"] == "abort" and "黑名单过滤" in result["custom_result"]["reason"]

    def test_rewrite_success_with_hit(self, tmp_path: Path) -> None:
        """命中黑名单且微调成功：返回微调后的文本（安全检查以删词后草稿为基准）。"""
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "items": ["涩涩"]},
        )
        self.mount(store)

        async def fake_rewrite(
            reference: str,
            response: str,
            original_response: str = "",
            forbidden_words: Optional[List[str]] = None,
            *, incoming_text: str = "",
        ) -> Optional[str]:
            return "今晚也一起吗？"

        plugin._rewrite_with_llm = fake_rewrite  # type: ignore[method-assign]
        result = run(plugin.rewrite_style_response(response="今晚一起涩涩吗", session_id="sess-1"))
        assert result["modified_kwargs"]["response"] == "今晚也一起吗？"

    def test_no_hit_unchanged(self, tmp_path: Path) -> None:
        """未命中黑名单时按原逻辑微调。"""
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "items": ["涩涩"]},
        )
        self.mount(store)

        async def fake_rewrite(
            reference: str,
            response: str,
            original_response: str = "",
            forbidden_words: Optional[List[str]] = None,
            *, incoming_text: str = "",
        ) -> Optional[str]:
            return "你好全世界呀"

        plugin._rewrite_with_llm = fake_rewrite  # type: ignore[method-assign]
        result = run(plugin.rewrite_style_response(response="你好全世界", session_id="sess-1"))
        assert result["modified_kwargs"]["response"] == "你好全世界呀"


class TestPreRewriteSemanticMatch:
    """微调前黑名单「语义换词匹配（保持原意）」：基于没删前的整句话理解意思后删除黑名单词，
    删除后尽量保持原意（可补字词）；语义换词匹配生效后跳过回复后风格微调，
    仅字面删词命中（或语义删除退回字面结果）时仍继续回复后微调。"""

    def make_env(self, tmp_path: Path, **config_patch: Any) -> tuple[ChatStyleTranslatorPlugin, StyleStore]:
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001"))
        run(store.save_model({
            "name": "群模型", "overall_style": "口语化", "members": [],
            "scenes": [{"name": "闲聊", "description": "", "style": "", "note": "",
                        "statements": [{"text": "咋回事捏", "speaker": "1", "speaker_name": "A",
                                        "target": "大家", "meaning": "疑问", "ai_note": "随便", "tags": []}]}],
        }))
        plugin = make_plugin(**config_patch)
        plugin._store = store
        plugin._session_map["sess-1"] = "88880001"
        plugin._set_context(FakeCtx(tmp_path))
        return plugin, store

    def mount(self, store: StyleStore) -> None:
        model_id = store.list_models()[0]["model_id"]
        run(store.set_group_field("88880001", "active_models", [model_id]))

    def test_semantic_remove_paraphrase(self, tmp_path: Path) -> None:
        """字面未命中：LLM 基于原句理解，输出删除同义换词表述后的整句。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["骂人"]},
        )
        seen: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen.append(str(messages[1].get("content") or ""))
            return {"success": True, "response": "今晚一起吗"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        result = run(plugin._semantic_remove_keep_meaning("今晚一起嘴臭一下吗", "今晚一起嘴臭一下吗", ["骂人"]))
        assert result == "今晚一起吗"
        assert len(seen) == 1
        assert "骂人" in seen[0]
        assert "嘴臭一下" in seen[0]  # 原句传给 LLM

    def test_semantic_remove_keep_meaning(self, tmp_path: Path) -> None:
        """用户示例：字面命中「属实」后，LLM 基于没删前的整句补「是」保持原意。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["属实"]},
        )
        seen: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen.append(str(messages[1].get("content") or ""))
            return {"success": True, "response": "这波是躺赢进世界赛了啊"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        original = "这波属实躺赢进世界赛了啊"
        literal = "这波躺赢进世界赛了啊"
        result = run(plugin._semantic_remove_keep_meaning(original, literal, ["属实"]))
        assert result == "这波是躺赢进世界赛了啊"
        # LLM 同时看到「没删前的原句」和「字面删除后的草稿」
        assert original in seen[0] and literal in seen[0]

    def test_semantic_remove_allows_synonym(self, tmp_path: Path) -> None:
        """近义表达（如「难绷」）允许出现，只要字面不出现黑名单词原文。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["属实", "绷不住了"]},
        )

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            return {"success": True, "response": "零杠十一这战绩是真的难绷"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        result = run(plugin._semantic_remove_keep_meaning(
            "零杠十一属实是绷不住了", "零杠十一是", ["属实", "绷不住了"]
        ))
        assert result == "零杠十一这战绩是真的难绷"  # 不含「属实」「绷不住了」→ 采用

    def test_semantic_remove_style_injected(self, tmp_path: Path) -> None:
        """选择「温柔」风格时，语义删除提示词会注入整句语气贴合风格的风格要求。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "rewrite_style": "温柔", "items": ["属实"]},
        )
        seen_system: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen_system.append(str(messages[0].get("content") or ""))
            return {"success": True, "response": "零杠十一这战绩是真的难绷"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        run(plugin._semantic_remove_keep_meaning("零杠十一属实是绷不住了", "零杠十一是", ["属实"]))
        assert "温柔" in seen_system[0] and "语气与措辞" in seen_system[0]

    def test_semantic_remove_style_none_no_guide(self, tmp_path: Path) -> None:
        """风格为「无」时不注入任何风格要求。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "rewrite_style": "无", "items": ["属实"]},
        )
        seen_system: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen_system.append(str(messages[0].get("content") or ""))
            return {"success": True, "response": "零杠十一这战绩是真的难绷"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        run(plugin._semantic_remove_keep_meaning("零杠十一属实是绷不住了", "零杠十一是", ["属实"]))
        assert "风格要求" not in seen_system[0]

    def test_semantic_remove_custom_style_injected(self, tmp_path: Path) -> None:
        """选「自定义」并填写描述时，语义删除提示词按描述注入风格。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={
                "enabled": True, "semantic_match": True,
                "rewrite_style": "自定义", "rewrite_style_custom": "像东北老铁唠嗑",
                "items": ["属实"],
            },
        )
        seen_system: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen_system.append(str(messages[0].get("content") or ""))
            return {"success": True, "response": "零杠十一这战绩是真的难绷"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        run(plugin._semantic_remove_keep_meaning("零杠十一属实是绷不住了", "零杠十一是", ["属实"]))
        assert "像东北老铁唠嗑" in seen_system[0]

    def test_semantic_remove_custom_style_empty_no_guide(self, tmp_path: Path) -> None:
        """选「自定义」但描述留空：不注入任何风格要求。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={
                "enabled": True, "semantic_match": True,
                "rewrite_style": "自定义", "rewrite_style_custom": "   ",
                "items": ["属实"],
            },
        )
        seen_system: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            seen_system.append(str(messages[0].get("content") or ""))
            return {"success": True, "response": "零杠十一这战绩是真的难绷"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        run(plugin._semantic_remove_keep_meaning("零杠十一属实是绷不住了", "零杠十一是", ["属实"]))
        assert "风格要求" not in seen_system[0]

    def test_semantic_remove_no_hit(self, tmp_path: Path) -> None:
        """没有黑名单表述时 LLM 原样输出，等于没删。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["骂人"]},
        )
        plugin.ctx.llm.generate = lambda messages, **kwargs: {"success": True, "response": "今晚一起玩吗"}  # type: ignore[attr-defined]
        assert run(plugin._semantic_remove_keep_meaning("今晚一起玩吗", "今晚一起玩吗", ["骂人"])) == "今晚一起玩吗"

    def test_semantic_remove_output_contains_word_falls_back(self, tmp_path: Path) -> None:
        """LLM 输出仍含黑名单词：退回字面删除结果。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["属实"]},
        )
        plugin.ctx.llm.generate = lambda messages, **kwargs: {"success": True, "response": "属实，这波是躺赢进世界赛了啊"}  # type: ignore[attr-defined]
        result = run(plugin._semantic_remove_keep_meaning(
            "这波属实躺赢进世界赛了啊", "这波躺赢进世界赛了啊", ["属实"]
        ))
        assert result == "这波躺赢进世界赛了啊"

    def test_semantic_remove_failure_keeps_text(self, tmp_path: Path) -> None:
        """LLM 调用失败：退回字面删除结果，不抛异常。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["骂人"]},
        )

        async def fail_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("llm 挂了")

        plugin.ctx.llm.generate = fail_generate  # type: ignore[attr-defined]
        assert run(plugin._semantic_remove_keep_meaning(
            "今晚一起嘴臭一下吗", "今晚一起嘴臭一下吗", ["骂人"]
        )) == "今晚一起嘴臭一下吗"

    def test_semantic_remove_too_short_falls_back(self, tmp_path: Path) -> None:
        """LLM 输出过短（比例 < 50%）：退回字面删除结果，防止误删。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["骂人"]},
        )
        plugin.ctx.llm.generate = lambda messages, **kwargs: {"success": True, "response": "吗"}  # type: ignore[attr-defined]
        assert run(plugin._semantic_remove_keep_meaning(
            "今晚一起嘴臭一下吗", "今晚一起嘴臭一下吗", ["骂人"]
        )) == "今晚一起嘴臭一下吗"

    def test_semantic_remove_non_chinese_words_skipped(self, tmp_path: Path) -> None:
        """纯英文/emoji 词条不参与语义删除（不调用 LLM）。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["lol", "😀"]},
        )
        called: List[bool] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            called.append(True)
            return {"success": True, "response": "哈哈哈"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        result = run(plugin._semantic_remove_keep_meaning("哈哈哈 lol 笑死", "哈哈哈 lol 笑死", ["lol", "😀"]))
        assert result == "哈哈哈 lol 笑死"
        assert called == []

    def test_hook_semantic_hit_skips_rewrite(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["骂人"]},
        )
        self.mount(store)
        calls: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            calls.append(messages[1]["content"])
            return {"success": True, "response": "[[SEMANTIC_BLACKLIST]]\n今晚一起吗"}

        plugin.ctx.llm.generate = fake_generate
        result = run(plugin.rewrite_style_response(response="今晚一起嘴臭一下吗", session_id="sess-1"))
        assert len(calls) == 1 and "语义黑名单词条" in calls[0] and "群聊风格资料" in calls[0]
        assert result["modified_kwargs"]["response"] == "今晚一起吗"
        assert plugin._last_rewrite_trace["reason"] == "已处理语义黑名单"

    def test_hook_literal_hit_keeps_meaning_without_second_call(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["属实"]},
        )
        self.mount(store)
        calls: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            calls.append(messages[1]["content"])
            return {"success": True, "response": "[[SEMANTIC_BLACKLIST]]\n这波是躺赢进世界赛了啊"}

        plugin.ctx.llm.generate = fake_generate
        result = run(plugin.rewrite_style_response(response="这波属实躺赢进世界赛了啊", session_id="sess-1"))
        assert result["modified_kwargs"]["response"] == "这波是躺赢进世界赛了啊"
        assert len(calls) == 1 and "删除前的原句" in calls[0]

    def test_hook_failed_request_returns_literal_filter_without_retry(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["属实"]},
        )
        self.mount(store)
        calls: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            calls.append(messages[1]["content"])
            return {"success": False}

        plugin.ctx.llm.generate = fake_generate
        result = run(plugin.rewrite_style_response(response="这波属实躺赢进世界赛了啊", session_id="sess-1"))
        assert result["modified_kwargs"]["response"] == "这波躺赢进世界赛了啊"
        assert len(calls) == 1

    def test_hook_only_literal_hit_can_still_rewrite_with_one_call(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": True, "items": ["涩涩"]},
        )
        self.mount(store)
        calls: List[str] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            calls.append(messages[1]["content"])
            return {"success": True, "response": "今晚一块儿吗"}

        plugin.ctx.llm.generate = fake_generate
        result = run(plugin.rewrite_style_response(response="今晚一起涩涩吗", session_id="sess-1"))
        assert result["modified_kwargs"]["response"] == "今晚一块儿吗"
        assert len(calls) == 1 and "涩涩" in calls[0]

    def test_hook_rewrite_output_rescans_blacklist(self, tmp_path: Path) -> None:
        """微调结果仍含黑名单词（LLM 恢复原句用词）：兜底再删一次。"""
        plugin, store = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "items": ["涩涩"]},
        )
        self.mount(store)

        async def fake_rewrite(
            reference: str,
            response: str,
            original_response: str = "",
            forbidden_words: Optional[List[str]] = None,
            *, incoming_text: str = "",
        ) -> Optional[str]:
            return "今晚一起涩涩吧！"  # LLM 把黑名单词写回来了

        plugin._rewrite_with_llm = fake_rewrite  # type: ignore[method-assign]
        result = run(plugin.rewrite_style_response(response="今晚一起涩涩吧", session_id="sess-1"))
        assert result["modified_kwargs"]["response"] == "今晚一起吧"  # 兜底删除后发送

    def test_hook_semantic_disabled_no_llm_call(self, tmp_path: Path) -> None:
        """语义匹配开关关闭时，字面未命中不调用 LLM。"""
        plugin, _ = self.make_env(
            tmp_path,
            pre_rewrite_blacklist={"enabled": True, "semantic_match": False, "items": ["骂人"]},
        )
        called: List[bool] = []

        async def fake_generate(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
            called.append(True)
            return {"success": True, "response": "今晚一起吗"}

        plugin.ctx.llm.generate = fake_generate  # type: ignore[attr-defined]
        result = run(plugin.rewrite_style_response(response="今晚一起嘴臭一下吗", session_id="sess-1"))
        assert called == []
        assert "modified_kwargs" not in result  # 未命中且未开启语义匹配，原回复不变


class TestLifecycle:
    def test_load_unload(self, tmp_path: Path) -> None:
        ctx = FakeCtx(tmp_path)
        plugin = ChatStyleTranslatorPlugin()
        plugin._set_context(ctx)
        plugin.set_plugin_config({"plugin": {"enabled": True, "config_version": "0.9.0"}})

        async def scenario() -> None:
            await plugin.on_load()
            assert plugin._store is not None
            assert plugin._analyzer is not None
            assert plugin._loop_task is not None and not plugin._loop_task.done()
            await plugin.on_unload()
            assert plugin._loop_task is None

        run(scenario())

    def test_on_config_update(self, tmp_path: Path) -> None:
        ctx = FakeCtx(tmp_path)
        plugin = ChatStyleTranslatorPlugin()
        plugin._set_context(ctx)
        plugin.set_plugin_config({"plugin": {"enabled": True, "config_version": "0.9.0"}})
        run(plugin.on_load())
        plugin.set_plugin_config({
            "plugin": {"enabled": True, "config_version": "0.9.0"},
            "phrase_blacklist": {"items": ["修仙"]},
        })
        run(plugin.on_config_update("self", {}, "0.9.0"))
        assert plugin._phrase_blacklist == ["修仙"]
        run(plugin.on_unload())

    def test_refresh_session_map_learns_account(self, tmp_path: Path) -> None:
        ctx = FakeCtx(tmp_path)
        ctx._streams_result = [
            {"session_id": "s1", "stream_id": "s1", "group_id": "88880001", "account_id": "10001"},
            {"session_id": "s2", "group_id": "", "account_id": "10001"},
        ]
        plugin = ChatStyleTranslatorPlugin()
        plugin._set_context(ctx)
        plugin.set_plugin_config({"plugin": {"enabled": True, "config_version": "0.9.0"}})
        plugin._store = StyleStore(tmp_path)
        run(plugin._refresh_session_map())
        assert plugin._session_map["s1"] == "88880001"
        assert plugin._stream_map["88880001"] == "s1"
        assert "10001" in plugin._self_account_ids

    def test_sync_groups_from_config(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.save_model({"name": "群模型", "scenes": [], "members": []}))
        run(store.bind_group("999", source="command"))  # 命令绑定群不受配置同步影响
        plugin = make_plugin(groups={"items": [
            {"group_id": "111", "platform": "qq", "models": ["群模型"]},
        ]})
        plugin._store = store
        run(plugin._sync_groups_from_config())
        assert store.is_bound("111")
        assert store.get_group("111")["active_models"] == [store.list_models()[0]["model_id"]]
        assert store.is_bound("999")
        # 从配置删除群自动解绑
        plugin2 = make_plugin(groups={"items": []})
        plugin2._store = store
        run(plugin2._sync_groups_from_config())
        assert not store.is_bound("111")
        assert store.is_bound("999")

    def test_sync_blacklist_from_config(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.save_model({"name": "群模型", "scenes": [], "members": []}))
        plugin = make_plugin(blacklist={"items": [
            {"text": "脏话", "meaning": "", "model": "群模型", "apply_all": False},
        ]})
        plugin._store = store
        run(plugin._sync_blacklist_from_config())
        item = store.list_blacklist()[0]
        assert item["model_id"] == store.list_models()[0]["model_id"]
        assert item["model_name"] == "群模型"

    def test_manual_analyze_trigger_button(self, tmp_path: Path) -> None:
        """WebUI「立即执行分析」开关打开并保存配置后，触发全部绑定群的后台分析。"""
        import time

        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", "测试群"))
        run(store.bind_group("88880002", "测试群2"))
        plugin = make_plugin(collection={
            "trigger_analyze_now": True,
            "max_messages": 300,
            "window_size": 60,
            "auto_analyze_count": 0,
            "daily_time": "",
        })
        plugin._store = store
        plugin._last_manual_trigger_at = 0.0
        triggered: List[str] = []

        async def fake_trigger(group_id: str, reason: str, *, force: bool = False) -> None:
            assert force
            triggered.append(group_id)

        plugin._trigger_auto_analyze = fake_trigger  # type: ignore[method-assign]
        run(plugin._maybe_manual_analyze_trigger())
        assert sorted(triggered) == ["88880001", "88880002"]

        # 30 秒防抖：连续再次触发不重复
        plugin._last_manual_trigger_at = time.time()
        triggered.clear()
        run(plugin._maybe_manual_analyze_trigger())
        assert triggered == []

    def test_manual_analyze_trigger_off(self, tmp_path: Path) -> None:
        """开关关闭时保存配置不触发分析。"""
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", "测试群"))
        plugin = make_plugin(collection={
            "trigger_analyze_now": False,
            "max_messages": 300,
            "window_size": 60,
            "auto_analyze_count": 0,
            "daily_time": "",
        })
        plugin._store = store
        plugin._last_manual_trigger_at = 0.0
        triggered: List[str] = []

        async def fake_trigger(group_id: str, reason: str) -> None:
            triggered.append(group_id)

        plugin._trigger_auto_analyze = fake_trigger  # type: ignore[method-assign]
        run(plugin._maybe_manual_analyze_trigger())
        assert triggered == []

    def test_manual_analyze_trigger_no_groups(self, tmp_path: Path) -> None:
        """开关打开但没有任何绑定群时静默跳过。"""
        store = StyleStore(tmp_path)
        plugin = make_plugin(collection={
            "trigger_analyze_now": True,
            "max_messages": 300,
            "window_size": 60,
            "auto_analyze_count": 0,
            "daily_time": "",
        })
        plugin._store = store
        plugin._last_manual_trigger_at = 0.0
        triggered: List[str] = []

        async def fake_trigger(group_id: str, reason: str) -> None:
            triggered.append(group_id)

        plugin._trigger_auto_analyze = fake_trigger  # type: ignore[method-assign]
        run(plugin._maybe_manual_analyze_trigger())
        assert triggered == []

    def test_notify_analysis_private(self, tmp_path: Path) -> None:
        """配置 notify_qq 后，进度/结果私聊发送给指定 QQ（不再发到群里）。"""
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", group_name="测试群"))
        plugin = make_plugin(analysis={"model": "", "notify_qq": "10086", "max_statements_per_scene": 30})
        ctx = FakeCtx(tmp_path)
        plugin._set_context(ctx)
        plugin._store = store
        run(plugin._notify_analysis("88880001", "正在分析…第 1/2 批（50%）"))
        assert len(ctx._api_calls) == 1
        call = ctx._api_calls[0]
        assert call["method"] == "adapter.napcat.message.send_msg"
        assert call["params"]["message_type"] == "private"
        assert call["params"]["user_id"] == 10086
        text = call["params"]["message"][0]["data"]["text"]
        assert "测试群" in text and "50%" in text

    def test_notify_analysis_no_qq(self, tmp_path: Path) -> None:
        """未配置 notify_qq 时不发送任何通知。"""
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", group_name="测试群"))
        plugin = make_plugin(analysis={"model": "", "notify_qq": "", "max_statements_per_scene": 30})
        ctx = FakeCtx(tmp_path)
        plugin._set_context(ctx)
        plugin._store = store
        run(plugin._notify_analysis("88880001", "正在分析…第 1/2 批（50%）"))
        assert ctx._api_calls == []

    def test_send_private_not_friend(self, tmp_path: Path) -> None:
        """非好友/发送失败时静默返回 False，不抛异常。"""
        store = StyleStore(tmp_path)
        plugin = make_plugin(analysis={"model": "", "notify_qq": "", "max_statements_per_scene": 30})
        ctx = FakeCtx(tmp_path)
        plugin._set_context(ctx)
        plugin._store = store

        async def fail_call(method: str, **kwargs: Any) -> Any:
            raise RuntimeError("非好友")

        ctx.api.call = fail_call  # type: ignore[method-assign]
        assert run(plugin._send_private("10086", "测试")) is False
        # 非法 QQ 号直接返回 False
        assert run(plugin._send_private("", "测试")) is False
        assert run(plugin._send_private("abc", "测试")) is False


class HistoryFakeCtx(FakeCtx):
    """带 NapCat 历史消息分页响应的假上下文。

    pages: 共享页面队列（所有群共用）；pages_by_group: 按群号分发页面（优先）。
    """

    def __init__(
        self,
        data_dir: Path,
        pages: Optional[List[Dict[str, Any]]] = None,
        pages_by_group: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__(data_dir)
        self._history_pages = list(pages or [])
        self._history_pages_by_group = {str(k): list(v) for k, v in (pages_by_group or {}).items()}

    async def _api_call(self, method: str, **kwargs: Any) -> Any:
        self._api_calls.append({"method": method, **kwargs})
        if method == "adapter.napcat.message.get_group_msg_history":
            group_id = str((kwargs.get("params") or {}).get("group_id") or "")
            pages = self._history_pages_by_group.get(group_id, self._history_pages)
            if pages:
                return {"data": {"messages": pages.pop(0)}}
            return {"data": {"messages": []}}
        return {"data": {"message_id": "1"}}


class TestHistoryBackfill:
    """历史消息回填：NapCat 消息转换、分页拉取过滤与去重、回填入库、自动分析触发。"""

    @staticmethod
    def napcat_message(
        message_id: str,
        seq: str,
        user_id: str,
        nickname: str,
        text: str,
        ts: float = 0.0,
    ) -> Dict[str, Any]:
        """构造 NapCat 群历史消息（NapCat 中 message_id 与 message_seq 是同一个短 ID）。"""
        return {
            "message_id": message_id,
            "message_seq": seq,
            "user_id": user_id,
            "sender": {"user_id": user_id, "nickname": nickname, "card": f"卡{nickname}"},
            "time": ts,
            "message": [{"type": "text", "data": {"text": text}}],
        }

    @staticmethod
    def collection_config(**patch: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "auto_analyze_count": 0,
            "daily_time": "",
            "max_messages": 300,
            "window_size": 60,
            "backfill_count": 100,
            "backfill_batch_size": 50,
            "trigger_analyze_now": False,
            "trigger_backfill_now": False,
        }
        base.update(patch)
        return base

    def test_napcat_message_to_record(self) -> None:
        raw = {
            "message_id": "100",
            "message_seq": 5,
            "user_id": "20001",
            "sender": {"user_id": "20001", "nickname": "昵称", "card": "卡昵称"},
            "time": 1700000000,
            "message": [
                {"type": "text", "data": {"text": "今天"}},
                {"type": "at", "data": {"qq": "20002", "name": "小红"}},
                {"type": "reply", "data": {"id": "999", "name": "小明"}},
                {"type": "face", "data": {"id": "1"}},
            ],
        }
        record = ChatStyleTranslatorPlugin._napcat_message_to_record(raw)
        assert record["text"] == "今天@小红[表情]"
        assert record["sender_id"] == "20001"
        assert record["sender_name"] == "卡昵称"
        assert record["at_user_id"] == "20002" and record["at_user_name"] == "小红"
        assert record["reply_to_id"] == "999" and record["reply_to_name"] == "小明"
        assert record["source"] == "history"
        assert record["ts"] == 1700000000.0
        assert ChatStyleTranslatorPlugin._napcat_message_to_record({}) is None
        assert ChatStyleTranslatorPlugin._napcat_message_to_record(
            {"message": [{"type": "image", "data": {}}]}
        ) is None

    def test_fetch_history_filters_and_dedup(self, tmp_path: Path) -> None:
        """分页拉取：屏蔽/黑名单/本人/机器人过滤，跨页按消息 ID 去重，按时间正序返回。

        NapCat 消息 ID 是哈希短 ID（数值与时间无关），翻页起点必须是最旧消息的 ID
        并配合 reverse_order=1 向前取；起点页无新消息 ID 时终止。
        """
        page1 = [
            self.napcat_message("m9", "m9", "20002", "B", "开始修仙了", ts=9),  # 字句黑名单过滤
            self.napcat_message("m10", "m10", "20001", "A", "/风格列表", ts=10),  # 带 / 过滤
        ]
        page2 = [
            self.napcat_message("m6", "m6", "10001", "self", "本人消息", ts=6),  # 本人过滤
            self.napcat_message("m7", "m7", "20003", "C", "正常聊天1", ts=7),
            self.napcat_message("m8", "m8", "888", "bot", "机器人话", ts=8),  # 机器人过滤
            self.napcat_message("m9", "m9", "20002", "B", "开始修仙了", ts=9),  # 与上页重复 -> 去重
        ]
        page3 = [
            self.napcat_message("m6", "m6", "10001", "self", "本人消息", ts=6),  # 起点无新消息 -> 终止
        ]
        ctx = HistoryFakeCtx(tmp_path, [page1, page2, page3])
        plugin = make_plugin(
            phrase_blacklist={"items": ["修仙"]},
            user_block={"block_slash": True, "block_qq_bots": True, "bot_ids": ["888"], "skip_self_messages": True},
        )
        plugin._set_context(ctx)
        plugin._self_account_ids.add("10001")
        records = run(plugin._fetch_group_history("88880001", count=100, batch_size=50))
        assert [r["text"] for r in records] == ["正常聊天1"]
        assert records[0]["ts"] == 7
        calls = [c for c in ctx._api_calls if c["method"] == "adapter.napcat.message.get_group_msg_history"]
        assert len(calls) == 3
        assert calls[0]["params"]["reverse_order"] == 1  # 反向翻页：向前（更早）取
        assert "message_seq" not in calls[0]["params"]  # 首次不带起点
        assert calls[1]["params"]["message_seq"] == "m9"  # 起点 = 本页时间最旧消息的 ID
        assert calls[2]["params"]["message_seq"] == "m6"

    def test_run_backfill_success(self, tmp_path: Path) -> None:
        """回填入库：记录带 source=history，按消息 ID 去重，记录 last_backfilled_at。"""
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", group_name="测试群"))
        page = [
            self.napcat_message("m1", "m1", "20001", "A", "第一条", ts=2),
            self.napcat_message("m2", "m2", "20002", "B", "第二条", ts=3),
        ]
        ctx = HistoryFakeCtx(tmp_path, [page])
        plugin = make_plugin(collection=self.collection_config())
        plugin._set_context(ctx)
        plugin._store = store
        result = run(plugin.run_backfill("88880001"))
        assert "拉取群 88880001 历史消息 2 条" in result
        assert "新增 2 条" in result
        assert store.message_count("88880001") == 2
        assert store.messages_for("88880001")[0]["source"] == "history"
        assert store.get_group("88880001")["last_backfilled_at"] > 0
        # 再次回填同一批：按消息 ID 去重，不重复收集
        ctx._history_pages = [page]
        result2 = run(plugin.run_backfill("88880001"))
        assert "新增 0 条" in result2
        assert store.message_count("88880001") == 2

    def test_run_backfill_auto_analyze(self, tmp_path: Path) -> None:
        """回填后收集条数达到 auto_analyze_count 自动触发后台分析。"""
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", group_name="测试群"))
        ctx = HistoryFakeCtx(tmp_path, [
            [
                self.napcat_message("m1", "m1", "20001", "A", "第一条", ts=1),
                self.napcat_message("m2", "m2", "20002", "B", "第二条", ts=2),
            ],
        ])
        plugin = make_plugin(collection=self.collection_config(auto_analyze_count=2))
        plugin._set_context(ctx)
        plugin._store = store
        triggered: List[str] = []

        async def fake_trigger(group_id: str, reason: str) -> None:
            triggered.append(group_id)

        plugin._trigger_auto_analyze = fake_trigger  # type: ignore[method-assign]
        result = run(plugin.run_backfill("88880001"))
        assert "已触发自动分析" in result
        assert triggered == ["88880001"]

    def test_run_backfill_errors(self, tmp_path: Path) -> None:
        """未绑定群、条数为 0、拉取不到历史时的提示。"""
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", group_name="测试群"))
        plugin = make_plugin(collection=self.collection_config(backfill_count=0))
        plugin._store = store
        assert "尚未绑定" in run(plugin.run_backfill("999"))
        assert "未指定拉取条数" in run(plugin.run_backfill("88880001"))
        plugin2 = make_plugin(collection=self.collection_config())
        ctx = HistoryFakeCtx(tmp_path, [])
        plugin2._set_context(ctx)
        plugin2._store = store
        assert "未拉取到历史消息" in run(plugin2.run_backfill("88880001"))

    def test_backfill_trigger_button(self, tmp_path: Path) -> None:
        """WebUI「立即执行历史回填」开关：打开并保存配置后触发全部绑定群回填，带防抖。"""
        import time

        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", group_name="测试群"))
        ctx = HistoryFakeCtx(tmp_path, [
            [self.napcat_message("m1", "m1", "20001", "A", "第一条", ts=1)],
        ])
        plugin = make_plugin(collection=self.collection_config(trigger_backfill_now=True))
        plugin._set_context(ctx)
        plugin._store = store
        plugin._last_backfill_trigger_at = 0.0
        run(plugin._maybe_manual_backfill_trigger())
        assert store.message_count("88880001") == 1
        # 30 秒防抖：连续再次触发不重复
        ctx._history_pages = [[self.napcat_message("m2", "m2", "20002", "B", "第二条", ts=2)]]
        run(plugin._maybe_manual_backfill_trigger())
        assert store.message_count("88880001") == 1

    def test_cmd_private_chat_all_groups(self, tmp_path: Path) -> None:
        """私聊（无群上下文）执行 /风格生成 与 /风格拉取历史：后台执行并作用于全部绑定群。"""
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", group_name="群A"))
        run(store.bind_group("88880002", group_name="群B"))
        plugin = make_plugin(owner={"qq": ["123456789"]}, collection=self.collection_config())
        plugin._store = store
        private_message = {"platform": "qq", "message_info": {"group_info": {}, "user_info": {"user_id": "123456789"}}}
        triggered: List[str] = []

        async def fake_analysis(group_id: str, reason: str = "", notify: bool = False) -> str:
            triggered.append(group_id)
            return ""

        plugin._run_analysis = fake_analysis  # type: ignore[method-assign]
        ok, text, _ = run(plugin.cmd_generate(**{
            "user_id": "123456789", "message": private_message, "matched_groups": {},
        }))
        assert ok and "已开始对 2 个绑定群" in text
        assert sorted(triggered) == ["88880001", "88880002"]

        plugin2 = make_plugin(owner={"qq": ["123456789"]}, collection=self.collection_config())
        plugin2._store = StyleStore(tmp_path / "empty")
        ok, text, _ = run(plugin2.cmd_generate(**{
            "user_id": "123456789", "message": private_message, "matched_groups": {},
        }))
        assert "尚未绑定任何群" in text

        # /风格拉取历史：私聊回填全部绑定群（后台执行）
        backfilled: List[str] = []
        ctx = HistoryFakeCtx(tmp_path)
        plugin3 = make_plugin(owner={"qq": ["123456789"]}, collection=self.collection_config())
        plugin3._set_context(ctx)
        plugin3._store = store

        async def fake_backfill(group_id: str, count: int = 0, notify: bool = False) -> str:
            backfilled.append(group_id)
            return ""

        plugin3._run_backfill = fake_backfill  # type: ignore[method-assign]
        ok, text, _ = run(plugin3.cmd_backfill(**{
            "user_id": "123456789", "message": private_message, "matched_groups": {},
        }))
        assert ok and "已开始对 2 个绑定群" in text
        assert sorted(backfilled) == ["88880001", "88880002"]
