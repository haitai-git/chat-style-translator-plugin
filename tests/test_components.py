"""聊天风格转译插件全量测试（SDK 组件注册与命令行为）。"""

from __future__ import annotations

import asyncio
import re
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


def sample_message(group_id: str = "88880001") -> Dict[str, Any]:
    return {
        "message_id": "msg-1",
        "session_id": "sess-1",
        "processed_plain_text": "测试",
        "is_command": False,
        "message_info": {
            "user_info": {"user_id": "20002", "user_nickname": "u", "user_cardname": "c"},
            "group_info": {"group_id": group_id, "group_name": "测试群"},
            "additional_config": {},
        },
        "raw_message": [],
    }


class FakeAnalyzer:
    async def analyze_window(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"overall_style": "风格", "members": [], "scenes": [{"name": "S", "statements": []}]}

    async def merge_partials(
        self,
        partials: List[Dict[str, Any]],
        name: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {"model_id": "m_new", "name": name, "overall_style": "s", "members": [], "scenes": []}

    async def filter_statements(
        self,
        statements: List[Dict[str, Any]],
        blacklist_items: List[Dict[str, Any]],
        threshold: float,
        model_id: str = "",
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return statements, []


class TestComponents:
    def test_component_types(self) -> None:
        from maibot_sdk.components import collect_components

        plugin = ChatStyleTranslatorPlugin()
        components = collect_components(plugin)
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for component in components:
            by_type.setdefault(component["type"], []).append(component)
        hooks = by_type.get("HOOK_HANDLER", [])
        commands = by_type.get("COMMAND", [])
        assert len(hooks) == 3, hooks
        assert len(commands) == 23, [c["name"] for c in commands]
        hook_names = {(h["metadata"]["hook"], h["metadata"]["handler_name"]) for h in hooks}
        assert ("chat.receive.after_process", "handle_message") in hook_names
        assert ("maisaka.replyer.after_response", "rewrite_style_response") in hook_names
        assert ("maisaka.replyer.after_response", "garbage_guard") in hook_names or \
            ("maisaka.replyer.after_response", "guard_garbage_response") in hook_names
        collector = next(h for h in hooks if h["metadata"]["handler_name"] == "handle_message")
        assert collector["metadata"]["mode"] == "blocking"
        assert collector["metadata"]["error_policy"] == "skip"
        assert collector["metadata"]["timeout_ms"] > 0
        rewrite = next(h for h in hooks if h["metadata"]["handler_name"] == "rewrite_style_response")
        assert rewrite["metadata"]["mode"] == "blocking"
        assert rewrite["metadata"]["error_policy"] == "skip"
        guard = next(h for h in hooks if h["metadata"]["handler_name"] == "guard_garbage_response")
        assert guard["metadata"]["mode"] == "blocking"
        assert guard["metadata"]["order"] == "late"
        assert guard["metadata"]["error_policy"] == "skip"

    def test_all_command_patterns_compile(self) -> None:
        from maibot_sdk.components import collect_components

        plugin = ChatStyleTranslatorPlugin()
        for component in collect_components(plugin):
            if component["type"] == "COMMAND":
                pattern = component["metadata"].get("command_pattern", "")
                assert pattern, component
                re.compile(pattern)

    @pytest.mark.parametrize("command, expected", [
        ("/风格", "style_help"),
        (" /风格帮助 ", "style_help"),
        ("／风格绑定群", "style_bind_group"),
        ("/风格详情", "style_detail"),
        ('/风格重命名 "旧 模型" “新 模型”', "style_rename"),
        ('/风格融合 A B “新 模型”', "style_merge"),
        ("/黑名单添加模型 A 骂人的话", "blacklist_add_model"),
        ("/风格拉取历史 300条", "style_backfill"),
        ("/风格状态x", None),
        ("风格列表", None),
        ("/风格绑定群没有空格", None),
    ])
    def test_command_patterns(self, command: str, expected: Optional[str]) -> None:
        from maibot_sdk.components import collect_components

        matches = [
            item["name"] for item in collect_components(ChatStyleTranslatorPlugin())
            if item["type"] == "COMMAND" and re.match(item["metadata"]["command_pattern"], command)
        ]
        assert matches == ([expected] if expected else [])


class TestCommands:
    def make_env(self, tmp_path: Path) -> tuple[ChatStyleTranslatorPlugin, StyleStore]:
        store = StyleStore(tmp_path)
        run(store.bind_group("88880001", source="config"))
        plugin = make_plugin(owner={"qq": ["123456789"]})
        plugin._store = store
        return plugin, store

    def save_model(self, store: StyleStore, name: str, with_scene: bool = True) -> str:
        model = {
            "name": name,
            "overall_style": "风格",
            "members": [{"user_id": "1", "name": "小明", "style": "s", "personality": "p", "speaking_traits": []}],
            "scenes": ([{
                "name": "闲聊", "description": "d", "style": "s", "note": "n",
                "statements": [{"id": "st0_0", "text": "原话", "speaker": "1", "speaker_name": "小明",
                                "target": "大家", "meaning": "m", "ai_note": "n", "tags": []}],
            }] if with_scene else []),
        }
        run(store.save_model(model))
        return str(model["model_id"])

    def owner_kwargs(self, **extra: Any) -> Dict[str, Any]:
        return {"platform": "qq", "user_id": "123456789", "message": sample_message(), **extra}

    def test_help(self) -> None:
        plugin = make_plugin()
        ok, text, flag = run(plugin.cmd_help(**{"user_id": "任何人"}))
        assert ok and "使用说明" in text and flag == 2

    def test_bind_unbind_status(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        ok, text, _ = run(plugin.cmd_bind_group(**self.owner_kwargs(matched_groups={"group_id": "999"})))
        assert ok and "已绑定群 999" in text
        assert store.is_bound("999")
        ok, text, _ = run(plugin.cmd_unbind_group(**self.owner_kwargs(matched_groups={"group_id": "999"})))
        assert ok and "已解绑" in text
        ok, text, _ = run(plugin.cmd_status(**self.owner_kwargs()))
        assert ok and "88880001" in text and "已收集" in text
        ok, text, _ = run(plugin.cmd_bind_group(**{"user_id": "路人", "matched_groups": {"group_id": "1"}}))
        assert ok and "权限不足" in text

    def test_model_commands(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        model_id = self.save_model(store, "风格模型A")
        ok, text, _ = run(plugin.cmd_list(**self.owner_kwargs(is_local_operator=True)))
        assert ok and "风格模型A" in text and model_id in text
        ok, text, _ = run(plugin.cmd_detail(**self.owner_kwargs(is_local_operator=True, matched_groups={"model_name": "风格模型A"})))
        assert ok and "整体风格" in text and "小明" in text
        ok, text, _ = run(plugin.cmd_scene(**self.owner_kwargs(is_local_operator=True, matched_groups={"model_name": "风格模型A"})))
        assert ok and "闲聊" in text and "：“原话”" not in text
        ok, text, _ = run(plugin.cmd_scene(**self.owner_kwargs(is_local_operator=True,
            matched_groups={"model_name": "风格模型A", "scene_index": "1"})))
        assert ok and "原话" in text
        ok, text, _ = run(plugin.cmd_rename(**self.owner_kwargs(
            matched_groups={"model_name": "风格模型A", "new_name": "新名字"})))
        assert ok and "新名字" in text
        assert store.find_model("新名字") is not None
        # 未找到模型
        ok, text, _ = run(plugin.cmd_detail(**self.owner_kwargs(is_local_operator=True, matched_groups={"model_name": "不存在"})))
        assert ok and "未找到" in text

    def test_enable_disable_flow(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        model_id = self.save_model(store, "风格模型A")
        ok, text, _ = run(plugin.cmd_enable(**self.owner_kwargs(
            matched_groups={"model_name": "风格模型A", "group_id": "999"})))
        assert ok and "未绑定" in text
        ok, text, _ = run(plugin.cmd_enable(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert ok and "已审核通过并挂载" in text
        assert store.get_model(model_id)["status"] == "approved"
        assert store.get_group("88880001")["active_models"] == [model_id]
        ok, text, _ = run(plugin.cmd_disable(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert ok and "已从群" in text
        assert store.get_group("88880001")["active_models"] == []

    def test_global_enable_disable(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        model_id = self.save_model(store, "风格模型A")
        ok, text, _ = run(plugin.cmd_global_enable(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert ok and "全局通用" in text
        assert plugin._global_models
        plugin._config_global_models = [model_id]
        run(plugin._refresh_global_models())
        ok, text, _ = run(plugin.cmd_global_disable(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert ok and "已关闭" in text
        run(plugin._refresh_global_models())
        assert plugin._resolve_active_models("其它群") == []
        assert store.get_model(model_id).get("global_apply") is False
        assert store.get_model(model_id).get("global_disabled") is True
        run(plugin.cmd_global_enable(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert plugin._resolve_active_models("其它群") == [model_id]

    def test_delete_and_remove(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        self.save_model(store, "风格模型A")
        ok, text, _ = run(plugin.cmd_remove_statement(**self.owner_kwargs(
            matched_groups={"model_name": "风格模型A", "statement_id": "st0_0"})))
        assert ok and "已删除语句" in text
        ok, text, _ = run(plugin.cmd_remove_scene(**self.owner_kwargs(
            matched_groups={"model_name": "风格模型A", "scene_ref": "1"})))
        assert ok and "已删除场景" in text
        assert store.find_model("风格模型A")["scenes"] == []
        ok, text, _ = run(plugin.cmd_delete_model(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert ok and "已删除模型" in text
        assert store.find_model("风格模型A") is None

    def test_blacklist_commands(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        ok, text, _ = run(plugin.cmd_blacklist_add(**self.owner_kwargs(matched_groups={"text": "骂人的话"})))
        assert ok and "已添加通用黑名单" in text
        item = store.list_blacklist()[0]
        ok, text, _ = run(plugin.cmd_blacklist_list(**self.owner_kwargs(is_local_operator=True)))
        assert ok and "骂人的话" in text and item["id"] in text
        ok, text, _ = run(plugin.cmd_blacklist_remove(**self.owner_kwargs(matched_groups={"item_id": item["id"]})))
        assert ok and "已删除" in text
        assert store.list_blacklist() == []
        self.save_model(store, "风格模型A")
        ok, text, _ = run(plugin.cmd_blacklist_add_model(**self.owner_kwargs(
            matched_groups={"model_name": "风格模型A", "text": "专属词"})))
        assert ok and "风格模型A" in text
        assert len(store.list_blacklist()) == 1
        run(plugin.cmd_blacklist_clear(**self.owner_kwargs()))
        assert store.list_blacklist() == []

    def test_generate_command_in_group(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        run(store.append_message("88880001", {"id": "1", "text": "消息", "sender_name": "小明"}))
        plugin._analyzer = FakeAnalyzer()
        ok, text, _ = run(plugin.cmd_generate(**self.owner_kwargs(matched_groups={})))
        assert ok and "已开始分析群 88880001" in text

    def test_merge_command(self, tmp_path: Path) -> None:
        plugin, store = self.make_env(tmp_path)
        self.save_model(store, "模型A", with_scene=False)
        self.save_model(store, "模型B", with_scene=False)
        plugin._analyzer = FakeAnalyzer()
        ok, text, _ = run(plugin.cmd_merge(**self.owner_kwargs(
            matched_groups={"model_a": "模型A", "model_b": "模型B"})))
        assert ok and "融合完成" in text
        assert store.find_model("模型A×模型B") is not None
        ok, text, _ = run(plugin.cmd_merge(**self.owner_kwargs(
            matched_groups={"model_a": "模型A", "model_b": "不存在"})))
        assert ok and "未找到" in text
