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

    @pytest.mark.parametrize(
        "pattern, good, bad",
        [
            (r"^/风格帮助$", ["/风格帮助"], ["/风格帮助 "]),
            (r"^/风格绑定群\s+(?P<group_id>\d+)$", ["/风格绑定群 123456"], ["/风格绑定群", "/风格绑定群 abc"]),
            (r"^/风格解绑群\s+(?P<group_id>\d+)$", ["/风格解绑群 123"], ["/风格解绑群"]),
            (r"^/风格状态$", ["/风格状态"], ["/风格状态x"]),
            (r"^/风格生成(?:\s+(?P<group_id>\d+))?$", ["/风格生成", "/风格生成 123"], ["/风格生成 12a"]),
            (r"^/风格列表$", ["/风格列表"], ["/风格列表 1"]),
            (r"^/风格详情\s+(?P<model_name>\S+)$", ["/风格详情 模型A"], ["/风格详情"]),
            (r"^/风格场景\s+(?P<model_name>\S+)(?:\s+(?P<scene_index>\d+))?$",
             ["/风格场景 模型A", "/风格场景 模型A 2"], ["/风格场景"]),
            (r"^/风格启用\s+(?P<model_name>\S+)(?:\s+(?P<group_id>\d+))?$",
             ["/风格启用 模型A", "/风格启用 模型A 123"], ["/风格启用"]),
            (r"^/风格停用\s+(?P<model_name>\S+)(?:\s+(?P<group_id>\d+))?$", ["/风格停用 模型A"], ["/风格停用"]),
            (r"^/风格全局启用\s+(?P<model_name>\S+)$", ["/风格全局启用 模型A"], ["/风格全局启用"]),
            (r"^/风格全局停用\s+(?P<model_name>\S+)$", ["/风格全局停用 模型A"], ["/风格全局停用"]),
            (r"^/风格合并\s+(?P<model_a>\S+)\s+(?P<model_b>\S+)(?:\s+(?P<new_name>\S+))?$",
             ["/风格合并 A B", "/风格合并 A B 新名"], ["/风格合并 A"]),
            (r"^/风格删除语句\s+(?P<model_name>\S+)\s+(?P<statement_id>\S+)$",
             ["/风格删除语句 模型A st1"], ["/风格删除语句 模型A"]),
            (r"^/风格删除场景\s+(?P<model_name>\S+)\s+(?P<scene_ref>\S+)$",
             ["/风格删除场景 模型A 1"], ["/风格删除场景 模型A"]),
            (r"^/风格重命名\s+(?P<model_name>\S+)\s+(?P<new_name>\S+)$",
             ["/风格重命名 模型A 新名"], ["/风格重命名 模型A"]),
            (r"^/风格删除模型\s+(?P<model_name>\S+)$", ["/风格删除模型 模型A"], ["/风格删除模型"]),
            (r"^/黑名单添加\s+(?P<text>.+)$", ["/黑名单添加 骂人的话"], ["/黑名单添加"]),
            (r"^/黑名单添加模型\s+(?P<model_name>\S+)\s+(?P<text>.+)$",
             ["/黑名单添加模型 模型A 骂人的话"], ["/黑名单添加模型 模型A"]),
            (r"^/黑名单删除\s+(?P<item_id>\S+)$", ["/黑名单删除 bl_1"], ["/黑名单删除"]),
            (r"^/黑名单列表$", ["/黑名单列表"], ["/黑名单列表 1"]),
            (r"^/黑名单清空$", ["/黑名单清空"], ["/黑名单清空 1"]),
            (r"^/风格拉取历史(?:\s+(?P<group_id>\d+))?(?:\s+(?P<count>\d+))?$",
             ["/风格拉取历史", "/风格拉取历史 123", "/风格拉取历史 123 300"], ["/风格拉取历史 abc"]),
        ],
    )
    def test_command_patterns(self, pattern: str, good: List[str], bad: List[str]) -> None:
        regex = re.compile(pattern)
        for text in good:
            assert regex.match(text), f"{text!r} 应匹配 {pattern}"
        for text in bad:
            assert regex.match(text) is None, f"{text!r} 不应匹配 {pattern}"


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
        return {"user_id": "123456789", "message": sample_message(), **extra}

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
        ok, text, _ = run(plugin.cmd_list(**self.owner_kwargs()))
        assert ok and "风格模型A" in text and model_id in text
        ok, text, _ = run(plugin.cmd_detail(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert ok and "整体风格" in text and "小明" in text
        ok, text, _ = run(plugin.cmd_scene(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert ok and "原话" in text
        ok, text, _ = run(plugin.cmd_scene(**self.owner_kwargs(
            matched_groups={"model_name": "风格模型A", "scene_index": "1"})))
        assert ok and "原话" in text
        ok, text, _ = run(plugin.cmd_rename(**self.owner_kwargs(
            matched_groups={"model_name": "风格模型A", "new_name": "新名字"})))
        assert ok and "新名字" in text
        assert store.find_model("新名字") is not None
        # 未找到模型
        ok, text, _ = run(plugin.cmd_detail(**self.owner_kwargs(matched_groups={"model_name": "不存在"})))
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
        # 「模型目录自动生效」默认开启：模型文件仍会自动全局生效，停用后仍在全局列表
        ok, text, _ = run(plugin.cmd_global_disable(**self.owner_kwargs(matched_groups={"model_name": "风格模型A"})))
        assert ok and "已关闭" in text
        assert model_id in plugin._global_models
        assert store.get_model(model_id).get("global_apply") is False
        # 关闭「模型目录自动生效」后，停用全局才会真正从全局列表移除
        plugin._auto_enable_models = False
        run(plugin._refresh_global_models())
        assert not plugin._global_models

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
        ok, text, _ = run(plugin.cmd_blacklist_list(**self.owner_kwargs()))
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
