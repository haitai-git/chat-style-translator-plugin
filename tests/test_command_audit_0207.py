from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from maibot_sdk.components import collect_components

from test_plugin import run, sample_message
from test_regressions_0201 import GROUP, SESSION, make_env


OWNER = "55331122"
PUBLIC = {"style_help", "style_status", "style_list", "style_detail"}
ROUTES = [
    "/风格帮助", "/风格绑定群", "/风格解绑群", "/风格状态", "/风格生成", "/风格拉取历史",
    "/风格列表", "/风格详情", "/风格场景", "/风格启用", "/风格停用", "/风格全局启用", "/风格全局停用",
    "/风格合并", "/风格删除语句", "/风格删除场景", "/风格重命名", "/风格删除模型",
    "/黑名单添加", "/黑名单添加模型", "/黑名单删除", "/黑名单列表", "/黑名单清空",
]


def command(plugin, text: str):
    matches = []
    for item in collect_components(plugin):
        if item["type"] != "COMMAND":
            continue
        info = item["metadata"]
        match = re.match(info["command_pattern"], text)
        if match is not None:
            matches.append((item["name"], getattr(plugin, info["handler_name"]), match.groupdict()))
    assert len(matches) == 1, text
    return matches[0]


async def execute(plugin, text: str, **kwargs):
    _, handler, groups = command(plugin, text)
    return (await handler(**{"user_id": OWNER, "platform": "qq", "matched_groups": groups, **kwargs}))[1]


@pytest.mark.parametrize("text", ROUTES)
def test_all_registered_commands_accept_qq_admins(tmp_path: Path, text: str) -> None:
    plugin = make_env(tmp_path, owner={"qq": [OWNER, "44221133"]})
    private = sample_message(sender=OWNER)
    private["message_info"].pop("group_info")
    name, handler, groups = command(plugin, text)
    for allowed in (OWNER, "44221133"):
        reply = run(handler(platform="qq", user_id=allowed, message=private, matched_groups=groups))[1]
        assert "权限不足" not in reply
    for outsider in ({"platform": "qq", "user_id": "11998877"}, {"platform": "other", "user_id": OWNER}):
        reply = run(handler(**outsider, message=private, matched_groups=groups))[1]
        assert ("权限不足" in reply) == (name not in PUBLIC)


def test_parameter_help_quotes_and_current_group_binding(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, owner={"qq": [OWNER]})
    reply = plugin.ctx.send.text = AsyncMock()
    request = {"group_id": GROUP, "message": sample_message(group_id=GROUP), "stream_id": SESSION}

    async def scenario() -> None:
        assert "用法" in await execute(plugin, "/风格详情", **request)
        assert "用法" in await execute(plugin, '/风格重命名 "缺引号', **request)
        assert "用法" in await execute(plugin, "/风格解绑群", **request)
        assert "已绑定" in await execute(plugin, "／风格绑定群 ", **request)
        assert plugin._session_map[SESSION] == GROUP
        assert "已绑定" in await execute(plugin, "/风格绑定群 77788", **request)
        assert plugin._session_map[SESSION] == GROUP
        private = {"message": {"platform": "qq", "message_info": {"user_info": {"user_id": OWNER}}}}
        assert "指定群号" in await execute(plugin, "/风格绑定群", **private)
        assert "已绑定" in await execute(plugin, "/风格绑定群 73331", **private, stream_id="private")
        assert "private" not in plugin._session_map

        await plugin._store.save_model({"model_id": "daily", "name": "群聊 模型"})
        assert "更新 测试" in await execute(plugin, '/风格重命名 “群聊 模型” "更新 测试"', **request)
        assert plugin._store.get_model("daily")["name"] == "更新 测试"
        assert reply.await_count > 0

    run(scenario())


def test_admin_private_receipt_and_napcat_error(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, owner={"qq": [OWNER]})
    send = plugin.ctx.send.text = AsyncMock()
    private = plugin.ctx.api.call = AsyncMock(side_effect=[
        {"status": "ok", "retcode": 0, "data": {"message_id": 9}},
        {"status": "failed", "retcode": 100, "data": None},
    ])
    request = {"group_id": GROUP, "message": sample_message(group_id=GROUP), "stream_id": SESSION}

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "review",
            "scenes": [{"name": "待审", "statements": [{"text": "私聊原话"}]}],
        })
        text = await execute(plugin, "/风格场景 review 1", **request)
        assert "查收机器人私聊" in text
        params = private.await_args.kwargs["params"]
        assert params["user_id"] == int(OWNER) and params["message_type"] == "private"
        assert "私聊原话" in params["message"][0]["data"]["text"]
        assert "私聊原话" not in send.await_args.args[0]
        failure = await execute(plugin, "/风格场景 review 1", **request)
        assert "私聊发送失败" in failure and "私聊原话" not in failure

    run(scenario())


def test_ambiguous_destructive_commands_require_exact_target(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, owner={"qq": [OWNER]})

    async def scenario() -> None:
        first = {"name": "聊天 今日", "model_id": "today", "scenes": [
            {"name": "普通", "statements": [{"id": "st0", "text": "甲"}]},
            {"name": "普通", "statements": [{"id": "st0", "text": "乙"}]},
        ]}
        await plugin._store.save_model(first)
        await plugin._store.save_model({"name": "聊天 昨日", "model_id": "yesterday"})
        assert "多个模型" in await execute(plugin, "/风格删除模型 聊天")
        assert plugin._store.get_model("today") and plugin._store.get_model("yesterday")
        assert "多个场景" in await execute(plugin, "/风格删除场景 today 普通")
        assert "加场景序号" in await execute(plugin, "/风格删除语句 today st0")
        assert "已删除语句" in await execute(plugin, "/风格删除语句 today st0 2")
        scenes = plugin._store.get_model("today")["scenes"]
        assert scenes[0]["statements"][0]["text"] == "甲"
        assert scenes[1]["statements"] == []
        assert "已删除模型" in await execute(plugin, "/风格删除模型 yesterday.json")
        assert plugin._store.get_model("today") and plugin._store.get_model("yesterday") is None

    run(scenario())


def test_history_count_shorthand_uses_the_current_group(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, owner={"qq": [OWNER]})
    backfill = plugin._run_backfill = AsyncMock()
    async def scenario() -> None:
        await plugin._store.bind_group(GROUP)
        text = await execute(plugin, "/风格拉取历史 300条", group_id=GROUP, message=sample_message(group_id=GROUP))
        assert "已开始回填" in text
        await asyncio.sleep(0)
        backfill.assert_awaited_once_with(GROUP, count=300, notify=True)
    run(scenario())


def test_scene_pages_are_read_only_and_link_to_the_next_page(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, owner={"qq": [OWNER]})
    request = {"message": {"platform": "qq", "message_info": {"user_info": {"user_id": OWNER}}}}

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "daily", "scenes": [{
                "name": "场景二",
                "statements": [
                    {"id": f"st{i}", "text": f"第 {i} 条原话"} for i in range(1, 13)
                ],
            }],
        })
        summary = await execute(plugin, "/风格场景 daily", **request)
        assert "场景二" in summary and "第 1 条原话" not in summary
        first = await execute(plugin, "/风格场景 daily 1", **request)
        assert "第 10 条原话" in first and "第 11 条原话" not in first
        assert '下一页 /风格场景 "daily" 1 2' in first
        second = await execute(plugin, '/风格场景 "daily" 1 2', **request)
        assert "第 11 条原话" in second and "第 2/2 页" in second
        assert "页码应在" in await execute(plugin, "/风格场景 daily 1 0", **request)
        assert len(plugin._store.get_model("daily")["scenes"][0]["statements"]) == 12

    run(scenario())
