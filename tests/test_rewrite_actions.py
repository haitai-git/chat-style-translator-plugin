from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chat_style_translator_plugin.plugin import ChatStyleTranslatorPlugin
from test_plugin import run
from test_regressions_0201 import SESSION, make_env


@pytest.mark.parametrize(("draft", "rewrite"), [
    ("咱们下午三点在东门碰头，到了给我发条消息", "下午三点东门见，到了说一声"),
    ("咱们下午三点在东门碰头，到了给我发条消息", "下午三点西门见，到了给我发条消息"),
    ("到了给我发条消息", "到了给他发条消息"),
    ("到了，给我发条消息", "给我发条消息"),
    ("等看完结果，给我回个电话", "给我回个电话"),
    ("别发到群里，私聊我", "发到群里，私聊我"),
    ("到家给我打个电话", "到家给我发条消息"),
    ("记得把文件发给管理员", "记得把文件发给群主"),
    ("能处理的话，别删除原文件", "能处理的话，删除原文件"),
])
def test_action_guard_rejects_changed_details(draft: str, rewrite: str) -> None:
    assert not ChatStyleTranslatorPlugin._rewrite_passes_safety(draft, rewrite)


@pytest.mark.parametrize(("strength", "degree"), [(30, "轻度"), (65, "较强"), (100, "强烈")])
def test_action_guard_applies_at_every_strength_and_keeps_sampling_temperature(
    tmp_path: Path, strength: int, degree: str
) -> None:
    plugin = make_env(tmp_path, inject={"rewrite_strength": strength})
    draft = "咱们下午三点在东门碰头，到了给我发条消息"
    safe = "下午三点东门见，到了给我发条消息。"
    plugin.ctx.llm.generate = AsyncMock(side_effect=[
        {"success": True, "response": "下午三点东门见，到了说一声"},
        {"success": True, "response": safe},
    ])

    async def scenario() -> None:
        await plugin._store.save_model({"model_id": "sample", "overall_style": "口语短句"})
        plugin._global_models = ["sample"]
        fallback = await plugin.rewrite_style_response(response=draft, session_id=SESSION)
        assert fallback == {"action": "continue"}
        result = await plugin.rewrite_style_response(response=draft, session_id=SESSION)
        assert result["modified_kwargs"]["response"] == safe
        for call in plugin.ctx.llm.generate.await_args_list:
            assert f"本次微调强度：{degree}" in call.args[0][0]["content"]
            assert '【必须逐字保留的动作及条件】\n["到了给我发条消息"]' in call.args[0][1]["content"]
            assert call.kwargs["temperature"] == 0.2

    run(scenario())


def test_semantic_blacklist_keeps_literal_result_when_action_is_changed(tmp_path: Path) -> None:
    plugin = make_env(
        tmp_path,
        pre_rewrite_blacklist={"enabled": True, "items": ["禁词"], "semantic_match": True},
        inject={"rewrite_strength": 0},
    )
    plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": "到了说一声"})

    async def scenario() -> None:
        result = await plugin.rewrite_style_response(response="禁词到了给我发条消息", session_id=SESSION)
        assert result["modified_kwargs"]["response"] == "到了给我发条消息"
        assert "到了给我发条消息" in plugin.ctx.llm.generate.await_args.args[0][1]["content"]

    run(scenario())
