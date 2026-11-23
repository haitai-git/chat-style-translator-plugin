from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from chat_style_translator_plugin.modules.analyzer import StyleAnalyzer
from chat_style_translator_plugin.modules.storage import StyleStore
from chat_style_translator_plugin.plugin import ChatStyleTranslatorPlugin

from test_plugin import FakeAnalyzer, FakeCtx, make_plugin, run, sample_message


GROUP = "90001"
SESSION = "sess-1"


def make_env(tmp_path: Path, **settings: Any):
    plugin = make_plugin(**settings)
    plugin._set_context(FakeCtx(tmp_path))
    plugin._store = StyleStore(tmp_path)
    plugin._session_map[SESSION] = GROUP
    return plugin


def test_analysis_keeps_failed_batch_and_new_messages(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, collection={"window_size": 5, "auto_analyze_count": 0})

    async def scenario() -> None:
        started = asyncio.Event()
        resume = asyncio.Event()

        class Analyzer(FakeAnalyzer):
            async def analyze_window(self, messages):
                if messages[0]["id"] == "5":
                    return None
                started.set()
                await resume.wait()
                return await super().analyze_window(messages)

        plugin._analyzer = Analyzer()
        await plugin._store.bind_group(GROUP)
        for index in range(6):
            await plugin._store.append_message(GROUP, {"id": str(index), "text": f"对话样例{index}"})
        task = asyncio.create_task(plugin.run_analysis(GROUP))
        await asyncio.wait_for(started.wait(), 1)
        await plugin._store.append_message(GROUP, {"text": "新的实时消息"})
        resume.set()
        result = await task
        assert "1 条消息所在批次失败" in result
        assert [record["text"] for record in plugin._store.messages_for(GROUP)] == ["对话样例5", "新的实时消息"]

    run(scenario())


def test_history_caps_requested_count_and_keeps_latest_by_time(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, collection={"auto_analyze_count": 0, "max_stored_messages": 2})
    plugin.ctx.api.call = AsyncMock(return_value={
        "data": {"messages": [
            {"message_id": timestamp, "time": timestamp, "message": [{"type": "text", "data": {"text": "历史"}}]}
            for timestamp in (100, 350, 250)
        ]}
    })

    async def scenario() -> None:
        await plugin._store.bind_group(GROUP)
        for timestamp in (300, 400):
            await plugin._store.append_message(
                GROUP, {"id": str(timestamp), "ts": timestamp, "text": "已收集消息"}
            )
        result = await plugin.run_backfill(GROUP, count=1)
        assert "历史消息 1 条（新增 1 条，现有 2 条）" in result
        assert [record["ts"] for record in plugin._store.messages_for(GROUP)] == [350, 400]

    run(scenario())


def test_zero_strength_still_filters_without_requesting_style(tmp_path: Path) -> None:
    plugin = make_env(
        tmp_path, inject={"rewrite_strength": 0}, pre_rewrite_blacklist={"enabled": True, "items": ["禁词"]}
    )
    plugin.ctx.llm.generate = AsyncMock()

    async def scenario() -> None:
        await plugin._store.save_model({"model_id": "style", "overall_style": "口语短句"})
        plugin._global_models = ["style"]
        result = await plugin.rewrite_style_response(response="禁词先放一边", session_id=SESSION)
        assert result["modified_kwargs"]["response"] == "先放一边"
        plugin.ctx.llm.generate.assert_not_called()

    run(scenario())


def test_blacklist_survives_short_response_and_overall_timeout(tmp_path: Path) -> None:
    plugin = make_env(
        tmp_path,
        inject={"hook_timeout_seconds": 3.0},
        pre_rewrite_blacklist={"enabled": True, "items": ["禁词"]},
    )

    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(10)

    plugin.ctx.llm.generate = AsyncMock(side_effect=slow_generate)

    async def scenario() -> None:
        await plugin._store.save_model({"model_id": "style", "overall_style": "口语短句"})
        plugin._global_models = ["style"]
        result = await plugin.rewrite_style_response(response="禁词暂且休息", session_id=SESSION)
        assert result["modified_kwargs"]["response"] == "暂且休息"
        plugin.ctx.llm.generate.assert_not_called()
        result = await plugin.rewrite_style_response(response="禁词先放到一边", session_id=SESSION)
        assert result["modified_kwargs"]["response"] == "先放到一边"
        assert plugin.ctx.llm.generate.await_count == 1

    run(scenario())


def test_model_scan_ignores_json_array(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, plugin={"config_version": "0.20.18", "auto_enable_models": True})
    plugin._store.model_dir.mkdir()
    (plugin._store.model_dir / "wrong.json").write_text("[]", encoding="utf-8")

    async def scenario() -> None:
        await plugin._store.save_model({"model_id": "good", "name": "正常模型", "overall_style": "短句", "status": "approved"})
        await plugin.on_load()
        try:
            assert plugin._resolve_active_models(GROUP) == ["good"]
            assert plugin._store.get_model("wrong") is None
        finally:
            await plugin.on_unload()

    run(scenario())


def test_config_model_jobs_acknowledge_early_and_keep_each_save(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)
    plugin._analyzer = StyleAnalyzer(plugin.ctx)

    async def scenario() -> None:
        for model_id in ("a", "b"):
            await plugin._store.save_model({"model_id": model_id, "name": model_id})
        first_started = asyncio.Event()
        finish = asyncio.Event()
        names = []

        async def slow_merge(model_a, model_b, new_name, **kwargs):
            names.append(new_name)
            first_started.set()
            await finish.wait()
            return "融合完成"

        plugin.merge_models = slow_merge
        try:
            for name in ("第一次", "第二次"):
                plugin.set_plugin_config(
                    {
                        "plugin": {"config_version": "0.18.0"},
                        "merge": {"trigger": True, "model_a": "a", "model_b": "b", "new_name": name},
                    }
                )
                await asyncio.wait_for(plugin.on_config_update("self", {}, ""), 1)
            await asyncio.wait_for(first_started.wait(), 1)
            assert names == ["第一次"]
            finish.set()
            await asyncio.wait_for(asyncio.gather(*plugin._config_tasks), 1)
            assert names == ["第一次", "第二次"]
        finally:
            await plugin.on_unload()

    run(scenario())


def test_distilled_scene_drives_natural_rewrite_without_changing_time(tmp_path: Path) -> None:
    plugin = make_env(
        tmp_path,
        collection={"auto_analyze_count": 0, "model": "test_analysis"},
        inject={"reference_mode": "按场景", "rewrite_model": "test_fast", "max_chars": 195},
    )
    plugin._analyzer = StyleAnalyzer(plugin.ctx)
    plugin._sync_config()
    scene_model = {
        "overall_style": "偏口语",
        "scenes": [
            {"name": "技术排障", "description": "软件启动故障排查" * 30},
            {
                "name": "约线下聚餐",
                "description": "确认聚餐的见面时间",
                "statements": [{
                    "text": "那就下午三点见吧",
                    "speaker": "902",
                    "speaker_name": "小林",
                    "target": "小陈",
                    "meaning": "确认聚餐时间",
                }],
            },
        ],
    }
    plugin.ctx.llm.generate = AsyncMock(side_effect=[
        {"success": True, "response": json.dumps(scene_model, ensure_ascii=False)},
        {"success": True, "response": "行，下午三点见"},
        {"success": True, "response": "行，下午四点见"},
    ])

    async def scenario() -> None:
        await plugin._store.bind_group(GROUP)
        sample = sample_message(
            sender="902", group_id=GROUP, text="那就下午三点见吧",
            at={"target_user_id": "903", "target_user_nickname": "小陈"},
        )
        await plugin.handle_message(sample)
        assert (await plugin.run_analysis(GROUP)).startswith("分析完成")
        first_call = plugin.ctx.llm.generate.await_args
        assert "对「小陈」说" in first_call.args[0][1]["content"]
        assert first_call.kwargs["model"] == "test_analysis"
        file_id = plugin._store.list_model_ids()[0]
        await plugin.cmd_global_enable(is_local_operator=True, matched_groups={"model_name": file_id})

        incoming = dict(sample_message(sender="903", group_id=GROUP, text="聚餐几点见"), message_id="trigger")
        await plugin.handle_message(incoming)
        draft = "好的，我们约定下午三点见面"
        result = await plugin.rewrite_style_response(
            response=draft, session_id=SESSION, reply_message_id="trigger"
        )
        assert result["modified_kwargs"]["response"] == "行，下午三点见"
        call = plugin.ctx.llm.generate.await_args
        assert call.kwargs["model"] == "test_fast"
        system, prompt = [entry["content"] for entry in call.args[0]]
        assert "避免客服腔" in system and "本次必须补词" in system
        assert "约线下聚餐" in prompt and "对小陈说" not in prompt
        assert "技术排障" not in prompt

        result = await plugin.rewrite_style_response(
            response=draft, session_id=SESSION, reply_message_id="trigger"
        )
        assert result == {"action": "continue"}
        assert plugin.ctx.llm.generate.await_count == 3

    run(scenario())



def test_fact_guard_allows_colloquial_phrasing_and_rejects_changed_details() -> None:
    check = ChatStyleTranslatorPlugin._rewrite_passes_safety
    assert check("今晚一起吗", "今晚一块儿吗")
    assert not check("咱们下午3点半见", "咱们下午3点见")
    assert not check("地址是 https://example.com/a", "地址是 https://example.com/b")
    assert not check("预算是99元", "预算是199元")
