from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

from chat_style_translator_plugin.modules.analyzer import StyleAnalyzer
from test_plugin import FakeAnalyzer, run, sample_message
from test_regressions_0201 import GROUP, SESSION, make_env


def test_group_can_read_only_its_model_summary(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, owner={"qq": ["99999"]})
    plugin.ctx.send.text = AsyncMock()

    async def scenario() -> None:
        await plugin._store.bind_group(GROUP)
        await plugin._store.bind_group("88776")
        for key, name in (("own", "本群风格"), ("other", "其他群风格")):
            await plugin._store.save_model({
                "model_id": key, "name": name, "status": "approved",
                "members": [{"name": "档案昵称", "user_id": "775533"}],
                "scenes": [{"name": "群内例句", "statements": [{"text": "仅限审核的原话", "context": "真实前文"}]}],
            })
        await plugin.cmd_enable(
            is_local_operator=True, matched_groups={"model_name": "own", "group_id": GROUP}
        )
        await plugin.cmd_enable(
            is_local_operator=True, matched_groups={"model_name": "other", "group_id": "88776"}
        )
        group_message = sample_message(group_id=GROUP)
        request = {"message": group_message, "matched_groups": {"model_name": "own"}}
        assert plugin._resolve_active_models("88776") == ["other"]
        assert "其他群风格" not in (await plugin.cmd_list(**request))[1]
        detail = (await plugin.cmd_detail(**request, user_id="50123"))[1]
        assert "1 个场景" in detail and "群挂载" in detail
        assert "档案昵称" not in detail and "775533" not in detail
        assert "权限不足" in (await plugin.cmd_scene(**request, user_id="50123"))[1]
        admin_reply = (await plugin.cmd_detail(**request, user_id="99999"))[1]
        assert "查收机器人私聊" in admin_reply and "档案昵称" not in admin_reply
        assert plugin.ctx._api_calls[-1]["params"]["user_id"] == 99999
        assert "88776" not in (await plugin.cmd_status(**request))[1]
        wrong_model = dict(request, matched_groups={"model_name": "other"})
        assert "未找到" in (await plugin.cmd_detail(**wrong_model))[1]

        private = sample_message(sender="99999")
        private["message_info"].pop("group_info")
        _, text, _ = await plugin.cmd_scene(**dict(request, message=private, user_id="99999",
                                             matched_groups={"model_name": "own", "scene_index": "1"}))
        assert "仅限审核的原话" in text and "真实前文" in text
        assert "未找到" in (await plugin.cmd_detail(**dict(request, message=private)))[1]
        local_text = (await plugin.cmd_detail(**request, is_local_operator=True, stream_id=SESSION))[1]
        assert "档案昵称" in local_text
        plugin.ctx.send.text.assert_not_awaited()
        await plugin.cmd_detail(**request, is_local_operator=True, platform="bot_console", stream_id="console")
        plugin.ctx.send.text.assert_awaited_once_with(local_text, "console")

    run(scenario())


def test_group_only_approval_does_not_become_a_directory_default(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)

    async def scenario() -> None:
        await plugin._store.bind_group(GROUP)
        await plugin._store.save_model({"model_id": "only", "status": "pending"})
        match = {"model_name": "only", "group_id": GROUP}
        await plugin.cmd_enable(is_local_operator=True, matched_groups=match)
        assert plugin._resolve_active_models(GROUP) == ["only"]
        assert plugin._resolve_active_models("not_this_group") == []
        await plugin.cmd_disable(is_local_operator=True, matched_groups=match)
        await plugin._refresh_global_models()
        assert plugin._resolve_active_models(GROUP) == []

    run(scenario())


def test_reply_trace_uses_exact_message_and_combined_budget(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={"reference_mode": "按场景", "max_chars": 200})
    rewrite = plugin._rewrite_with_llm = AsyncMock(return_value="[[KEEP_ORIGINAL]]")
    earlier = "试试周末聚餐"
    draft = "好的，我们到时碰面吧"

    async def scenario() -> None:
        for model_id, character in (("food_a", "甲"), ("food_b", "乙")):
            await plugin._store.save_model({
                "model_id": model_id, "name": model_id,
                "scenes": [{
                    "name": "周末聚餐", "description": "聚餐约时间",
                    "statements": [{"text": "某句真实范例", "ai_note": "自然确认时间，别重复客套"}],
                    "style": "短句、语气自然" + character * 8,
                }],
            })
        plugin._config_global_models = ["food_a", "food_b"]
        await plugin._refresh_global_models()
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=earlier), message_id="earlier"))
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text="新话题"), message_id="latest"))
        await plugin.handle_message(dict(
            sample_message(group_id="89000", session_id="another-session", text="错误会话里的新来话"),
            message_id="earlier",
        ))
        assert await plugin.rewrite_style_response(
            response=draft, session_id=SESSION, reply_message_id="earlier"
        ) == {"action": "continue"}
        reference = rewrite.await_args.args[0]
        assert 0 < len(reference) <= 200
        assert "food_a" in reference and "food_b" in reference and "周末聚餐" in reference
        widgets = await plugin.get_config_widget_state()
        preview = widgets["reply_preview"].text
        assert "周末聚餐" in preview and "短句、语气自然" in preview and "未找到安全的补词方式" in preview
        assert f"微调前　{draft}" in preview and f"微调后　{draft}" in preview
        assert earlier not in preview and "某句真实范例" not in preview
        assert "food_a（全局配置）" in widgets["active_models"].text

        for reply_id in ("unavailable", ""):
            await plugin.rewrite_style_response(response=draft, session_id=SESSION, reply_message_id=reply_id)
            assert "周末聚餐" not in rewrite.await_args.args[0]

    run(scenario())


def test_same_nickname_uses_actual_sender_and_quoted_target(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)
    analyzer = StyleAnalyzer(plugin.ctx)
    first, second = "11882277", "22883399"
    recorded = [
        {"id": "a", "sender_id": first, "sender_name": "小满", "text": "等下开一把吗"},
        {"id": "b", "sender_id": second, "sender_name": "小满", "text": "现在就来", "reply_to_message_id": "a"},
    ]
    output = {
        "members": [{"name": "小满", "user_id": "U1"}, {"name": "小满", "user_id": "U2"}],
        "scenes": [{"name": "约游戏", "statements": [
            {"source_line": 2, "text": "现在就来", "speaker": "U1", "target": "模型误写的昵称"},
        ]}],
    }
    plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": json.dumps(output)})

    async def scenario() -> None:
        result = await analyzer.analyze_window(analyzer.with_reply_contexts(recorded))
        transcript = plugin.ctx.llm.generate.await_args.args[0][1]["content"]
        assert "小满（U2） 对「小满」说（对象 U1）" in transcript
        assert first not in transcript and second not in transcript
        assert [member["user_id"] for member in result["members"]] == [first, second]
        statement = result["scenes"][0]["statements"][0]
        assert (statement["speaker"], statement["target"], statement["context"]) == (
            second, "小满", "等下开一把吗",
        )

    run(scenario())


def test_fusion_keeps_distinct_contexts_and_no_new_size_limit(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, merge={"ratio_a": 0, "max_total_statements": 0})
    options = plugin._merge_options_from_config()
    assert options["ratio"] == (0, 100)
    analyzer = StyleAnalyzer(plugin.ctx)
    common = {"text": "那行", "context": "明天来吗", "speaker": "101", "speaker_name": "小满"}
    first = {
        "overall_style": "习惯用短句",
        "style_summary": "简短、自然地接话",
        "style_snapshots": ["习惯用短句", "存档中的历史表达"],
        "scenes": [{"name": "闲聊", "statements": [
            dict(common, id="from-a"),
            *({"text": f"第 {index} 项内容", "speaker": "101"} for index in range(53)),
        ]}],
    }
    second = {
        "overall_style": "问句很少",
        "scenes": [{"name": "闲聊", "statements": [
            dict(common, id="from-b"),
            dict(common, id="different", context="今晚开游戏吗", speaker="202"),
        ]}],
    }
    invalid_model = AsyncMock(return_value=None)
    analyzer._generate_json = invalid_model

    async def scenario() -> None:
        merged = await analyzer.merge_partials([first, second], options=options)
        system, request = invalid_model.await_args.args
        assert "来源A:来源B = 0:100" in system and "整个模型所有场景合计" not in system
        assert "from-a" in request and "from-b" not in request and "different" in request
        assert second["scenes"][0]["statements"][0]["id"] == "from-b"
        statements = merged["scenes"][0]["statements"]
        assert len(statements) == 55
        matching = [(item["speaker"], item["context"]) for item in statements if item["text"] == "那行"]
        assert matching == [("101", "明天来吗"), ("202", "今晚开游戏吗")]
        assert "简短、自然地接话" in request and "存档中的历史表达" not in request
        assert merged["style_snapshots"] == ["习惯用短句", "存档中的历史表达", "问句很少"]

        keep_all = await analyzer.merge_partials([first, second], options=dict(options, dedup_level="不去重"))
        assert "from-b" in invalid_model.await_args.args[1]
        assert len(keep_all["scenes"][0]["statements"]) == 56

    run(scenario())


def test_failed_daily_analysis_retries_after_cooldown(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, collection={"daily_time": "00:00", "auto_analyze_count": 0})
    analyzer = plugin._analyzer = FakeAnalyzer()
    analyze = analyzer.analyze_window = AsyncMock(side_effect=[None, FakeAnalyzer.analyze_result])

    async def scenario() -> None:
        await plugin._store.bind_group(GROUP)
        await plugin.handle_message(sample_message(group_id=GROUP))
        await plugin._check_scheduled_analyze()
        await asyncio.sleep(0)
        tasks = asyncio.all_tasks() - {asyncio.current_task()}
        await asyncio.gather(*tasks)
        assert analyze.await_count == 1
        assert GROUP not in plugin._daily_trigger_date
        assert plugin._store.message_count(GROUP) == 1
        await plugin._check_scheduled_analyze()
        await plugin._trigger_auto_analyze(GROUP, reason="新消息触发")
        assert not (asyncio.all_tasks() - {asyncio.current_task()})
        assert analyze.await_count == 1

        plugin._analysis_retry_after[GROUP] = 0
        await plugin._check_scheduled_analyze()
        tasks = asyncio.all_tasks() - {asyncio.current_task()}
        await asyncio.gather(*tasks)
        assert analyze.await_count == 2
        assert GROUP in plugin._daily_trigger_date
        assert GROUP not in plugin._analysis_retry_after
        assert plugin._store.message_count(GROUP) == 0

    run(scenario())


def test_legacy_personal_style_flag_does_not_include_member_profiles(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={"enable_style_matching": True})
    plugin._rewrite_with_llm = AsyncMock(return_value="[[KEEP_ORIGINAL]]")

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "daily",
            "style_summary": "常用简短的自然口语",
            "style_snapshots": ["冗长但仍保存在模型的来源快照"],
            "members": [{"user_id": "101", "name": "档案昵称", "style": "单独的个人风格"}],
        })
        plugin._global_models = ["daily"]
        await plugin.handle_message(dict(sample_message(group_id=GROUP, sender="101"), message_id="reply"))
        await plugin.rewrite_style_response(
            session_id=SESSION, reply_message_id="reply", response="我也觉得今天的天气还不错"
        )
        reference = plugin._rewrite_with_llm.await_args.args[0]
        assert "常用简短的自然口语" in reference
        assert "档案昵称" not in reference and "来源快照" not in reference

    run(scenario())
