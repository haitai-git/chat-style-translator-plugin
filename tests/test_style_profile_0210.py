from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from chat_style_translator_plugin.modules import storage
from chat_style_translator_plugin.modules.analyzer import StyleAnalyzer
from chat_style_translator_plugin.modules.storage import StyleStore
from test_plugin import FakeAnalyzer, run, sample_message
from test_regressions_0201 import GROUP, SESSION, make_env


def test_generate_saves_profile_before_any_reply_and_rebuilds_manually_edited_file(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)
    plugin._analyzer = FakeAnalyzer()
    statements = ["先看看吧", "等会再来吧", "差不多就行吧"]

    async def scenario() -> None:
        store = plugin._store
        await store.bind_group(GROUP)
        for text in statements:
            await store.append_message(GROUP, {"text": text})
        assert "已生成模型" in await plugin.run_analysis(GROUP)
        model_path, = store.model_dir.glob("*.json")
        file_id = model_path.stem
        reopened = StyleStore(tmp_path)
        with patch.object(storage, "_read_json", side_effect=AssertionError("read full model")):
            profile = reopened.get_model_profile(file_id)
            assert "「吧」收尾" in profile["observations"]
            assert profile["statement_count"] == 3
            assert "scenes" not in profile
            assert reopened.find_model_file(profile["name"]) == file_id

        full = reopened.get_model(file_id)
        path = reopened.model_dir / f"{file_id}.json"
        full["scenes"][0]["statements"] = [
            {"text": text} for text in ["已经开门嘛", "是这里嘛", "这个能换嘛", "原来这样嘛"]
        ]
        path.write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
        newer = StyleStore(tmp_path)
        assert "「嘛」收尾" in newer.get_model_profile(file_id)["observations"]
        assert newer.get_model_profile(file_id)["statement_count"] == 4

    run(scenario())


def test_large_import_bootstraps_once_then_replies_with_model_summary(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={"global_models": ["large"], "semantic_retrieval": True})
    texts = [f"这个是第{number}条的例句啊" for number in range(6000)]
    model = {
        "model_id": "large",
        "style_summary": "少铺垫，常用口语",
        "members": [
            {"user_id": "20002", "style": "回话短，语气平和"},
            {"user_id": "20003", "style": "另一个人的表达"},
        ],
        "scenes": [{"name": "分享数据", "statements": [{"text": text, "context": "原始前文"} for text in texts]}],
    }
    store = plugin._store
    store.model_dir.mkdir()
    (store.model_dir / "large.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")

    async def scenario() -> None:
        await plugin._refresh_global_models()
        embed = plugin.ctx.llm.embed = AsyncMock(side_effect=AssertionError("requested embedding"))
        generate = plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": "明天七点碰头吧"})
        plugin._store = StyleStore(tmp_path)
        read_json = storage._read_json

        def guard_model_file(path, default):
            assert path.parent != store.model_dir, "large model was read again"
            return read_json(path, default)

        with patch.object(storage, "_read_json", side_effect=guard_model_file):
            await plugin._refresh_global_models()
            await plugin.handle_message(sample_message(group_id=GROUP, text="时间怎么安排"))
            result = await plugin.rewrite_style_response(
                response="我们定在明天七点会面吧", session_id=SESSION, reply_message_id="msg-1"
            )
            assert result["modified_kwargs"]["response"] == "明天七点碰头吧"
            evidence = generate.await_args.args[0][1]["content"]
            assert "少铺垫，常用口语" in evidence and "「啊」收尾" in evidence
            assert "回话短，语气平和" not in evidence and "另一个人的表达" not in evidence
            assert "分享数据" not in evidence and "原始前文" not in evidence and "第0条的例句" not in evidence
            assert len(evidence) < 1000
            embed.assert_not_called()

    run(scenario())


def test_merge_and_compact_cross_group_sources_then_override_global_model_for_one_group(tmp_path: Path) -> None:
    group_a, group_b = "90002", "90003"
    plugin = make_env(
        tmp_path, groups={"items": [{"group_id": GROUP, "models": ["合用"], "prefer_group_models": True}]},
        inject={"global_models": ["global"]},
    )
    plugin._analyzer = StyleAnalyzer(plugin.ctx)
    plugin._analyzer._generate_json = AsyncMock(return_value=None)
    a_texts = ["先过去吧", "一会儿再说吧", "那就改天吧"]
    b_texts = ["现在就走嘛", "今天开门嘛", "是这一家嘛"]

    async def scenario() -> None:
        store = plugin._store
        a = {
            "model_id": "a", "group_id": group_a, "name": "甲", "style_summary": "说话比较省略",
            "scenes": [{"name": "商量时间", "statements": [{"text": text, "context": "问什么时候"} for text in a_texts]}],
        }
        b = {
            "model_id": "b", "group_id": group_b, "name": "乙", "style_summary": "常用小疑问表达关心",
            "scenes": [{"name": "问行程", "statements": [{"text": text, "context": "问去哪儿"} for text in b_texts]}],
        }
        for model in (a, b, {"model_id": "global", "overall_style": "很规整，完整的句子"}):
            await store.save_model(model)
        assert (await plugin.merge_models(a, b, "合用")).startswith("融合完成")
        merged_file = store.find_model_file("合用")
        merged = store.get_model(merged_file)
        assert merged["source_group_ids"] == [group_a, group_b]
        assert not merged["group_id"] and merged["group_only"]
        assert {record["text"] for scene in merged["scenes"] for record in scene["statements"]} == set(a_texts + b_texts)
        assert "「吧、嘛」收尾" in store.get_model_profile(merged_file)["observations"]

        assert (await plugin.compact_model(merged, "合用精简")).startswith("精简完成")
        compacted_file = store.find_model_file("合用精简")
        compacted = store.get_model(compacted_file)
        assert compacted["source_group_ids"] == [group_a, group_b] and compacted["group_only"]
        assert store.get_model_profile(compacted_file)["statement_count"] == 6

        await plugin._sync_groups_from_config()
        await plugin._refresh_global_models()
        assert plugin._resolve_active_models(GROUP) == [merged_file]
        assert plugin._resolve_active_models(group_a) == ["global"]
        assert plugin._resolve_active_models(group_b) == ["global"]
        generate = plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": "明天七点碰头吧"})
        result = await plugin.rewrite_style_response(response="我们定在明天七点会面吧", session_id=SESSION)
        assert result["modified_kwargs"]["response"] == "明天七点碰头吧"
        evidence = generate.await_args.args[0][1]["content"]
        assert "「吧、嘛」收尾" in evidence and "很规整" not in evidence
        assert "说话比较省略" in evidence and "常用小疑问表达关心" in evidence
        assert all(text not in evidence for text in a_texts + b_texts)
        assert "群挂载" in plugin._last_rewrite_trace["models"][0]
        await store.delete_model(merged_file)
        assert plugin._resolve_active_models(GROUP) == ["global"]

    run(scenario())
