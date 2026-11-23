from pathlib import Path
from unittest.mock import AsyncMock

from chat_style_translator_plugin import plugin as plugin_module
from chat_style_translator_plugin.modules.retrieval import Scene
from test_plugin import run, sample_message
from test_regressions_0201 import GROUP, SESSION, make_env


def test_whole_group_style_works_on_unmatched_drafts_without_scene_or_embedding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugin_module, "retrieve_scenes", AsyncMock(side_effect=AssertionError("unexpected scene search")))
    def unexpected_scene_index(*args):
        raise AssertionError("unexpected scene index")

    monkeypatch.setattr(Scene, "from_data", unexpected_scene_index)
    plugin = make_env(tmp_path, inject={"semantic_retrieval": True, "global_models": ["daily"]})
    embed = plugin.ctx.llm.embed = AsyncMock()
    plugin.ctx.llm.generate = AsyncMock(side_effect=[
        {"success": True, "response": "明天七点碰头吧"},
        {"success": True, "response": "这张卡得去柜台激活"},
    ])
    originals = ["要不今天就撤吧", "那谁在线上嘛", "行，先回去吧", "这个有点贵啊"]

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "daily", "style_summary": "少铺垫，常把书面词换成口语词",
            "scenes": [{
                "name": "讨论游戏掉率",
                "statements": [{"text": text, "ai_note": "统一先说当然可以，再询问下一步"} for text in originals],
            }],
        })
        await plugin._refresh_global_models()
        assert plugin._semantic_index is None
        for index, (draft, expected) in enumerate((
            ("我们定在明天七点会面吧", "明天七点碰头吧"),
            ("这张卡需要到柜台完成激活", "这张卡得去柜台激活"),
        )):
            message = dict(sample_message(group_id=GROUP, text="请帮我确认一下"), message_id=f"in{index}")
            await plugin.handle_message(message)
            result = await plugin.rewrite_style_response(
                response=draft, session_id=SESSION, reply_message_id=f"in{index}"
            )
            assert result["modified_kwargs"]["response"] == expected
            messages = plugin.ctx.llm.generate.await_args.args[0]
            evidence = messages[1]["content"]
            assert "样本长度中位数" in evidence and "省略句末标点" in evidence
            assert "部分句子用「吧」收尾" in evidence
            assert "少铺垫，常把书面词换成口语词" in evidence
            assert "统一先说当然可以" not in evidence and "讨论游戏掉率" not in evidence
            assert all(sample not in evidence for sample in originals)
            assert "不加通用开场、固定收尾" in messages[0]["content"]
        embed.assert_not_awaited()
        widgets = await plugin.get_config_widget_state()
        assert "整体风格" in widgets["reply_preview"].text
        assert "表达风格参考" in widgets["reply_preview"].text

    run(scenario())


def test_whole_group_observations_refresh_after_model_replacement(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)

    async def scenario() -> None:
        for batch, style in (
            (["先看看吧", "待会再来吧", "差不多就行吧"], "「吧」收尾"),
            (["这么贵嘛", "已经开门嘛", "你确定了嘛"], "「嘛」收尾"),
        ):
            await plugin._store.save_model({"model_id": "daily", "scenes": [{
                "name": "群内聊天", "statements": [{"text": text} for text in batch],
            }]})
            assert style in plugin._get_search_models(["daily"])[0][1].group_observations

    run(scenario())
