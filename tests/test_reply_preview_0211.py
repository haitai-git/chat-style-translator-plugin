from pathlib import Path
from unittest.mock import AsyncMock

from test_plugin import run
from test_regressions_0201 import SESSION, make_env


def test_preview_shows_before_and_adopted_reply_with_short_style_line(tmp_path: Path) -> None:
    bot = make_env(tmp_path, inject={"global_models": ["daily"]})
    draft = "我们定在明天七点会面吧"
    adopted = "明天七点碰头吧"
    rejected = "明天八点碰头吧"
    generate = bot.ctx.llm.generate = AsyncMock(side_effect=[
        {"success": True, "response": adopted},
        {"success": True, "response": rejected},
    ])
    snapshot = "历史讨论细节占了很多篇幅，" * 12

    async def scenario() -> None:
        await bot._store.save_model({
            "model_id": "daily",
            "style_snapshots": [str(number) + snapshot for number in range(7)],
            "scenes": [{"statements": [{"text": text} for text in ["那先这样吧", "等一下吧", "回头再看吧"]]}],
        })
        await bot._refresh_global_models()

        result = await bot.rewrite_style_response(response=draft, session_id=SESSION)
        assert result["modified_kwargs"]["response"] == adopted
        preview = (await bot.get_config_widget_state())["reply_preview"].text
        assert f"微调前　{draft}" in preview and f"微调后　{adopted}" in preview
        evidence = next(line for line in preview.splitlines() if line.startswith("表达风格参考"))
        assert "样本长度中位数" in evidence and len(evidence) < 75
        assert "历史讨论细节" not in preview
        assert "历史讨论细节" in generate.await_args.args[0][1]["content"]

        assert await bot.rewrite_style_response(response=draft, session_id=SESSION) == {"action": "continue"}
        preview = (await bot.get_config_widget_state())["reply_preview"].text
        assert f"微调前　{draft}" in preview and f"微调后　{draft}" in preview
        assert rejected not in preview and "未通过内容校验" in preview

    run(scenario())


def test_preview_shows_blacklist_fallback_and_blocked_output(tmp_path: Path) -> None:
    bot = make_env(tmp_path, pre_rewrite_blacklist={"enabled": True, "items": ["禁词"]})

    async def scenario() -> None:
        result = await bot.rewrite_style_response(response="禁词先放一边", session_id=SESSION)
        assert result["modified_kwargs"]["response"] == "先放一边"
        preview = (await bot.get_config_widget_state())["reply_preview"].text
        assert "微调前　禁词先放一边" in preview and "微调后　先放一边" in preview

        result = await bot.rewrite_style_response(response="禁词", session_id=SESSION)
        assert result["action"] == "abort"
        preview = (await bot.get_config_widget_state())["reply_preview"].text
        assert "微调前　禁词" in preview and "微调后　已拦截发送" in preview

    run(scenario())
