from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chat_style_translator_plugin.modules.injector import build_rewrite_reference
from test_plugin import run, sample_message
from test_regressions_0201 import GROUP, SESSION, make_env


def test_ordinary_context_uses_style_and_still_rewrites(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={"reference_mode": "按场景"})
    incoming = "这几件外观里有合适的吗"
    original_example = "这几件都算了，下一个"
    plugin.ctx.llm.generate = AsyncMock(return_value={
        "success": True, "response": "这几件都不太合适，再看下一组吧"
    })

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "daily", "scenes": [{
                "name": "挑外观", "description": "评价几件外观，避免接梗", "style": "少铺垫，用短句",
                "statements": [{
                    "context": incoming,
                    "text": original_example,
                    "ai_note": "直接给出偏好和下一步，别用群内梗",
                }]
            }]
        })
        plugin._global_models = ["daily"]
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=incoming), message_id="in"))
        result = await plugin.rewrite_style_response(
            response="我认为这批都不怎么合适，我们接着看看下一组吧",
            session_id=SESSION, reply_message_id="in",
        )
        assert result["modified_kwargs"]["response"] == "这几件都不太合适，再看下一组吧"
        plugin.ctx.llm.generate.assert_awaited_once()
        system, user = plugin.ctx.llm.generate.await_args.args[0]
        assert "口语用词、语气、句子节奏" in system["content"]
        assert "少铺垫，用短句" in user["content"]
        assert "直接给出偏好和下一步" not in user["content"]
        assert original_example not in user["content"]

    run(scenario())


def test_unmatched_context_still_sends_group_guidance(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)
    plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": "好，那就这么定吧"})

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "daily", "scenes": [{"name": "下棋", "statements": [{"text": "兵往前走"}]}]
        })
        plugin._global_models = ["daily"]
        result = await plugin.rewrite_style_response(response="可以，就这么确定吧", session_id=SESSION)
        assert result["modified_kwargs"]["response"] == "好，那就这么定吧"
        prompt = plugin.ctx.llm.generate.await_args.args[0][1]["content"]
        assert "接话提示" not in prompt and "兵往前走" not in prompt

    run(scenario())


def test_only_close_contextual_case_may_include_one_quote() -> None:
    incoming = "我打得太菜，干脆当训练假人吧"
    text = "行，那我先测试一下输出"
    model = {"scenes": [
        {"name": "输局自嘲", "statements": [
            {"context": incoming, "text": text, "ai_note": "对方先自嘲再接梗"},
            {"context": incoming, "text": "我来装作没看见", "ai_note": "熟人互损"},
        ]},
        {"name": "游戏群内梗", "statements": [
            {"context": incoming, "text": "这是另一份原话", "ai_note": "群内暗号"},
        ]},
    ]}
    examples: set[str] = set()
    reference = build_rewrite_reference(
        model, context=incoming, visible_examples=examples, style_only=True
    )
    assert examples == {text} and "前文" in reference
    weak = build_rewrite_reference(model, context="游戏输了该怎么打", style_only=True)
    assert text not in weak


@pytest.mark.parametrize(("draft", "rewrite"), [
    ("行，那我先测试一下输出啦", "行，那我先测试一下输出吧"),
    ("行，那我先测试一下输出吧。", "行，那我先测试一下输出吧！"),
    ("行，那我先测试一下输出", "行，那我先测试一下输出吧"),
])
def test_low_strength_accepts_a_single_edit_even_close_to_example(
    tmp_path: Path, draft: str, rewrite: str
) -> None:
    plugin = make_env(tmp_path, inject={
        "reference_mode": "按场景", "rewrite_strength": 5, "enrich_min_chars": 20,
        "enrich_short_replies": True,
    })
    incoming = "我打得太菜，干脆当训练假人吧"
    plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": rewrite})

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "special", "scenes": [{
                "name": "自嘲接梗", "statements": [{
                    "context": incoming, "text": "行，那我先测试一下输出吧",
                    "ai_note": "朋友先自嘲，顺势开个玩笑",
                }]
            }]
        })
        plugin._global_models = ["special"]
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=incoming), message_id="case"))
        result = await plugin.rewrite_style_response(
            response=draft, session_id=SESSION, reply_message_id="case"
        )
        assert result["modified_kwargs"]["response"] == rewrite
        assert "说：“行，那我先测试一下输出吧”" in plugin.ctx.llm.generate.await_args.args[0][1]["content"]

    run(scenario())


def test_explicit_keep_returned_only_after_the_rewrite_call(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={"rewrite_strength": 5})
    plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": "[[KEEP_ORIGINAL]]"})

    async def scenario() -> None:
        await plugin._store.save_model({"model_id": "daily", "overall_style": "日常简短口语"})
        plugin._global_models = ["daily"]
        result = await plugin.rewrite_style_response(response="行，那我先去看看", session_id=SESSION)
        assert result == {"action": "continue"}
        plugin.ctx.llm.generate.assert_awaited_once()
        assert "低强度可以只改一个字或标点" in plugin.ctx.llm.generate.await_args.args[0][0]["content"]

    run(scenario())



@pytest.mark.parametrize(("draft", "minimum", "judge_short", "candidate", "mode", "expected"), [
    ("现在出发！！😀", 5, False, "现在就出发！！", "skip", "现在出发！！😀"),
    ("现在就出发。", 5, False, "现在这就出发。", "force", "现在这就出发。"),
    ("现在就出发。", 5, False, "现在就出发！", "force", "现在就出发。"),
    ("现在出发", 4, False, "现在就出发", "force", "现在就出发"),
    ("现在出发", 5, True, "现在就出发", "judge", "现在就出发"),
    ("现在就出发", 6, True, "[[KEEP_ORIGINAL]]", "judge", "现在就出发"),
    ("嗯", 5, True, "[[KEEP_ORIGINAL]]", "judge", "嗯"),
    ("嗯", 0, False, "嗯嗯", "force", "嗯嗯"),
    ("选 A 号 2", 4, False, "我选 A 号 2", "force", "我选 A 号 2"),
])
def test_word_enrichment_threshold_and_short_judgment(
    tmp_path: Path, draft: str, minimum: int, judge_short: bool, candidate: str, mode: str, expected: str,
) -> None:
    bot = make_env(tmp_path, inject={
        "global_models": ["daily"], "enrich_min_chars": minimum, "enrich_short_replies": judge_short,
    })
    generate = bot.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": candidate})

    async def scenario() -> None:
        await bot._store.save_model({"model_id": "daily", "overall_style": "口语简短"})
        await bot._refresh_global_models()
        incoming = "准备好了吗"
        await bot.handle_message(dict(sample_message(group_id=GROUP, text=incoming), message_id="target"))

        result = await bot.rewrite_style_response(
            response=draft, session_id=SESSION, reply_message_id="target",
        )
        assert result.get("modified_kwargs", {}).get("response", draft) == expected
        if mode == "skip":
            generate.assert_not_awaited()
            return
        generate.assert_awaited_once()
        assert generate.await_args.kwargs["model"] == "planner"
        system, prompt = generate.await_args.args[0]
        assert incoming in prompt["content"]
        assert ("本次必须补词，即使草稿已经自然也要补" in system["content"]) == (mode == "force")
        assert ("先结合来话判断" in system["content"]) == (mode == "judge")

    run(scenario())
