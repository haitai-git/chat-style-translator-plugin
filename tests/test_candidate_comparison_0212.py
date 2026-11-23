from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from test_plugin import run, sample_message
from test_regressions_0201 import GROUP, SESSION, make_env


@pytest.mark.parametrize(("draft", "candidate", "accepted"), [
    ("我们定在明天七点会面吧", "明天七点碰头吧", True),
    ("我们定在明天七点会面吧", "明天八点碰头吧", False),
    ("我们应该可以提前到", "我们肯定能提前到", False),
    ("我不会答应这件事", "我会答应这件事", False),
    ("我今天不来现场了", "我今天会来现场", False),
    ("这做出来效果挺不错的", "做出来效果挺好的", True),
    ("我可能可以早点过去", "我可能能早点过去", True),
    ("我可能可以早点过去", "[[KEEP_ORIGINAL]]", False),
])
def test_planner_rewrites_once_with_matched_context(tmp_path: Path, draft: str, candidate: str, accepted: bool) -> None:
    plugin = make_env(
        tmp_path, inject={"global_models": ["daily"], "rewrite_model": "planner", "candidate_model": "replyer"},
        pre_rewrite_blacklist={"enabled": True, "items": ["不存在的词"], "semantic_match": True},
    )
    target = "明天的事怎么安排"
    later = "另外还要看看别的消息"
    generate = plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": candidate})

    async def scenario() -> None:
        await plugin._store.save_model({"model_id": "daily", "style_summary": "短句，换口语词"})
        await plugin._refresh_global_models()
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=target), message_id="target"))
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=later), message_id="later"))
        result = await plugin.rewrite_style_response(response=draft, session_id=SESSION, reply_message_id="target")
        assert result.get("modified_kwargs", {}).get("response", draft) == (candidate if accepted else draft)
        generate.assert_awaited_once()
        call = generate.await_args
        assert call.kwargs["model"] == "planner"
        prompt = call.args[0][1]["content"]
        assert target in prompt and later not in prompt and "语义黑名单词条" in prompt

    run(scenario())
