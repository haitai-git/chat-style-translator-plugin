from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from chat_style_translator_plugin.modules.analyzer import StyleAnalyzer
from chat_style_translator_plugin.modules.injector import build_rewrite_reference

from test_plugin import run, sample_message
from test_regressions_0201 import GROUP, SESSION, make_env


def test_quoted_style_example_is_learned_without_copying_the_reply(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, collection={"auto_analyze_count": 0}, inject={"reference_mode": "按场景"})
    plugin._analyzer = StyleAnalyzer(plugin.ctx)
    prior = "你看这几个外观里有喜欢的吗？"
    reply = "这几个造型太普通，先看下一批"
    model = {"overall_style": "短句口语", "scenes": [{
        "name": "评价外观",
        "statements": [
            {"text": reply, "source_line": 3, "context": "分析模型编造的前文", "target": "大家",
             "ai_note": "熟人互损时才用的群内梗"},
            {"text": "再看看其他款吧", "source_line": 2, "context": "没有记录的前文"},
        ],
    }]}
    plugin.ctx.llm.generate = AsyncMock(side_effect=[
        {"success": True, "response": json.dumps(model, ensure_ascii=False)},
        {"success": True, "response": reply + "！"},
        {"success": True, "response": "这几个都挺一般，先看看别的吧"},
    ])

    async def scenario() -> None:
        await plugin._store.bind_group(GROUP)
        for message_id, text, quoted in (
            ("old", prior, None),
            ("other", "再看看其他款吧", {"target_message_id": "unavailable"}),
            ("answer", reply, {"target_message_id": "old"}),
        ):
            message = sample_message(group_id=GROUP, text=text, reply=quoted)
            await plugin.handle_message(dict(message, message_id=message_id))
        assert (await plugin.run_analysis(GROUP)).startswith("分析完成")
        transcript = plugin.ctx.llm.generate.await_args.args[0][1]["content"]
        assert f"引用前文「{prior}」" in transcript
        statements = plugin._store.list_models()[0]["scenes"][0]["statements"]
        assert len(statements) == 2
        assert statements[0]["context"] == prior
        assert "context" not in statements[1] and "source_line" not in statements[0]
        file_id = plugin._store.list_model_ids()[0]
        await plugin.cmd_global_enable(is_local_operator=True, matched_groups={"model_name": file_id})

        incoming = sample_message(group_id=GROUP, text="这几个外观里有喜欢的吗？")
        await plugin.handle_message(dict(incoming, message_id="target"))
        kwargs = {"response": "我觉得这几款都一般，还是看看别的吧", "session_id": SESSION, "reply_message_id": "target"}
        assert await plugin.rewrite_style_response(**kwargs) == {"action": "continue"}
        messages = plugin.ctx.llm.generate.await_args.args[0]
        assert "不能照抄" in messages[0]["content"]
        assert f"前文「{prior}」" in messages[1]["content"]
        result = await plugin.rewrite_style_response(**kwargs)
        assert result["modified_kwargs"]["response"] == "这几个都挺一般，先看看别的吧"

    run(scenario())


def test_backfill_pairs_only_explicit_unfiltered_quotes(tmp_path: Path) -> None:
    plugin = make_env(
        tmp_path, collection={"auto_analyze_count": 0},
        phrase_blacklist={"items": ["被屏蔽"]},
    )

    def raw(message_id, text, reference=None):
        content = [{"type": "text", "data": {"text": text}}]
        if reference is not None:
            content.insert(0, {"type": "reply", "data": {"id": reference}})
        return {"message_id": message_id, "time": message_id, "message": content}

    page = [
        raw(1, "你感觉这件怎么样"),
        raw(2, "还挺顺眼的", 1),
        raw(3, "被屏蔽的聊天内容"),
        raw(4, "再说吧", 3),
        raw(5, "我先撤了"),
    ]
    plugin.ctx.api.call = AsyncMock(side_effect=[{"data": {"messages": page}}, {"data": {"messages": []}}])

    async def scenario() -> None:
        await plugin._store.bind_group(GROUP)
        assert "历史消息 4 条" in await plugin.run_backfill(GROUP, count=5)
        prepared = StyleAnalyzer.with_reply_contexts(plugin._store.messages_for(GROUP))
        assert prepared[1]["reply_to_message_id"] == "1"
        assert prepared[1]["quoted_context"] == "你感觉这件怎么样"
        assert all("quoted_context" not in record for record in prepared[2:])

    run(scenario())


def test_fusion_restores_only_source_contexts_and_compaction_keeps_distinct_pairs(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)
    analyzer = StyleAnalyzer(plugin.ctx)
    first, second = {
        "scenes": [{"name": "群聊", "statements": [{"text": "都可以", "context": "准备去哪家吃饭"}, {"text": "我选第一个"}]}],
    }, {
        "scenes": [{"name": "群聊", "statements": [{"text": "都可以", "context": "明天还是后天出门"}, {"text": "我下午再看", "context": "现在方便确认吗"}]}],
    }
    plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": json.dumps({
        "scenes": [{"name": "群聊", "statements": [
            {"text": "都可以", "context": "准备去哪家吃饭"},
            {"text": "都可以", "context": "明天还是后天出门"},
            {"text": "我选第一个", "context": "融合模型自行补写"},
            {"text": "我下午再看"},
        ]}]
    }, ensure_ascii=False)})

    fused = run(analyzer.merge_partials([first, second]))
    compacted, _ = analyzer.compact_model(fused)
    statements = compacted["scenes"][0]["statements"]
    assert [statement.get("context") for statement in statements] == [
        "准备去哪家吃饭", "明天还是后天出门", None, "现在方便确认吗"
    ]


def test_context_can_promote_an_example_after_the_first_four() -> None:
    model = {"scenes": [{"name": "群聊", "statements": [
        *({"text": f"日常聊天范例{index}"} for index in range(4)),
        {"text": "这次我就先不跟了", "context": "今晚还要一起刷副本吗"},
    ]}]}
    examples: set[str] = set()
    reference = build_rewrite_reference(
        model, context="今晚还要一起刷副本吗\n我想拒绝", incoming_text="今晚还要一起刷副本吗",
        visible_examples=examples,
    )
    assert "前文「今晚还要一起刷副本吗」" in reference
    assert "这次我就先不跟了" in examples
    assert "日常聊天范例3" not in examples



def test_fusion_cannot_borrow_context_from_a_different_speaker(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)
    analyzer = StyleAnalyzer(plugin.ctx)
    first = {"scenes": [{"name": "群聊", "statements": [
        {"text": "晚点再看吧", "speaker": "9001", "context": "方便帮我看一下吗"},
    ]}]}
    second = {"scenes": [{"name": "群聊", "statements": [
        {"text": "晚点再看吧", "speaker": "9002"},
    ]}]}
    plugin.ctx.llm.generate = AsyncMock(return_value={"success": True, "response": json.dumps({
        "scenes": [{"name": "群聊", "statements": [
            {"text": "晚点再看吧", "speaker": "9002", "context": "方便帮我看一下吗"},
        ]}]
    }, ensure_ascii=False)})
    fused = run(analyzer.merge_partials([first, second]))
    assert "context" not in fused["scenes"][0]["statements"][0]
