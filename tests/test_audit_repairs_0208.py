from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chat_style_translator_plugin.modules.analyzer import StyleAnalyzer
from test_plugin import run
from test_regressions_0201 import SESSION, make_env


def test_unlimited_merge_recovers_examples_from_a_valid_partial_model_result() -> None:
    analyzer = StyleAnalyzer(None)
    originals = [
        {"text": f"原始语句 {index:02}", "context": f"真实前文 {index:02}", "speaker": "101"}
        for index in range(70)
    ]
    first = {"scenes": [{"name": "约饭", "statements": originals[:35]}]}
    second = {"scenes": [
        {"name": "约饭", "statements": originals[35:]},
        {"name": "聊摄影", "statements": [{"text": "还有机身选择", "context": "镜头已经选好了"}]},
    ]}
    analyzer._generate_json = AsyncMock(return_value={
        "style_summary": "说话简短直白",
        "scenes": [{"name": "约饭", "statements": [originals[0], originals[-1]]}],
    })
    merged = run(analyzer.merge_partials([first, second], options={"max_total_statements": 0}))
    dinner, camera = merged["scenes"]
    actual = Counter((item["text"], item["context"]) for item in dinner["statements"])
    assert actual == Counter((item["text"], item["context"]) for item in originals)
    assert camera["statements"][0]["context"] == "镜头已经选好了"
    assert merged["style_summary"] == "说话简短直白"
    prompt = analyzer._generate_json.await_args.args[0]
    assert "每个场景最终保留不超过" not in prompt and "整个模型所有场景合计" not in prompt


def test_unlimited_merge_keeps_known_dedup_and_distinct_targets_and_contexts() -> None:
    analyzer = StyleAnalyzer(None)
    base = {"speaker": "101", "target": "小王", "context": "今天几点集合", "meaning": "确认七点集合"}
    first = {"scenes": [{"name": "邀约", "statements": [
        dict(base, text="今天七点到"),
        dict(base, text="今天七点到", target="小李"),
        dict(base, text="改天也七点到", context="周六再来吗"),
    ]}]}
    kept = dict(base, text="七点见面")
    second = {"scenes": [{"name": "邀约", "statements": [kept]}]}
    analyzer._generate_json = AsyncMock(return_value={
        "scenes": [{"name": "邀约", "statements": [kept]}],
    })
    merged = run(analyzer.merge_partials([first, second], options={"ratio": (0, 100)}))
    actual = [(item["text"], item["target"], item["context"]) for item in merged["scenes"][0]["statements"]]
    assert actual == [
        ("七点见面", "小王", "今天几点集合"),
        ("今天七点到", "小李", "今天几点集合"),
        ("改天也七点到", "小王", "周六再来吗"),
    ]


def test_unlimited_merge_with_dedup_off_recovers_original_multiplicity() -> None:
    analyzer = StyleAnalyzer(None)
    original = {"text": "我马上过来", "speaker": "101", "context": "出来吃饭吧"}
    left = {"scenes": [{"name": "约饭", "statements": [original, original]}]}
    right = {"scenes": [{"name": "约饭", "statements": [original, original]}]}
    analyzer._generate_json = AsyncMock(return_value=right)
    merged = run(analyzer.merge_partials([left, right], options={"dedup_level": "不去重"}))
    assert [item["text"] for item in merged["scenes"][0]["statements"]] == ["我马上过来"] * 4


def test_positive_explicit_total_cap_also_applies_when_the_llm_fails() -> None:
    analyzer = StyleAnalyzer(None)
    analyzer._generate_json = AsyncMock(return_value=None)
    first, second = [
        {"scenes": [{"name": topic, "statements": [{"text": f"{topic}语句 {i}"} for i in range(6)]}]}
        for topic in ("约饭", "看电影")
    ]
    merged = run(analyzer.merge_partials([first, second], options={"max_total_statements": 4}))
    assert sum(len(scene["statements"]) for scene in merged["scenes"]) == 4


@pytest.mark.parametrize("draft", ["禁词", "禁词禁词！！", "【禁词】……", "「禁词」", "禁另一个词词"])
def test_blacklist_abort_applies_before_model_rewrite_and_send(tmp_path: Path, draft: str) -> None:
    plugin = make_env(
        tmp_path,
        inject={"rewrite_strength": 0},
        pre_rewrite_blacklist={"enabled": True, "items": ["禁词", "另一个词"], "semantic_match": True},
    )
    generate = plugin.ctx.llm.generate = AsyncMock()
    result = run(plugin.rewrite_style_response(response=draft, session_id=SESSION))
    assert result == {
        "action": "abort",
        "custom_result": {"reason": "草稿的内容全部被微调前黑名单过滤，已拦截发送"},
    }
    assert "已拦截发送" in plugin._last_rewrite_trace["reason"]
    generate.assert_not_awaited()


def test_partially_filtered_draft_keeps_allowed_content(tmp_path: Path) -> None:
    plugin = make_env(
        tmp_path,
        inject={"rewrite_strength": 0},
        pre_rewrite_blacklist={"enabled": True, "items": ["禁词", "另一个词"]},
    )
    result = run(plugin.rewrite_style_response(response="禁另一个词词，下次见。", session_id=SESSION))
    assert result["action"] == "continue"
    assert result["modified_kwargs"]["response"] == "下次见"
