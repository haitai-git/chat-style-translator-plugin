from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from chat_style_translator_plugin import plugin as plugin_module
from chat_style_translator_plugin.modules import retrieval
from chat_style_translator_plugin.modules.injector import build_ranked_scene_reference
from chat_style_translator_plugin.modules.retrieval import Scene, SceneHit, SearchModel, SearchModelCache, SemanticIndex
from test_plugin import run, sample_message
from test_regressions_0201 import GROUP, SESSION, make_env


DINNER = {
    "name": "线下聚餐",
    "description": "相约用餐，商量去哪家餐馆",
    "style": "随口确认，句子简短",
    "statements": [{"text": "人工整理的例句", "meaning": "先确定去不去，再约地点"}],
}
GAMES = {
    "name": "排位组队", "description": "确认几人组队和所需位置",
    "statements": [{"meaning": "先询问队伍还缺哪个位置"}],
}
JOKE = {
    "name": "自嘲接梗", "description": "朋友自嘲时接群内梗",
    "statements": [{"text": "旧群梗原话", "ai_note": "对熟人开玩笑", "context": "吃点东西就随机摸盒吧"}],
}


def payload(values: list[float], model: str = "fake-embedding") -> dict:
    return {"success": True, "embedding": values, "model_name": model}


def for_documents(documents: list[str]) -> dict:
    vectors = [
        payload([1.0, 0.0] if doc.startswith("线下聚餐") else [0.0, 1.0])
        for doc in documents
    ]
    return {"success": True, "results": vectors}


def test_hybrid_routes_synonym_and_preview_uses_actual_reference(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={
        "reference_mode": "按场景", "semantic_retrieval": True, "global_models": ["daily"], "max_chars": 2500,
    })
    embed = plugin.ctx.llm.embed = AsyncMock(side_effect=[
        for_documents([DINNER["name"], GAMES["name"]]),
        payload([1.0, 0.0]),
    ])
    rewrite = plugin._rewrite_with_llm = AsyncMock(return_value="好，今晚挑个地方一起吃吧")

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "daily", "style_summary": "平时短句接话",
            "members": [{"user_id": "774401", "name": "秘密昵称"}],
            "scenes": [DINNER, GAMES, JOKE],
        })
        await plugin._refresh_global_models()
        await plugin._semantic_index._worker
        documents = embed.await_args.kwargs["texts"]
        assert len(documents) == 2
        assert any("先确定去不去" in doc for doc in documents)
        assert all("旧群梗原话" not in doc and "人工整理的例句" not in doc and "774401" not in doc for doc in documents)

        lexical = "线下聚餐吃哪家"
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=lexical), message_id="exact"))
        result = await plugin.rewrite_style_response(
            session_id=SESSION, reply_message_id="exact", response="好，我们今晚找个地方吃饭吧"
        )
        assert result["modified_kwargs"]["response"] == "好，今晚挑个地方一起吃吧"
        assert embed.await_count == 1
        assert "随口确认，句子简短" in rewrite.await_args.args[0]
        assert "先确定去不去" not in rewrite.await_args.args[0]

        incoming = "要不要出去搓一顿"
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=incoming), message_id="synonym"))
        await plugin.rewrite_style_response(
            session_id=SESSION, reply_message_id="synonym", response="好，我们今晚找个地方吃饭吧"
        )
        assert embed.await_count == 2
        assert embed.await_args.kwargs == {"text": incoming, "task_name": "embedding"}
        reference = rewrite.await_args.args[0]
        assert len(reference) <= 2500
        assert "线下聚餐" in reference and "排位组队" not in reference and "旧群梗原话" not in reference
        widgets = await plugin.get_config_widget_state()
        assert "2/2 个场景已索引" in widgets["active_models"].text
        preview = widgets["reply_preview"].text
        assert "混合检索" in preview and "线下聚餐（向量 1.00）" in preview and incoming not in preview

        await plugin.rewrite_style_response(
            session_id=SESSION, reply_message_id="synonym", response="好，我们今晚找个地方吃饭吧"
        )
        assert embed.await_count == 2

        await plugin.rewrite_style_response(
            session_id=SESSION, reply_message_id="missing", response="那你有什么好主意吗"
        )
        assert embed.await_count == 2

    run(scenario())


def test_new_fusion_reuses_cached_vectors_across_plugin_restarts(tmp_path: Path) -> None:
    old_embed = AsyncMock(return_value=for_documents([DINNER["name"], GAMES["name"]]))
    current_embed = AsyncMock(return_value=for_documents([DINNER["name"]]))
    after = dict(DINNER, description="相约用餐，优先确认离谁更近")
    path = tmp_path / "vectors.sqlite3"

    async def scenario() -> None:
        original = SemanticIndex(path, "embedding", old_embed)
        original.update([Scene.from_data(DINNER), Scene.from_data(GAMES)])
        await original._worker
        old_embed.assert_awaited_once()

        updated = SemanticIndex(path, "embedding", current_embed)
        scenes = tuple(Scene.from_data(item) for item in (after, GAMES, GAMES))
        updated.update(list(scenes))
        await updated._worker
        assert updated.ready_for(SearchModel({}, scenes))
        documents = current_embed.await_args.kwargs["texts"]
        assert len(documents) == 1 and "优先确认离谁更近" in documents[0]
        current_embed.assert_awaited_once()

        reopened = SemanticIndex(path, "embedding", current_embed)
        reopened.update(list(scenes))
        await reopened._worker
        assert reopened.ready_for(SearchModel({}, scenes))
        current_embed.assert_awaited_once()

    run(scenario())


def test_ambiguous_vectors_use_random_expression_without_exposing_topics(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={"reference_mode": "按场景", "semantic_retrieval": True, "global_models": ["ambiguous"]})
    embed = plugin.ctx.llm.embed = AsyncMock(side_effect=[
        {"success": True, "results": [payload([1.0, 0.0]), payload([1.0, 0.01])]},
        payload([1.0, 0.0]),
    ])
    rewrite = plugin._rewrite_with_llm = AsyncMock(return_value="好，我觉得这样就行")

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "ambiguous", "overall_style": "日常口语", "scenes": [DINNER, GAMES],
        })
        await plugin._refresh_global_models()
        await plugin._semantic_index._worker
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text="感觉这么弄就成"), message_id="ordinary"))
        result = await plugin.rewrite_style_response(
            response="好的，我认为这样就可以了", session_id=SESSION, reply_message_id="ordinary"
        )
        assert result["modified_kwargs"]["response"] == "好，我觉得这样就行"
        assert embed.await_count == 2
        reference = rewrite.await_args.args[0]
        assert "随口确认，句子简短" in reference
        assert "线下聚餐" not in reference and "排位组队" not in reference
        assert "人工整理的例句" not in reference
        preview = (await plugin.get_config_widget_state())["reply_preview"].text
        assert "候选相近" in preview and "随机场景" in preview

    run(scenario())


def test_keyword_index_finds_later_examples_and_ignores_false_special_hint(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={"reference_mode": "按场景", "global_models": ["daily"]})
    embed = plugin.ctx.llm.embed = AsyncMock()
    rewrite = plugin._rewrite_with_llm = AsyncMock(return_value="[[KEEP_ORIGINAL]]")

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "daily", "name": "通用风格", "scenes": [{
                "name": "技术咨询", "style": "简短直说", "statements": [
                    *({"text": "旧例句", "meaning": "示例解答"} for _ in range(20)),
                    {"text": "北风镜像换源后重新登录", "ai_note": "先确认镜像版本"},
                    {"text": "旧群梗原话", "ai_note": "拿群内梗继续互损"},
                ],
            }],
        })
        await plugin._refresh_global_models()
        await plugin.handle_message(dict(sample_message(
            group_id=GROUP, text="北风镜像换源后要重新登录吗"
        ), message_id="later-example"))
        await plugin.rewrite_style_response(
            response="可以换源后再重新登录一次", session_id=SESSION, reply_message_id="later-example"
        )
        reference = rewrite.await_args.args[0]
        assert "技术咨询" in reference and "简短直说" in reference and "先确认镜像版本" not in reference
        assert "拿群内梗继续互损" not in reference and "旧群梗原话" not in reference
        embed.assert_not_awaited()

    run(scenario())


def test_tight_reference_budget_prioritizes_the_best_scene_across_models(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={
        "reference_mode": "按场景", "global_models": ["first", "later"], "max_chars": 85,
    })
    rewrite = plugin._rewrite_with_llm = AsyncMock(return_value="[[KEEP_ORIGINAL]]")
    incoming = "镜像更换后的登录凭证该从哪看"
    weakest = "镜像"
    strongest = "镜像更换后的登录凭证"
    strongest_style = "用简短的口语确认状态"

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "first", "name": "前面的模型", "style_summary": "甲模型摘要" * 60,
            "scenes": [{
                "name": weakest, "description": "镜像", "style": "弱关联场景说明" * 30,
                "statements": [{"text": "不得进入提示的旧语句"}],
            }],
        })
        await plugin._store.save_model({
            "model_id": "later", "name": "后面的模型", "style_summary": "乙模型摘要" * 60,
            "scenes": [
                {"name": f"无关资料{index}", "description": "安排收书", "style": "用简短肯定句"}
                for index in range(20)
            ] + [{
                "name": strongest, "description": "镜像更换后重新确认登录凭证", "style": strongest_style,
                "statements": [{"text": "不得进入提示的历史原话"}],
            }],
        })
        await plugin._refresh_global_models()
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=incoming), message_id="matched"))
        await plugin.rewrite_style_response(
            response="我先帮你找一下吧", session_id=SESSION, reply_message_id="matched",
        )

        rewrite.assert_awaited_once()
        reference = rewrite.await_args.args[0]
        assert 0 < len(reference) <= 85
        assert f"【场景：{strongest}】" in reference and strongest_style in reference
        assert "甲模型摘要" not in reference and "乙模型摘要" not in reference
        assert "不得进入提示" not in reference and "无关资料" not in reference
        preview = (await plugin.get_config_widget_state())["reply_preview"].text
        assert f"{strongest}（关键词" in preview and "later（全局配置）" in preview

    run(scenario())


def test_scene_references_are_not_capped_at_two_hits_per_model(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={
        "reference_mode": "按场景", "global_models": ["daily"], "max_chars": 220,
    })
    rewrite = plugin._rewrite_with_llm = AsyncMock(return_value="[[KEEP_ORIGINAL]]")
    names = ["踏青装备", "踏青路线", "踏青出发", "踏青参与"]

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "daily", "name": "示例模型",
            "scenes": [{"name": name, "style": "简单说清楚"} for name in names],
        })
        await plugin._refresh_global_models()
        await plugin.handle_message(dict(sample_message(
            group_id=GROUP, text="、".join(names)
        ), message_id="multiple"))
        await plugin.rewrite_style_response(
            response="我问一下再回你吧", session_id=SESSION, reply_message_id="multiple",
        )

        rewrite.assert_awaited_once()
        reference = rewrite.await_args.args[0]
        assert 0 < len(reference) <= 220
        assert all(f"【场景：{name}】" in reference for name in names)

    run(scenario())


def test_scene_reference_fills_2500_chars_in_rank_order_with_whole_blocks() -> None:
    names = [f"场景编号{rank:02}" for rank in range(12)]
    styles = [f"风格提示{rank:02}" + "字" * 230 + f"结尾{rank:02}" for rank in range(12)]
    scenes = [Scene.from_data({"name": name, "style": style}) for name, style in zip(names, styles)]
    models = [
        SearchModel({"name": label}, tuple(scenes[index::2]))
        for index, label in enumerate(("第一份风格模型", "第二份风格模型"))
    ]
    hits = [SceneHit(scene, 40 - rank, confident=True) for rank, scene in enumerate(scenes)]
    ranked = retrieval.rank_scene_hits(models, [hits[::2], hits[1::2]])
    candidates = (
        (model_index, models[model_index].data["name"], hit.scene.data, hit.label())
        for model_index, hit in ranked
    )

    reference, included = build_ranked_scene_reference(candidates, 2500, visible_examples=set())
    blocks = reference.split("\n\n")
    assert len(reference) <= 2500
    assert len(blocks) == 9
    assert [name for _, name, _ in included] == names[:9]
    assert [index for index, _, _ in included] == [0, 1] * 4 + [0]
    for block, name, style in zip(blocks, names, styles):
        assert block.startswith(f"【场景：{name}】\n")
        assert f"场景风格：{style}" in block
    assert all(name not in reference for name in names[9:])


def test_unmatched_reply_samples_distinct_style_only_scenes_across_models(tmp_path: Path, monkeypatch) -> None:
    plugin = make_env(tmp_path, inject={
        "reference_mode": "按场景", "global_models": ["first", "later"], "max_chars": 300,
    })
    rewrite = plugin._rewrite_with_llm = AsyncMock(return_value="[[KEEP_ORIGINAL]]")
    ordinary = ("尽量口语化", "说话节奏快", "短句直说", "用词平实")

    def choose_last(population, k):
        assert len(population) == 4
        assert k == 3
        return list(reversed(population))[:k]

    monkeypatch.setattr(plugin_module, "sample", choose_last)

    async def scenario() -> None:
        for name, start, end in (("first", 0, 2), ("later", 2, 4)):
            await plugin._store.save_model({
                "model_id": name,
                "scenes": [
                    {
                        "name": f"原主题{index}",
                        "description": "播报钟表",
                        "style": ordinary[index],
                        "statements": [{"text": f"旧对话内容{index}", "context": "旧对话前文"}],
                    }
                    for index in range(start, end)
                ] + [{
                    "name": "自嘲接梗", "style": "复读群内暗号",
                    "statements": [{"text": "暗号原句"}],
                }],
            })
        await plugin._refresh_global_models()
        await plugin.handle_message(dict(
            sample_message(group_id=GROUP, text="需要些颜料"), message_id="no-match"
        ))
        await plugin.rewrite_style_response(
            response="行，那我这就去看看吧", session_id=SESSION, reply_message_id="no-match",
        )

        rewrite.assert_awaited_once()
        reference = rewrite.await_args.args[0]
        assert 0 < len(reference) <= 300
        assert reference.count("【随机场景表达参考 ") == 3
        assert all(style in reference for style in ordinary[1:]) and ordinary[0] not in reference
        assert not any(text in reference for text in ("原主题", "旧对话", "自嘲接梗", "暗号"))
        preview = (await plugin.get_config_widget_state())["reply_preview"].text
        assert "未命中，随机场景" in preview
        assert "原主题0（随机）" not in preview
        assert all(f"原主题{index}（随机）" in preview for index in (1, 2, 3))

    run(scenario())


def test_unmatched_reply_uses_summary_when_no_ordinary_scene_has_style(tmp_path: Path) -> None:
    plugin = make_env(tmp_path, inject={
        "reference_mode": "按场景", "global_models": ["sample"], "max_chars": 200,
    })
    rewrite = plugin._rewrite_with_llm = AsyncMock(return_value="[[KEEP_ORIGINAL]]")

    async def scenario() -> None:
        await plugin._store.save_model({
            "model_id": "sample",
            "style_summary": "说话直接，少用长句",
            "scenes": [
                {"name": "播报钟表", "description": "报时", "statements": [{"text": "整点啦"}]},
                {"name": "群内玩梗", "style": "复读群内暗号", "statements": [{"text": "旧梗原话"}]},
            ],
        })
        await plugin._refresh_global_models()
        await plugin.rewrite_style_response(response="这事我先了解一下", session_id=SESSION)

        rewrite.assert_awaited_once()
        reference = rewrite.await_args.args[0]
        assert "说话直接，少用长句" in reference
        assert not any(text in reference for text in ("随机场景", "整点啦", "旧梗原话", "群内暗号"))

    run(scenario())


def test_semantic_timeout_uses_keyword_fallback_and_limits_retries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT", 0.01)

    async def fake_embed(*, text=None, texts=None, **kwargs):
        if texts:
            return for_documents(texts)
        await asyncio.Event().wait()

    embed = AsyncMock(side_effect=fake_embed)
    index = SemanticIndex(tmp_path / "vectors.sqlite3", "embedding", embed)

    async def scenario() -> None:
        scenes = tuple(Scene.from_data(item) for item in (DINNER, GAMES))
        model = SearchModel({}, scenes)
        index.update(list(scenes))
        await index._worker
        for _ in range(2):
            matches, mode = await retrieval.retrieve_scenes([model], "要不要出去搓一顿", "我想先问问", index)
            assert matches == [[]] and "向量查询不可用" in mode
        assert embed.await_count == 2

    run(scenario())


def test_cached_models_refresh_after_replacement(tmp_path: Path) -> None:
    plugin = make_env(tmp_path)
    cache = SearchModelCache()

    async def scenario() -> None:
        await plugin._store.save_model({"model_id": "daily", "scenes": [DINNER]})
        old = cache.get(plugin._store, "daily")
        assert cache.get(plugin._store, "daily") is old
        await plugin._store.save_model({"model_id": "daily", "scenes": [GAMES]})
        new = cache.get(plugin._store, "daily")
        assert new.scenes[0].data["name"] == "排位组队"
        assert new is not old

    run(scenario())


def test_task_model_change_does_not_compare_incompatible_vectors(tmp_path: Path) -> None:
    embed = AsyncMock(side_effect=[
        for_documents([DINNER["name"], GAMES["name"]]),
        payload([1.0, 0.0], "new-embedding"),
        {"results": [payload([1.0, 0.0], "new-embedding"), payload([0.0, 1.0], "new-embedding")]},
    ])
    index = SemanticIndex(tmp_path / "vectors.sqlite3", "embedding", embed)

    async def scenario() -> None:
        model = SearchModel({}, (Scene.from_data(DINNER), Scene.from_data(GAMES)))
        index.update(list(model.scenes))
        await index._worker
        selection, _ = await retrieval.retrieve_scenes([model], "要不要出去搓一顿", "好啊", index)
        assert selection == [[]]
        await index._worker
        selection, _ = await retrieval.retrieve_scenes([model], "要不要出去搓一顿", "好啊", index)
        assert selection[0][0].scene.data["name"] == DINNER["name"]
        assert embed.await_count == 3

    run(scenario())


def test_switch_embedding_source_routes_index_and_query_to_the_selected_task(tmp_path: Path) -> None:
    settings = {"reference_mode": "按场景", "semantic_retrieval": True, "global_models": ["daily"]}
    plugin = make_env(tmp_path, inject=settings)

    async def select_task(task: str) -> None:
        config = {
            "plugin": {"enabled": True, "config_version": "0.20.6"},
            "inject": {**settings, "embedding_task": task},
        }
        plugin.set_plugin_config(config)
        await plugin.on_config_update("self", config, "0.20.6")
        await plugin._semantic_index._worker

    async def fake_embed(*, task_name: str, text=None, texts=None, **kwargs):
        model = "text-vectors" if task_name == "embedding" else "multimodal-vectors"
        if texts:
            values = [[1.0, 0.0] if doc.startswith(DINNER["name"]) else [0.0, 1.0] for doc in texts]
            return {"results": [payload(vector, model) for vector in values]}
        return payload([1.0, 0.0], model)

    embed = plugin.ctx.llm.embed = AsyncMock(side_effect=fake_embed)
    plugin._rewrite_with_llm = AsyncMock(return_value="[[KEEP_ORIGINAL]]")

    async def scenario() -> None:
        await plugin._store.save_model({"model_id": "daily", "scenes": [DINNER, GAMES]})
        await plugin._refresh_global_models()
        await plugin._semantic_index._worker
        assert embed.await_args.kwargs["task_name"] == "embedding"

        await select_task("image_embedding")
        assert embed.await_args.kwargs["task_name"] == "image_embedding"
        assert "multimodal-vectors" in (await plugin.get_config_widget_state())["active_models"].text

        incoming = "要不要出去搓一顿"
        await plugin.handle_message(dict(sample_message(group_id=GROUP, text=incoming), message_id="query"))
        await plugin.rewrite_style_response(
            response="可以，咱们找个地方吧", reply_message_id="query", session_id=SESSION,
        )
        assert embed.await_args.kwargs == {"text": incoming, "task_name": "image_embedding"}
        assert "线下聚餐" in plugin._rewrite_with_llm.await_args.args[0]

        await select_task("embedding")
        assert "text-vectors" in (await plugin.get_config_widget_state())["active_models"].text
        assert embed.await_count == 3

    run(scenario())
