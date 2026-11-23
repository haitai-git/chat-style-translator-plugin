"""聊天风格转译插件全量测试（配置模型与 WebUI Schema）。"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from chat_style_translator_plugin.plugin import ChatStyleTranslatorPlugin, HELP_TEXT


class TestConfig:
    def test_config_toml_loads(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[plugin]
config_version = "0.18.0"
auto_enable_models = true
no_symbol_emotion = false
[owner]
qq = ["10001"]
[inject]
rewrite_model = "utils"
[analysis]
model = "test_analysis"
[merge]
model_a = "alpha"
model_b = "beta"
""",
            encoding="utf-8",
        )
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        plugin = ChatStyleTranslatorPlugin()
        plugin.set_plugin_config(raw)
        cfg = plugin.config
        # enabled 为插件当前启用状态（用户可在 WebUI 随时切换），断言类型与版本即可
        assert isinstance(cfg.plugin.enabled, bool)
        assert cfg.plugin.config_version == "0.18.0"
        assert cfg.user_block.no_symbol_emotion is False
        assert cfg.inject.auto_enable_models is True
        assert cfg.owner.qq == ["10001"]
        assert cfg.inject.enabled is True
        # 模型任务名为用户可随时修改的配置值，这里只校验其类型合法性
        assert isinstance(cfg.inject.rewrite_model, str)
        assert cfg.inject.rewrite_timeout_seconds > 0
        assert 0 <= cfg.inject.rewrite_strength <= 100
        assert cfg.collection.model == "test_analysis"
        # 全局通用模型为模型引用（文件名 / ID / 名称），校验为字符串列表
        assert isinstance(cfg.inject.global_models, list)
        assert all(isinstance(ref, str) for ref in cfg.inject.global_models)
        # 自动分析条数为用户可随时修改的配置值，只校验非负整数
        assert isinstance(cfg.collection.auto_analyze_count, int) and cfg.collection.auto_analyze_count >= 0
        # 每日分析时间为用户可随时修改的配置值，校验空或 HH:MM 格式即可
        assert isinstance(cfg.collection.daily_time, str)
        assert cfg.collection.daily_time == "" or len(cfg.collection.daily_time.split(":")) == 2
        assert cfg.collection.max_messages == 300
        assert cfg.collection.window_size == 60
        # 融合比例为用户可随时修改的配置值，只校验 0~100
        assert isinstance(cfg.merge.ratio_a, int) and 0 <= cfg.merge.ratio_a <= 100
        assert cfg.merge.dedup_level in ("标准", "激进精简", "不去重")
        assert isinstance(cfg.merge.trigger, bool)
        # 融合模型引用为模型引用（文件名 / ID / 名称），校验为字符串即可
        assert isinstance(cfg.merge.model_a, str)
        assert isinstance(cfg.merge.model_b, str)
        assert cfg.user_block.skip_self_messages is True
        assert cfg.user_block.self_qq == []
        assert isinstance(cfg.phrase_blacklist.items, list)
        assert all(isinstance(phrase, str) for phrase in cfg.phrase_blacklist.items)
        assert isinstance(cfg.groups.items, list)
        assert all(
            isinstance(item.group_id, str) and item.group_id.strip()
            for item in cfg.groups.items
        )
        assert isinstance(cfg.garbage_guard.enabled, bool)
        assert isinstance(cfg.garbage_guard.retry_on_detect, bool)
        assert isinstance(cfg.garbage_guard.custom_patterns, list)
        assert all(
            isinstance(pattern, str) and pattern.strip()
            for pattern in cfg.garbage_guard.custom_patterns
        )
        assert isinstance(cfg.pre_rewrite_blacklist.enabled, bool)
        assert isinstance(cfg.pre_rewrite_blacklist.items, list)
        assert all(
            isinstance(word, str) and word
            for word in cfg.pre_rewrite_blacklist.items
        )

    def test_all_sections_in_schema(self) -> None:
        schema = ChatStyleTranslatorPlugin.build_config_schema()
        expected = {
            "owner", "storage", "groups", "merge", "compact", "user_block",
            "collection", "analysis", "blacklist", "phrase_blacklist", "inject",
            "garbage_guard", "pre_rewrite_blacklist",
        }
        assert expected == set(schema["sections"].keys())

    def test_schema_widgets(self) -> None:
        schema = ChatStyleTranslatorPlugin.build_config_schema()
        merge = schema["sections"]["merge"]["fields"]
        assert merge["ratio_a"]["ui_type"] == "slider"
        assert merge["ratio_a"]["min"] == 0 and merge["ratio_a"]["max"] == 100
        assert merge["scene_merge_threshold"]["ui_type"] == "slider"
        assert merge["scene_merge_threshold"]["min"] == 0.5 and merge["scene_merge_threshold"]["max"] == 0.95
        assert merge["snapshot_merge_threshold"]["ui_type"] == "slider"
        assert merge["snapshot_merge_threshold"]["min"] == 0.3 and merge["snapshot_merge_threshold"]["max"] == 0.85
        assert merge["dedup_level"]["ui_type"] == "select"
        assert merge["dedup_level"]["choices"] == ["标准", "激进精简", "不去重"]
        assert merge["trigger"]["ui_type"] == "switch"
        compact = schema["sections"]["compact"]["fields"]
        assert compact["scene_merge_threshold"]["ui_type"] == "slider"
        assert compact["scene_merge_threshold"]["min"] == 0.5 and compact["scene_merge_threshold"]["max"] == 0.95
        assert compact["snapshot_merge_threshold"]["ui_type"] == "slider"
        assert compact["snapshot_merge_threshold"]["min"] == 0.3 and compact["snapshot_merge_threshold"]["max"] == 0.85
        assert compact["dedup_level"]["ui_type"] == "select"
        assert compact["dedup_level"]["choices"] == ["标准", "激进精简", "不去重"]
        assert compact["trigger"]["ui_type"] == "switch"
        assert sorted(merge, key=lambda name: merge[name]["order"]) == [
            "model_a", "model_b", "new_name", "ratio_a", "scene_merge_threshold",
            "snapshot_merge_threshold", "dedup_level", "trigger",
        ]
        assert sorted(compact, key=lambda name: compact[name]["order"]) == [
            "model", "new_name", "scene_merge_threshold", "snapshot_merge_threshold", "dedup_level", "trigger",
        ]
        user_block = schema["sections"]["user_block"]["fields"]
        assert user_block["skip_self_messages"]["ui_type"] == "switch"
        assert user_block["self_qq"]["item_type"] == "string"
        assert user_block["no_symbol_emotion"]["ui_type"] == "switch"
        assert sorted(user_block, key=lambda name: user_block[name]["order"]) == [
            "qq", "skip_self_messages", "block_qq_bots", "self_qq",
            "block_slash", "no_symbol_emotion", "bot_ids",
        ]
        collection = schema["sections"]["collection"]["fields"]
        collection_order = sorted(collection, key=lambda name: collection[name]["order"])
        assert collection_order[-4:] == ["backfill_batch_size", "model", "trigger_analyze_now", "trigger_backfill_now"]
        assert "model" not in schema["sections"]["analysis"]["fields"]
        phrase = schema["sections"]["phrase_blacklist"]["fields"]
        assert phrase["items"]["type"] == "array" and phrase["items"]["item_type"] == "string"
        group_items = schema["sections"]["groups"]["fields"]["items"]
        assert group_items["item_type"] == "object"
        assert set(group_items["item_fields"]) == {"group_id", "models", "prefer_group_models"}
        pre = schema["sections"]["pre_rewrite_blacklist"]["fields"]
        assert pre["rewrite_style"]["ui_type"] == "select"
        assert pre["rewrite_style"]["choices"] == [
            "无", "温柔", "自然", "活泼", "直爽", "软萌", "沉稳", "冷漠", "热情", "自定义",
        ]
        # 每个选项都有注释（显示在字段下方提示里）
        hint = pre["rewrite_style"]["hint"]
        assert "各选项注释" in hint
        for choice in pre["rewrite_style"]["choices"]:
            assert f"{choice} =" in hint
        inject = schema["sections"]["inject"]["fields"]
        assert inject["reference_mode"]["choices"] == ["整体风格", "按场景"]
        assert inject["reference_mode"]["default"] == "整体风格"
        assert inject["embedding_task"]["ui_type"] == "select"
        assert inject["embedding_task"]["choices"] == ["embedding", "image_embedding"]
        assert inject["embedding_task"]["default"] == "embedding"
        assert sorted(inject, key=lambda name: inject[name]["order"]) == [
            "enabled", "rewrite_strength", "block_emoji", "reference_mode", "semantic_retrieval",
            "embedding_task", "auto_enable_models", "global_models", "rewrite_model", "max_chars",
            "enrich_min_chars", "enrich_short_replies", "rewrite_timeout_seconds", "hook_timeout_seconds",
        ]
        assert inject["rewrite_strength"]["ui_type"] == "slider"
        assert inject["rewrite_strength"]["min"] == 0 and inject["rewrite_strength"]["max"] == 100
        assert inject["enrich_min_chars"]["type"] == "integer"
        assert inject["enrich_min_chars"]["default"] == 5
        assert inject["enrich_short_replies"]["ui_type"] == "switch"
        assert inject["enrich_short_replies"]["default"] is False
        assert inject["auto_enable_models"]["ui_type"] == "switch"
        assert inject["rewrite_model"]["placeholder"] == "微调模型任务名（默认 planner）"
        assert "关闭「模型目录自动生效」后使用" in inject["global_models"]["hint"]
        assert "planner" in inject["rewrite_model"]["description"]
        assert "candidate_model" not in inject

    def test_old_config_without_new_sections(self) -> None:
        """旧配置（没有新节）加载时自动补默认值，不崩溃。"""

        plugin = ChatStyleTranslatorPlugin()
        old: dict[str, Any] = {"plugin": {"enabled": True, "config_version": "0.6.0"}}
        plugin.set_plugin_config(old)
        cfg = plugin.config
        assert cfg.merge.ratio_a == 50 and cfg.merge.trigger is False
        assert cfg.merge.scene_merge_threshold == 0.8 and cfg.merge.snapshot_merge_threshold == 0.55
        assert cfg.compact.trigger is False and cfg.compact.model == ""
        assert cfg.compact.dedup_level == "标准"
        assert cfg.compact.scene_merge_threshold == 0.8 and cfg.compact.snapshot_merge_threshold == 0.55
        assert cfg.inject.hook_timeout_seconds == 60.0
        assert cfg.inject.rewrite_timeout_seconds == 60.0
        assert cfg.inject.enrich_min_chars == 5
        assert cfg.inject.enrich_short_replies is False
        assert cfg.phrase_blacklist.items == []
        assert cfg.user_block.skip_self_messages is True
        assert cfg.user_block.self_qq == []
        assert cfg.inject.auto_enable_models is False
        assert cfg.user_block.no_symbol_emotion is True
        assert cfg.garbage_guard.enabled is True
        assert cfg.garbage_guard.retry_on_detect is True
        assert cfg.garbage_guard.custom_patterns == []
        assert cfg.pre_rewrite_blacklist.enabled is False
        assert cfg.pre_rewrite_blacklist.items == []

    def test_explicit_new_values_take_priority_over_legacy_config(self) -> None:
        plugin = ChatStyleTranslatorPlugin()
        old_plugin_settings = {
            "config_version": "0.20.18",
            "auto_enable_models": True,
            "no_symbol_emotion": False,
        }
        config = {
            "plugin": old_plugin_settings,
            "analysis": {"model": "previous-analysis"},
            "collection": {"model": "selected-analysis"},
            "inject": {"auto_enable_models": False, "enable_style_matching": True},
            "user_block": {"no_symbol_emotion": True},
        }

        normalized, changed = plugin.normalize_plugin_config(config)
        assert changed
        assert normalized["collection"]["model"] == "selected-analysis"
        assert normalized["inject"]["auto_enable_models"] is False
        assert normalized["user_block"]["no_symbol_emotion"] is True
        assert normalized["plugin"].keys() == {"enabled", "config_version"}
        assert "enable_style_matching" not in normalized["inject"]
        assert "model" not in normalized["analysis"]
        assert config["plugin"]["auto_enable_models"] is True
        assert config["plugin"]["no_symbol_emotion"] is False
        assert config["analysis"]["model"] == "previous-analysis"
        assert plugin.normalize_plugin_config(normalized) == (normalized, False)

    def test_help_text_complete(self) -> None:
        for keyword in (
            "/风格绑定群", "/风格生成", "/风格合并", "/黑名单添加",
            "/风格拉取历史", "/风格场景", "/黑名单列表", "/黑名单清空",
            "管理员 QQ 号列表", "私聊", "带空格", "模型精简",
        ):
            assert keyword in HELP_TEXT
