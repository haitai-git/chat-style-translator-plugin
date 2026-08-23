"""聊天风格转译插件全量测试（配置模型与 WebUI Schema）。"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from chat_style_translator_plugin.plugin import ChatStyleTranslatorPlugin, HELP_TEXT


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


class TestConfig:
    def test_config_toml_loads(self) -> None:
        with open(CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
        plugin = ChatStyleTranslatorPlugin()
        plugin.set_plugin_config(raw)
        cfg = plugin.config
        # enabled 为插件当前启用状态（用户可在 WebUI 随时切换），断言类型与版本即可
        assert isinstance(cfg.plugin.enabled, bool)
        assert cfg.plugin.config_version == "0.18.0"
        assert cfg.plugin.no_symbol_emotion is True
        assert cfg.plugin.auto_enable_models is True
        assert cfg.owner.qq == []
        assert cfg.inject.enabled is True
        assert cfg.inject.enable_style_matching is True
        # 模型任务名为用户可随时修改的配置值，这里只校验其类型合法性
        assert isinstance(cfg.inject.rewrite_model, str)
        assert cfg.inject.rewrite_timeout_seconds > 0
        assert 0 <= cfg.inject.rewrite_strength <= 100
        assert isinstance(cfg.analysis.model, str)
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
        assert isinstance(cfg.merge.max_total_statements, int) and cfg.merge.max_total_statements >= 0
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
            "plugin", "owner", "storage", "groups", "merge", "compact", "user_block",
            "collection", "analysis", "blacklist", "phrase_blacklist", "inject",
            "garbage_guard", "pre_rewrite_blacklist",
        }
        assert expected <= set(schema["sections"].keys())

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
        user_block = schema["sections"]["user_block"]["fields"]
        assert user_block["skip_self_messages"]["ui_type"] == "switch"
        assert user_block["self_qq"]["item_type"] == "string"
        phrase = schema["sections"]["phrase_blacklist"]["fields"]
        assert phrase["items"]["type"] == "array" and phrase["items"]["item_type"] == "string"
        assert schema["sections"]["groups"]["fields"]["items"]["item_type"] == "object"
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
        assert inject["rewrite_style"]["ui_type"] == "select"
        assert inject["rewrite_style"]["choices"] == [
            "无", "温柔", "自然", "活泼", "直爽", "软萌", "沉稳", "冷漠", "热情", "自定义",
        ]
        inject_hint = inject["rewrite_style"]["hint"]
        assert "各选项注释" in inject_hint
        for choice in inject["rewrite_style"]["choices"]:
            assert f"{choice} =" in inject_hint
        assert "小" in inject["rewrite_model"]["hint"] or "小" in inject["rewrite_model"]["description"]
        assert "rewrite_style_custom" in inject and inject["rewrite_style_custom"]["type"] == "string"

    def test_old_config_without_new_sections(self) -> None:
        """旧配置（没有新节）加载时自动补默认值，不崩溃。"""

        plugin = ChatStyleTranslatorPlugin()
        old: dict[str, Any] = {"plugin": {"enabled": True, "config_version": "0.6.0"}}
        plugin.set_plugin_config(old)
        cfg = plugin.config
        assert cfg.merge.ratio_a == 50 and cfg.merge.trigger is False
        assert cfg.merge.scene_merge_threshold == 0.8 and cfg.merge.snapshot_merge_threshold == 0.55
        assert cfg.compact.trigger is False and cfg.compact.model == ""
        assert cfg.compact.dedup_level == "标准" and cfg.compact.max_total_statements == 0
        assert cfg.compact.scene_merge_threshold == 0.8 and cfg.compact.snapshot_merge_threshold == 0.55
        assert cfg.inject.hook_timeout_seconds == 60.0
        assert cfg.inject.rewrite_timeout_seconds == 60.0
        assert cfg.inject.rewrite_style == "无"
        assert cfg.inject.rewrite_style_custom == ""
        assert cfg.phrase_blacklist.items == []
        assert cfg.user_block.skip_self_messages is True
        assert cfg.user_block.self_qq == []
        assert cfg.plugin.auto_enable_models is True  # 旧配置无此字段时默认开启
        assert cfg.garbage_guard.enabled is True
        assert cfg.garbage_guard.retry_on_detect is True
        assert cfg.garbage_guard.custom_patterns == []
        assert cfg.pre_rewrite_blacklist.enabled is False
        assert cfg.pre_rewrite_blacklist.items == []

    def test_help_text_complete(self) -> None:
        for keyword in (
            "/风格绑定群", "/风格生成", "/风格合并", "/黑名单添加",
            "字句黑名单", "跳过收集本人消息", "融合比例",
            "必填", "无需参数", "场景合并阈值", "模型精简",
            "微调前黑名单",
        ):
            assert keyword in HELP_TEXT
