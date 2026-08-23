"""聊天风格转译插件全量测试（存储层）。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from chat_style_translator_plugin.modules.storage import StyleStore


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestGroups:
    def test_bind_unbind(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.bind_group("111", source="config"))
        run(store.bind_group("111", source="config", group_name="群A"))
        run(store.bind_group("222"))
        assert store.is_bound("111") and store.is_bound("222")
        group = store.get_group("111")
        assert group["group_id"] == "111"
        assert group["source"] == "config" and group["group_name"] == "群A"
        assert set(store.list_groups().keys()) == {"111", "222"}
        run(store.append_message("222", {"text": "hi", "id": "1"}))
        assert run(store.unbind_group("222")) is True
        assert store.is_bound("222") is False
        assert store.messages_for("222") == []
        assert run(store.unbind_group("999")) is False

    def test_group_field(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.bind_group("111"))
        run(store.set_group_field("111", "active_models", ["m1"]))
        assert store.get_group("111")["active_models"] == ["m1"]
        run(store.set_group_field("999", "x", 1))  # 未绑定群忽略

    def test_corrupted_json_falls_back(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        (tmp_path / "groups.json").write_text("{broken", encoding="utf-8")
        assert store.list_groups() == {}
        assert store.get_group("111") is None


class TestMessages:
    def test_roundtrip(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        assert run(store.append_message("111", {"id": "1", "text": "a"})) == 1
        assert run(store.append_message("111", {"id": "2", "text": "b"})) == 2
        assert run(store.append_message("111", {"id": "3", "text": ""})) == 2  # 空文本不追加
        assert store.message_count("111") == 2
        assert store.messages_for("111")[0]["text"] == "a"
        assert store.messages_for("999") == []
        run(store.clear_messages("111"))
        assert store.message_count("111") == 0

    def test_batch_append_dedup(self, tmp_path: Path) -> None:
        """批量追加（历史回填用）：按消息 ID 去重、空文本跳过、不污染已有数据。"""
        store = StyleStore(tmp_path)
        run(store.append_message("111", {"id": "1", "text": "已有"}))
        batch = [
            {"id": "1", "text": "已有（重复）"},  # 与已有记录重复 -> 跳过
            {"id": "2", "text": "新消息"},
            {"id": "2", "text": "新消息（批内重复）"},  # 批内重复 -> 跳过
            {"id": "3", "text": "  "},  # 空文本 -> 跳过
            {"id": "4", "text": "最后一条"},
        ]
        assert run(store.append_messages("111", batch)) == 3
        assert store.message_count("111") == 3
        texts = [m["text"] for m in store.messages_for("111")]
        assert texts == ["已有", "新消息", "最后一条"]
        assert run(store.append_messages("111", [])) == 3  # 空批量不报错
        assert run(store.append_messages("999", [{"id": "x", "text": "hi"}])) == 1
        assert store.messages_for("999")[0]["text"] == "hi"

    def test_append_trims_oldest(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        for i in range(1, 6):
            run(store.append_message("111", {"id": f"{i}", "text": f"m{i}"}, max_stored=3))
        assert store.message_count("111") == 3
        assert [m["text"] for m in store.messages_for("111")] == ["m3", "m4", "m5"]  # 最旧的被删

    def test_batch_append_trims_oldest(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.append_message("111", {"id": "1", "text": "已有"}))
        batch = [{"id": f"{i}", "text": f"新{i}"} for i in range(2, 6)]
        assert run(store.append_messages("111", batch, max_stored=3)) == 3
        assert [m["text"] for m in store.messages_for("111")] == ["新3", "新4", "新5"]

    def test_max_stored_zero_disables_trim(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        for i in range(1, 6):
            run(store.append_message("111", {"id": f"{i}", "text": f"m{i}"}, max_stored=0))
        assert store.message_count("111") == 5

    def test_truncate_messages(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        for i in range(1, 8):
            run(store.append_message("111", {"id": f"{i}", "text": f"m{i}"}))
        assert run(store.truncate_messages("111", keep=3)) == 3
        assert [m["text"] for m in store.messages_for("111")] == ["m5", "m6", "m7"]
        assert run(store.truncate_messages("111", keep=10)) == 3  # 不足上限不改动
        assert store.message_count("111") == 3


class TestModels:
    def test_save_find_delete(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        model = {"name": "测试模型", "overall_style": "s", "members": [], "scenes": []}
        run(store.save_model(model))
        model_id = model["model_id"]
        assert model_id.startswith("m_")
        assert store.get_model(model_id)["name"] == "测试模型"
        assert store.find_model("测试模型")["model_id"] == model_id
        assert store.find_model("测试")["model_id"] == model_id  # 前缀匹配
        assert store.find_model(model_id)["model_id"] == model_id
        assert store.find_model("不存在") is None
        run(store.bind_group("111"))
        run(store.set_group_field("111", "active_models", [model_id]))
        assert run(store.delete_model(model_id)) is True
        assert store.get_model(model_id) is None
        assert store.get_group("111")["active_models"] == []
        assert run(store.delete_model(model_id)) is False

    def test_dir_scan_and_corruption(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        run(store.save_model({"name": "A", "scenes": []}))
        run(store.save_model({"name": "B", "scenes": []}))
        time.sleep(0.01)
        run(store.save_model({"name": "C", "scenes": []}))
        assert len(store.list_model_ids()) == 3
        assert store.list_model_ids()[0] == store.list_models()[0]["model_id"]
        (store.model_dir / "broken.json").write_text("{not json", encoding="utf-8")
        ids = store.list_model_ids()
        assert all(not Path(store.model_dir / f"{mid}.json").name == "broken.json" for mid in ids)
        assert len(store.list_models()) == 3


class TestBlacklist:
    def test_ops(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        item = run(store.add_blacklist("骂人", apply_all=True))
        assert store.list_blacklist()[0]["id"] == item["id"]
        assert run(store.add_blacklist("专属", model_id="m1", model_name="M1", apply_all=False))
        assert len(store.list_blacklist()) == 2
        assert run(store.remove_blacklist(item["id"])) is True
        assert len(store.list_blacklist()) == 1
        assert run(store.remove_blacklist("nope")) is False
        run(store.clear_blacklist())
        assert store.list_blacklist() == []

    def test_sync_config_idempotent(self, tmp_path: Path) -> None:
        store = StyleStore(tmp_path)
        config_items = [
            {"text": "脏话", "meaning": "骂人", "model_id": "", "model_name": "", "apply_all": True},
            {"text": "专属", "meaning": "", "model_id": "m1", "model_name": "M1", "apply_all": False},
        ]
        run(store.sync_blacklist_config(config_items))
        assert len(store.list_blacklist()) == 2
        run(store.sync_blacklist_config(config_items))
        assert len(store.list_blacklist()) == 2  # 幂等
        run(store.add_blacklist("命令条目", apply_all=True))
        run(store.sync_blacklist_config(config_items[:1]))
        items = store.list_blacklist()
        assert len(items) == 2
        sources = {b["source"] for b in items}
        assert "command" in sources and "config" in sources
        ids = [b["id"] for b in items if b["source"] == "config"]
        assert ids == ["blc_" + ids[0][4:] if len(ids) == 1 else ids[0]]  # 稳定 ID
