"""聊天风格转译插件：JSON 持久化存储。

负责群绑定信息、收集到的群消息、风格模型、模型索引与黑名单的读写。
所有写操作都是原子写（先写临时文件再替换），避免中断导致文件损坏。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("chat_style_translator.storage")


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    """把字典以 UTF-8 原子写入指定路径。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    """读取 JSON 文件，文件不存在或损坏时返回默认值。"""

    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("配置文件损坏，使用默认值: %s", path)
        return default


class StyleStore:
    """插件全部持久化数据的中转层。"""

    def __init__(self, data_dir: Path, model_dir: Optional[Path] = None) -> None:
        """初始化存储。

        Args:
            data_dir: 插件持久化数据目录（self.ctx.paths.data_dir）。
            model_dir: 模型文件保存目录；留空时使用 data_dir/models。
        """
        self._data_dir = data_dir
        self._model_dir = Path(model_dir) if model_dir is not None else data_dir / "models"
        self._lock = asyncio.Lock()
        self._groups_path = data_dir / "groups.json"
        self._messages_dir = data_dir / "messages"
        self._blacklist_path = data_dir / "blacklist.json"

    @property
    def model_dir(self) -> Path:
        """返回模型文件保存目录。"""

        return self._model_dir

    # ------------------------------------------------------------------
    # 群绑定
    # ------------------------------------------------------------------

    def list_groups(self) -> Dict[str, Dict[str, Any]]:
        """返回全部绑定群：{group_id: group_info}。"""

        return _read_json(self._groups_path, {})

    def get_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        """按群号返回群信息，未绑定时返回 None。"""

        return self.list_groups().get(str(group_id).strip())

    def is_bound(self, group_id: str) -> bool:
        """该群是否已绑定。"""

        return self.get_group(group_id) is not None

    async def bind_group(
        self,
        group_id: str,
        platform: str = "qq",
        group_name: str = "",
        source: str = "command",
    ) -> Dict[str, Any]:
        """绑定一个群，返回群信息。重复绑定只更新元信息。

        Args:
            group_id: QQ 群号。
            platform: 平台名。
            group_name: 群名称（可选）。
            source: 绑定来源，``command`` 表示 /风格绑定群 命令，``config`` 表示插件配置。
        """

        group_id = str(group_id).strip()
        async with self._lock:
            groups = _read_json(self._groups_path, {})
            group = groups.get(group_id)
            if group is None:
                group = {
                    "group_id": group_id,
                    "platform": platform or "qq",
                    "group_name": group_name,
                    "source": source or "command",
                    "bind_time": time.time(),
                    "active_models": [],
                    "last_analyzed_at": 0.0,
                }
                groups[group_id] = group
            else:
                group["platform"] = platform or group.get("platform", "qq")
                group["source"] = source or group.get("source", "command")
                if group_name:
                    group["group_name"] = group_name
            _atomic_write(self._groups_path, groups)
            return group

    async def unbind_group(self, group_id: str) -> bool:
        """解绑群，同时删除收集的消息。返回是否真的解绑了。"""

        group_id = str(group_id).strip()
        async with self._lock:
            groups = _read_json(self._groups_path, {})
            if group_id not in groups:
                return False
            del groups[group_id]
            _atomic_write(self._groups_path, groups)
        messages_path = self._messages_dir / f"{group_id}.json"
        if messages_path.is_file():
            messages_path.unlink(missing_ok=True)
        return True

    async def set_group_field(self, group_id: str, key: str, value: Any) -> None:
        """更新群信息中的单个字段。"""

        group_id = str(group_id).strip()
        async with self._lock:
            groups = _read_json(self._groups_path, {})
            group = groups.get(group_id)
            if group is None:
                return
            group[key] = value
            _atomic_write(self._groups_path, groups)

    # ------------------------------------------------------------------
    # 消息收集
    # ------------------------------------------------------------------

    def _messages_path(self, group_id: str) -> Path:
        return self._messages_dir / f"{str(group_id).strip()}.json"

    def messages_for(self, group_id: str) -> List[Dict[str, Any]]:
        """返回指定群已收集的消息列表（按收集顺序）。"""

        data = _read_json(self._messages_path(group_id), {})
        items = data.get("items", [])
        return items if isinstance(items, list) else []

    async def append_messages(self, group_id: str, messages: List[Dict[str, Any]], max_stored: int = 0) -> int:
        """批量追加收集消息（历史回填用），按消息 ID 去重，返回该群当前消息总数。

        Args:
            group_id: QQ 群号。
            messages: 收集记录列表；无有效文本的记录会被跳过，ID 与已有记录重复的会去重。
            max_stored: 存储上限；>0 时超出部分自动删除最旧消息，只保留最近 max_stored 条。
        """

        group_id = str(group_id).strip()
        valid = [m for m in messages if str(m.get("text") or "").strip() and isinstance(m, dict)]
        if not valid:
            return self.message_count(group_id)
        async with self._lock:
            data = _read_json(self._messages_path(group_id), {})
            items = data.get("items", [])
            if not isinstance(items, list):
                items = []
            seen = {str(item.get("id") or "") for item in items}
            for message in valid:
                message_id = str(message.get("id") or "").strip()
                if message_id:
                    if message_id in seen:
                        continue
                    seen.add(message_id)
                items.append(message)
            if max_stored > 0 and len(items) > max_stored:
                items = items[-max_stored:]
            data["items"] = items
            data["count"] = len(items)
            _atomic_write(self._messages_path(group_id), data)
            return len(items)

    async def append_message(self, group_id: str, message: Dict[str, Any], max_stored: int = 0) -> int:
        """追加一条收集消息，返回该群当前消息总数。

        max_stored > 0 时超出部分自动删除最旧消息，只保留最近 max_stored 条，
        防止消息文件无限增长。
        """

        group_id = str(group_id).strip()
        if not message.get("text"):
            return self.message_count(group_id)
        async with self._lock:
            data = _read_json(self._messages_path(group_id), {})
            items = data.get("items", [])
            if not isinstance(items, list):
                items = []
            items.append(message)
            if max_stored > 0 and len(items) > max_stored:
                items = items[-max_stored:]
            data["items"] = items
            data["count"] = len(items)
            _atomic_write(self._messages_path(group_id), data)
            return len(items)

    def message_count(self, group_id: str) -> int:
        """返回指定群已收集的消息条数。"""

        data = _read_json(self._messages_path(group_id), {})
        count = data.get("count")
        if isinstance(count, int):
            return count
        items = data.get("items", [])
        return len(items) if isinstance(items, list) else 0

    async def truncate_messages(self, group_id: str, keep: int) -> int:
        """把该群收集的消息裁剪到最多 keep 条（保留最近 keep 条），返回裁剪后条数。

        消息数不超过 keep 时不改动文件；keep <= 0 表示不裁剪。
        """

        group_id = str(group_id).strip()
        if keep <= 0:
            return self.message_count(group_id)
        async with self._lock:
            data = _read_json(self._messages_path(group_id), {})
            items = data.get("items", [])
            if not isinstance(items, list):
                items = []
            if len(items) <= keep:
                return len(items)
            items = items[-keep:]
            data["items"] = items
            data["count"] = len(items)
            _atomic_write(self._messages_path(group_id), data)
            return len(items)

    async def clear_messages(self, group_id: str) -> None:
        """清空指定群已收集的消息。"""

        group_id = str(group_id).strip()
        async with self._lock:
            _atomic_write(self._messages_path(group_id), {"items": [], "count": 0})

    # ------------------------------------------------------------------
    # 风格模型
    # ------------------------------------------------------------------

    def _model_path(self, model_id: str) -> Path:
        return self._model_dir / f"{model_id}.json"

    def list_model_ids(self) -> List[str]:
        """返回模型目录中的全部模型 ID。

        直接扫描模型保存目录，用户手动新增、修改、删除模型文件都会自动生效；
        内容损坏（JSON 解析失败）的文件会被跳过。按修改时间倒序返回。
        """

        if not self._model_dir.is_dir():
            return []
        entries = []
        for path in self._model_dir.glob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
                entries.append((path, path.stat().st_mtime))
            except (json.JSONDecodeError, OSError):
                continue
        entries.sort(key=lambda item: item[1], reverse=True)
        return [path.stem for path, _ in entries]

    def list_models(self) -> List[Dict[str, Any]]:
        """返回全部模型的摘要信息。"""

        models = []
        for model_id in self.list_model_ids():
            model = self.get_model(model_id)
            if model is not None:
                models.append(model)
        return models

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """按模型文件名、ID 或名称读取模型，不存在时返回 None。"""

        model_id = str(model_id).strip()
        path = self._model_path(model_id)
        if path.is_file():
            return _read_json(path, {})
        for model_file_id in self.list_model_ids():
            model = self.get_model(model_file_id)
            if model is None:
                continue
            if model.get("model_id") == model_id or model.get("name") == model_id:
                return model
        return None

    def find_model_file(self, name_or_id: str) -> Optional[str]:
        """按模型文件名、ID 或名称（支持前缀）返回模型文件名（不带 .json）。

        返回的是模型保存目录中的文件名，可直接用于 get_model / 挂载引用。
        """

        needle = str(name_or_id or "").strip()
        if not needle:
            return None
        model_ids = self.list_model_ids()
        if needle in model_ids:
            return needle
        for model_file_id in model_ids:
            model = self.get_model(model_file_id)
            if model is None:
                continue
            if model.get("model_id") == needle or model.get("name") == needle:
                return model_file_id
        for model_file_id in model_ids:
            model = self.get_model(model_file_id)
            if model is None:
                continue
            if str(model.get("name", "")).startswith(needle):
                return model_file_id
        return None

    def find_model(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """按模型名（支持前缀）、ID 或模型文件名查找模型。"""

        model_file_id = self.find_model_file(name_or_id)
        if model_file_id is None:
            return None
        return self.get_model(model_file_id)

    async def save_model(self, model: Dict[str, Any]) -> None:
        """保存模型（写入模型保存目录，可直接手动编辑该文件）。

        更新已有模型（如 /风格启用、/风格重命名）时：按内部 model_id 精确定位
        原文件并更新，保持文件名不变；新建模型（分析/融合生成的新 model_id）
        时：始终写入 model_id.json 新文件，绝不因名称相同而覆盖其他模型文件。
        """

        model_id = str(model.get("model_id") or "").strip()
        if not model_id:
            model_id = self.new_model_id()
            model["model_id"] = model_id
        model["updated_at"] = time.time()
        target_file = None
        for file_id in self.list_model_ids():
            existing = self.get_model(file_id)
            if existing is not None and str(existing.get("model_id") or "") == model_id:
                target_file = file_id
                break
        path = self._model_path(target_file) if target_file else self._model_path(model_id)
        async with self._lock:
            _atomic_write(path, model)

    async def delete_model(self, model_id: str) -> bool:
        """删除模型（按文件名 / ID / 名称匹配），并从所有绑定群的挂载列表移除。

        返回是否删除成功。
        """

        model_id = str(model_id).strip()
        target_file = self.find_model_file(model_id)
        path = self._model_path(target_file) if target_file else self._model_path(model_id)
        async with self._lock:
            if not path.is_file():
                return False
            path.unlink(missing_ok=True)
        for group in self.list_groups().values():
            active = group.get("active_models", [])
            if isinstance(active, list) and target_file in active:
                active.remove(target_file)
                await self.set_group_field(str(group.get("group_id", "")), "active_models", active)
        return True

    @staticmethod
    def new_model_id() -> str:
        """生成新的模型 ID。"""

        return f"m_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # ------------------------------------------------------------------
    # 黑名单
    # ------------------------------------------------------------------

    def list_blacklist(self) -> List[Dict[str, Any]]:
        """返回黑名单条目列表。"""

        data = _read_json(self._blacklist_path, {"items": []})
        items = data.get("items", [])
        return items if isinstance(items, list) else []

    @staticmethod
    def _new_blacklist_id() -> str:
        """生成命令方式添加的黑名单条目 ID。"""

        return f"bl_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

    async def add_blacklist(
        self,
        text: str,
        meaning: str = "",
        model_id: str = "",
        model_name: str = "",
        apply_all: bool = False,
    ) -> Dict[str, Any]:
        """添加一条黑名单条目，返回新条目。

        Args:
            text: 拉黑的语句（原话或意思描述）。
            meaning: 语句含义描述（可选，用于语义相似度匹配）。
            model_id: 限定模型 ID；留空表示所有模型。
            model_name: 限定的模型名（展示用）。
            apply_all: 通用拉黑开关，为 True 时按相似度对所有模型生效。
        """

        item = {
            "id": self._new_blacklist_id(),
            "text": str(text or "").strip(),
            "meaning": str(meaning or "").strip(),
            "model_id": str(model_id or "").strip(),
            "model_name": str(model_name or "").strip(),
            "apply_all": bool(apply_all),
            "source": "command",
            "added_at": time.time(),
        }
        async with self._lock:
            data = _read_json(self._blacklist_path, {"items": []})
            items = data.get("items", [])
            if not isinstance(items, list):
                items = []
            items.append(item)
            data["items"] = items
            _atomic_write(self._blacklist_path, data)
        return item

    async def sync_blacklist_config(self, config_items: List[Dict[str, Any]]) -> None:
        """把插件配置里的黑名单条目同步进存储。

        先移除旧的全配置来源条目，再按配置写入（条目 ID 由内容哈希生成，重复同步不会堆积）。
        """

        async with self._lock:
            data = _read_json(self._blacklist_path, {"items": []})
            items = data.get("items", [])
            if not isinstance(items, list):
                items = []
            items = [item for item in items if item.get("source") != "config"]
            for config_item in config_items:
                text = str(config_item.get("text") or "").strip()
                if not text:
                    continue
                identity = "|".join(
                    [
                        text,
                        str(config_item.get("meaning") or "").strip(),
                        str(config_item.get("model_id") or "").strip(),
                        "1" if config_item.get("apply_all") else "0",
                    ]
                )
                item_id = "blc_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
                items.append(
                    {
                        "id": item_id,
                        "text": text,
                        "meaning": str(config_item.get("meaning") or "").strip(),
                        "model_id": str(config_item.get("model_id") or "").strip(),
                        "model_name": str(config_item.get("model_name") or "").strip(),
                        "apply_all": bool(config_item.get("apply_all")),
                        "source": "config",
                        "added_at": time.time(),
                    }
                )
            data["items"] = items
            _atomic_write(self._blacklist_path, data)

    async def remove_blacklist(self, item_id: str) -> bool:
        """按 ID 删除黑名单条目。"""

        item_id = str(item_id or "").strip()
        async with self._lock:
            data = _read_json(self._blacklist_path, {"items": []})
            items = data.get("items", [])
            if not isinstance(items, list):
                return False
            remaining = [item for item in items if str(item.get("id", "")) != item_id]
            if len(remaining) == len(items):
                return False
            data["items"] = remaining
            _atomic_write(self._blacklist_path, data)
        return True

    async def clear_blacklist(self) -> None:
        """清空黑名单。"""

        async with self._lock:
            _atomic_write(self._blacklist_path, {"items": []})
