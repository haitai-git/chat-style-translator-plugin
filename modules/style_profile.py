from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable


logger = logging.getLogger("chat_style_translator.style_profile")
_PROFILE_TEXT_BUDGET = 1800
_MODEL_FIELDS = ("model_id", "name", "status", "group_id", "group_only", "global_apply", "global_disabled")

_NON_CHAT_TEXT = re.compile(r"https?://|\[(?:图片|表情包?|视频|语音|image|video|record|face)(?:[^]]*)\]", re.I)
_CLAUSE_BREAK = re.compile(r"[，。！？；,.!?;\n]+")
_END_PARTICLES = frozenset("吧啊呢嘛啦呗嗷呀哇哦诶")
_CONTEXTUAL_CUES = r"互损|自嘲|反讽|反话|双关|群内梗|群内暗号|黑话|接梗|谐音梗"
_CONTEXTUAL = re.compile(_CONTEXTUAL_CUES)
_AVOID_CONTEXTUAL = re.compile(rf"(?:避免|不要|不宜|不适合|禁止|禁用|不用|别)[^，。；、]{{0,6}}(?:{_CONTEXTUAL_CUES})")


def contextual_style(text: str) -> bool:
    return bool(_CONTEXTUAL.search(_AVOID_CONTEXTUAL.sub("", text)))


def _records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        yield from (item for item in value if isinstance(item, dict))


def _ordinary_scenes(scenes: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    return (
        scene for scene in scenes
        if isinstance(scene, dict) and not contextual_style(
            " ".join(str(scene.get(key) or "") for key in ("name", "style", "note", "description"))
        )
    )


def observed_group_style(scenes: list[dict[str, Any]]) -> str:
    return observed_style(
        statement
        for scene in _ordinary_scenes(scenes)
        for statement in _records(scene.get("statements"))
    )


def observed_style(statements: Iterable[Dict[str, Any]]) -> str:
    texts = dict.fromkeys(
        text for statement in statements
        if (text := str(statement.get("text") or "").strip())
        and any(ch.isalnum() for ch in text)
        and not _NON_CHAT_TEXT.search(text)
        and not text.startswith("/")
    )
    if len(texts) < 3:
        return ""
    lengths = [len("".join(text.split())) for text in texts]
    observations = [f"样本长度中位数 {median(lengths):g} 字符"]
    if 2 * sum(text[-1] not in "。！？.!?…；;" for text in texts) > len(texts):
        observations.append("多数句子省略句末标点")
    if 2 * sum(len([part for part in _CLAUSE_BREAK.split(text) if part.strip()]) > 1 for text in texts) > len(texts):
        observations.append("常用标点隔开几段短语")
    tails = Counter(
        ending[-1] for text in texts
        if (ending := text.rstrip(" \t。！？.!?…；;，,")) and ending[-1] in _END_PARTICLES
    )
    particles = [particle for particle, count in tails.most_common(3) if count >= 2 and 4 * count >= len(texts)]
    if particles:
        observations.append("部分句子用「" + "、".join(particles) + "」收尾")
    return "，".join(observations)


def _profile_snapshots(model: dict[str, Any], scenes: list[dict[str, Any]]) -> list[str]:
    snapshots = model.get("style_snapshots")
    if isinstance(snapshots, list) and any(isinstance(value, str) and value.strip() for value in snapshots):
        descriptions = reversed(snapshots)
    else:
        descriptions = (model.get("overall_style"),)
    result = _bounded_descriptions(descriptions)
    if not result:
        result = _bounded_descriptions(scene.get("style") for scene in _ordinary_scenes(scenes))
    return result


def _bounded_descriptions(descriptions: Iterable[Any]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    remaining = _PROFILE_TEXT_BUDGET
    for value in descriptions:
        if not isinstance(value, str) or not (text := " ".join(value.split())) or text in seen:
            continue
        if selected and len(text) > remaining:
            continue
        selected.append(text[:remaining])
        seen.add(text)
        remaining -= len(selected[-1])
        if remaining <= 0:
            break
    return selected


def build_model_profile(model: dict[str, Any]) -> dict[str, Any]:
    scenes = list(_records(model.get("scenes")))
    profile = {field: model[field] for field in _MODEL_FIELDS if field in model}
    summary = " ".join(str(model.get("style_summary") or "").split())
    if summary:
        profile["style_summary"] = summary[:_PROFILE_TEXT_BUDGET]
    else:
        profile["style_snapshots"] = list(reversed(_profile_snapshots(model, scenes)))
    profile["observations"] = observed_group_style(scenes)
    profile["scene_count"] = len(scenes)
    profile["statement_count"] = sum(1 for scene in scenes for _ in _records(scene.get("statements")))
    return profile


def _member_styles(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    styles: dict[str, dict[str, Any]] = {}
    for member in _records(model.get("members")):
        if not (user_id := str(member.get("user_id") or "").strip()):
            continue
        style = " ".join(str(member.get("style") or "").split())[:180]
        traits = member.get("speaking_traits")
        traits = [" ".join(str(trait).split())[:60] for trait in traits[:4]] if isinstance(traits, list) else []
        styles.setdefault(user_id, {"user_id": user_id, "style": style, "speaking_traits": traits})
    return styles


class StyleProfileCache:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self, query: str, parameters: tuple[str, ...]) -> dict[str, Any] | None:
        if not self._path.is_file():
            return None
        try:
            with closing(sqlite3.connect(self._path, timeout=0.25)) as db:
                row = db.execute(query, parameters).fetchone()
            if row is not None:
                result = json.loads(row[0])
                return result if isinstance(result, dict) else None
        except (sqlite3.Error, ValueError):
            return None
        return None

    def get(self, key: str, revision: str) -> dict[str, Any] | None:
        return self._read("SELECT profile FROM models WHERE key = ? AND revision = ?", (key, revision))

    def get_member(self, key: str, revision: str, user_id: str) -> dict[str, Any] | None:
        return self._read(
            "SELECT members.style FROM members JOIN models USING (key) "
            "WHERE members.key = ? AND models.revision = ? AND members.user_id = ?",
            (key, revision, user_id),
        )

    def replace(
        self, key: str, revision: str, profile: dict[str, Any], model: dict[str, Any],
    ) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self._path, timeout=0.25)) as db, db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS models (key TEXT PRIMARY KEY, revision TEXT NOT NULL, profile TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS members (key TEXT, user_id TEXT, style TEXT NOT NULL, PRIMARY KEY (key, user_id))"
                )
                db.execute(
                    "INSERT OR REPLACE INTO models VALUES (?, ?, ?)",
                    (key, revision, json.dumps(profile, ensure_ascii=False)),
                )
                db.execute("DELETE FROM members WHERE key = ?", (key,))
                db.executemany(
                    "INSERT INTO members VALUES (?, ?, ?)",
                    ((key, user_id, json.dumps(style, ensure_ascii=False)) for user_id, style in _member_styles(model).items()),
                )
        except (OSError, sqlite3.Error):
            logger.warning("风格档案写入失败，下次读取时会重建", exc_info=True)

    def remove(self, key: str) -> None:
        if not self._path.is_file():
            return
        try:
            with closing(sqlite3.connect(self._path, timeout=0.25)) as db, db:
                db.execute("DELETE FROM models WHERE key = ?", (key,))
                db.execute("DELETE FROM members WHERE key = ?", (key,))
        except sqlite3.Error:
            logger.warning("清理风格档案失败：%s", self._path, exc_info=True)
