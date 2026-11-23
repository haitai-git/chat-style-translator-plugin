from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import sqlite3
import struct
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from .storage import StyleStore
from .style_profile import contextual_style, observed_group_style

logger = logging.getLogger("chat_style_translator.retrieval")
_MIN_SIMILARITY = 0.72
_MIN_MARGIN = 0.045
_QUERY_TIMEOUT = 1.5
_RETRY_SECONDS = 60.0
_COMMON_TERMS = frozenset(
    "哈哈 呵呵 这个 那个 什么 一下 一个 有点 觉得 知道 怎么 这么 那么 这样 可以 时候 "
    "不是 还是 没有 我们 你们 他们 一点 真的 现在 但是 可能".split()
)


def scene_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", text.casefold()):
        if "\u3400" <= token[0] <= "\u9fff":
            terms.update(token[index:index + 2] for index in range(len(token) - 1))
        elif len(token) > 1:
            terms.add(token)
    return terms


def _compact_key(text: str) -> str:
    return "".join(char for char in text.casefold() if char.isalnum())


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Scene:
    data: dict[str, Any]
    name_key: str
    topic: set[str]
    example_postings: dict[str, list[int]]
    prior_terms: tuple[set[str], ...]
    contextual: bool
    document: str
    fingerprint: str

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Scene:
        name, description, note, style = (
            str(data.get(field) or "").strip() for field in ("name", "description", "note", "style")
        )
        statements = [item for item in data.get("statements") or [] if isinstance(item, dict)]
        example_postings: dict[str, list[int]] = defaultdict(list)
        meanings = {}
        contexts = []
        for index, item in enumerate(statements):
            meaning = str(item.get("meaning") or "").strip()
            for term in scene_terms(f"{item.get('text') or ''} {meaning}"):
                example_postings[term].append(index)
            if meaning and len(meanings) < 6:
                meanings.setdefault(meaning[:80], None)
            context = item.get("context")
            if isinstance(context, str) and (terms := scene_terms(context)):
                contexts.append(terms)
        document = "\n".join(filter(None, (name[:80], description[:220], note[:100], *meanings)))
        return cls(
            data, _compact_key(name) if name != "未命名场景" else "",
            scene_terms(f"{name} {description} {note}"), dict(example_postings), tuple(contexts),
            contextual_style(f"{name} {style} {note} {description}"), document, _digest(document),
        )


@dataclass(frozen=True)
class SearchModel:
    data: dict[str, Any]
    scenes: tuple[Scene, ...]

    @cached_property
    def group_observations(self) -> str:
        if "observations" in self.data:
            return self.data["observations"]
        return observed_group_style(self.data.get("scenes") or [])


class SearchModelCache:
    def __init__(self) -> None:
        self._models: OrderedDict[tuple[Path, str, bool], tuple[tuple[int, int, int], SearchModel]] = OrderedDict()

    def get(self, store: StyleStore, file_id: str, *, index_scenes: bool = True) -> SearchModel | None:
        revision = store.model_revision(file_id)
        if revision is None:
            file_id = store.find_model_file(file_id) or file_id
            revision = store.model_revision(file_id)
        key = (store.model_dir, file_id, index_scenes)
        cached = self._models.get(key)
        if revision is not None and cached is not None and cached[0] == revision:
            self._models.move_to_end(key)
            return cached[1]
        self._models.pop(key, None)
        data = store.get_model(file_id) if index_scenes else store.get_model_profile(file_id)
        if not data:
            return None
        scenes = tuple(
            Scene.from_data(scene) for scene in data.get("scenes") or [] if isinstance(scene, dict)
        ) if index_scenes else ()
        model = SearchModel(data, scenes)
        if revision is not None:
            self._models[key] = (revision, model)
            if len(self._models) > 32:
                self._models.popitem(last=False)
        return model


@dataclass(frozen=True)
class SceneHit:
    scene: Scene
    keyword: int
    confident: bool
    similarity: float | None = None
    anchored: bool = False

    def label(self) -> str:
        if self.similarity is None:
            return f"关键词 {self.keyword}"
        if self.keyword:
            return f"向量 {self.similarity:.2f}，关键词 {self.keyword}"
        return f"向量 {self.similarity:.2f}"


def keyword_matches(model: SearchModel, incoming: str, draft: str) -> list[SceneHit]:
    query = incoming or draft
    terms, query_key = scene_terms(query), _compact_key(query)
    if not terms:
        return []
    search_terms = terms - _COMMON_TERMS
    draft_terms = scene_terms(draft) - terms - _COMMON_TERMS if incoming else set()
    results = []
    for scene in model.scenes:
        exact = len(scene.name_key) >= 2 and scene.name_key in query_key
        topic_hits = len(search_terms & scene.topic)
        counts = Counter(index for term in search_terms for index in scene.example_postings.get(term, ()))
        example_hits = max(counts.values(), default=0)
        prior, paired = 0, False
        if incoming:
            for words in scene.prior_terms:
                overlap = len(terms & words)
                prior = max(prior, len(search_terms & words))
                paired |= overlap >= 3 and overlap >= 0.55 * max(len(terms), len(words))
        if scene.contextual and not (incoming and (paired or exact)):
            continue
        score = 12 * exact + 3 * topic_hits + example_hits + 4 * max(prior, 3 * paired)
        if score == 0:
            continue
        score += min(2, len(draft_terms & scene.topic))
        confident = bool(exact or paired or topic_hits >= 2 or topic_hits and example_hits or example_hits >= 3)
        results.append(SceneHit(scene, score, confident, anchored=exact or paired))
    return sorted(results, key=lambda hit: (hit.confident, hit.keyword), reverse=True)


def select_keyword_hits(hits: list[SceneHit], *, global_ranking: bool = False) -> list[SceneHit]:
    strong = [hit for hit in hits if hit.confident]
    if strong:
        if global_ranking:
            return strong
        return [hit for hit in strong[:2] if hit.keyword >= 0.75 * strong[0].keyword]
    return [hit for hit in hits[:1] if hit.keyword >= 3]


def _global_hit_key(hit: SceneHit) -> tuple[bool, float, bool, int]:
    # Two topic matches (six keyword points) and the semantic cutoff both map to 0.5.
    keyword_relevance = hit.keyword / (hit.keyword + 6.0)
    relevance = keyword_relevance
    if hit.similarity is not None:
        confidence_above_cutoff = (min(hit.similarity, 1.0) - _MIN_SIMILARITY) / (1.0 - _MIN_SIMILARITY)
        relevance = max(keyword_relevance, (1.0 + confidence_above_cutoff) / 2.0)
    return hit.confident, relevance, hit.anchored, hit.keyword


def rank_scene_hits(
    models: Sequence[SearchModel], selected: Sequence[Sequence[SceneHit]],
) -> list[tuple[int, SceneHit]]:
    """Rank accepted scenes together, retaining each scene's source model index."""
    if len(models) != len(selected):
        raise ValueError("Scene selections must align with the source models")
    ranked = [(model_index, hit) for model_index, hits in enumerate(selected) for hit in hits]
    return sorted(ranked, key=lambda entry: _global_hit_key(entry[1]), reverse=True)


@dataclass(frozen=True)
class Vector:
    model: str
    values: tuple[float, ...]

    @property
    def signature(self) -> tuple[str, int]:
        return (self.model, len(self.values))

    @classmethod
    def from_payload(cls, item: Any) -> Vector | None:
        raw = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(raw, list) or not raw or not all(type(value) in (float, int) for value in raw):
            return None
        try:
            values = tuple(map(float, raw))
            norm = math.hypot(*values)
        except (ValueError, OverflowError):
            return None
        if not norm or not math.isfinite(norm):
            return None
        name = str(item.get("model_name") or item.get("model") or "").strip()
        return cls(name, tuple(value / norm for value in values))


class SemanticIndex:
    def __init__(self, path: Path, task_name: str, embed: Any) -> None:
        self._path, self.task_name, self._embed = path, task_name, embed
        self._documents: dict[str, str] = {}
        self._vectors: dict[str, Vector] = {}
        self._current: tuple[str, int] | None = None
        self._worker: asyncio.Task | None = None
        self._index_retry = 0.0
        self._query_retry = 0.0
        self._query_slots = asyncio.Semaphore(2)
        self._queries: OrderedDict[str, tuple[float, Vector]] = OrderedDict()

    def update(self, scenes: list[Scene]) -> None:
        documents = {
            scene.fingerprint: scene.document for scene in scenes if scene.document and not scene.contextual
        }
        if documents != self._documents:
            self._documents = documents
            self._vectors = {key: vec for key, vec in self._vectors.items() if key in documents}
        self._schedule()

    def _schedule(self) -> None:
        if (
            self._documents.keys() - self._vectors.keys()
            and (self._worker is None or self._worker.done())
            and time.monotonic() >= self._index_retry
        ):
            self._worker = asyncio.create_task(self._index())

    async def close(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    def status(self) -> str:
        available = len(self._vectors)
        total = len(self._documents)
        label = f"{self.task_name} · {available}/{total} 个场景已索引"
        models = sorted({vector.model for vector in self._vectors.values() if vector.model})
        if models:
            label += " · 索引模型 " + "、".join(models)
        if self._worker is not None and not self._worker.done():
            return label + " · 正在更新"
        if time.monotonic() < max(self._index_retry, self._query_retry):
            return label + " · 向量服务暂不可用，关键词检索仍可用"
        return label

    def _database(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._path)

    def _load(self, wanted: set[str]) -> dict[str, Vector]:
        with closing(self._database()) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS scene_vectors ("
                "task TEXT, fingerprint TEXT, model TEXT, vector BLOB, "
                "PRIMARY KEY (task, fingerprint))"
            )
            return {
                key: Vector(model, tuple(value for (value,) in struct.iter_unpack("<f", blob)))
                for key, model, blob in db.execute(
                    "SELECT fingerprint, model, vector FROM scene_vectors WHERE task = ?", (self.task_name,)
                )
                if key in wanted and blob and len(blob) % 4 == 0
            }

    def _save(self, vectors: dict[str, Vector]) -> None:
        with closing(self._database()) as db, db:
            db.executemany(
                "INSERT OR REPLACE INTO scene_vectors VALUES (?, ?, ?, ?)",
                (
                    (self.task_name, key, vector.model, struct.pack(f"<{len(vector.values)}f", *vector.values))
                    for key, vector in vectors.items()
                ),
            )

    def _accept(self, vectors: dict[str, Vector]) -> None:
        for key, vec in vectors.items():
            if key in self._documents and (self._current is None or vec.signature == self._current):
                self._vectors[key] = vec

    async def _index(self) -> None:
        try:
            wanted = set(self._documents) - self._vectors.keys()
            self._accept(await asyncio.to_thread(self._load, wanted))
            pending = [(key, doc) for key, doc in self._documents.items() if key not in self._vectors]
            for start in range(0, len(pending), 24):
                batch = [(key, doc) for key, doc in pending[start:start + 24] if key in self._documents]
                if not batch:
                    continue
                result = await asyncio.wait_for(
                    self._embed(texts=[doc for _, doc in batch], task_name=self.task_name, max_concurrent=2),
                    timeout=40.0,
                )
                items = result.get("results") if isinstance(result, dict) and result.get("success", True) else None
                if not isinstance(items, list) or len(items) != len(batch):
                    raise ValueError("向量批次结果不完整")
                saved = {
                    key: vec for (key, _), item in zip(batch, items)
                    if key in self._documents and (vec := Vector.from_payload(item)) is not None
                    and (self._current is None or vec.signature == self._current)
                }
                if saved:
                    self._accept(saved)
                    await asyncio.to_thread(self._save, saved)
                if any(key in self._documents and key not in saved for key, _ in batch):
                    self._index_retry = time.monotonic() + _RETRY_SECONDS
                    break
        except Exception as exc:
            self._index_retry = time.monotonic() + _RETRY_SECONDS
            logger.warning("风格向量索引更新失败（%s），将使用关键词匹配", type(exc).__name__)

    def ready_for(self, model: SearchModel) -> bool:
        indexed = [scene.fingerprint for scene in model.scenes if scene.document and not scene.contextual]
        return bool(indexed) and all(key in self._vectors for key in indexed)

    async def query(self, text: str) -> Vector | None:
        text = text.strip()[:400]
        if len(text) < 3 or time.monotonic() < self._query_retry:
            return None
        key = _digest(text)
        cached = self._queries.get(key)
        if cached is not None and time.monotonic() - cached[0] <= 300:
            self._queries.move_to_end(key)
            return cached[1]
        try:
            async with asyncio.timeout(_QUERY_TIMEOUT):
                async with self._query_slots:
                    result = await self._embed(text=text, task_name=self.task_name)
            vector = Vector.from_payload(result)
        except Exception:
            vector = None
        if vector is None:
            self._query_retry = time.monotonic() + _RETRY_SECONDS
            return None
        if self._current != vector.signature:
            self._current = vector.signature
            self._queries.clear()
            self._vectors = {
                key: existing for key, existing in self._vectors.items() if existing.signature == self._current
            }
            self._schedule()
        self._queries[key] = (time.monotonic(), vector)
        if len(self._queries) > 256:
            self._queries.popitem(last=False)
        return vector

    def match(self, model: SearchModel, hits: list[SceneHit], query: Vector) -> SceneHit | None:
        if not self.ready_for(model):
            return None
        lexical = {hit.scene.fingerprint: hit for hit in hits}
        scores = []
        for scene in model.scenes:
            if not scene.document or scene.contextual:
                continue
            vector = self._vectors[scene.fingerprint]
            if vector.signature != query.signature:
                continue
            similarity = sum(a * b for a, b in zip(vector.values, query.values))
            lexical_hit = lexical.get(scene.fingerprint)
            keyword = lexical_hit.keyword if lexical_hit else 0
            hit = SceneHit(scene, keyword, True, similarity, anchored=bool(lexical_hit and lexical_hit.anchored))
            scores.append((similarity + min(keyword * 0.008, 0.04), hit))
        if not scores:
            return None
        best, winner = max(scores, key=lambda item: item[0])
        runner_up = max(
            (score for score, hit in scores if hit.scene.fingerprint != winner.scene.fingerprint),
            default=0.0,
        )
        floor = _MIN_SIMILARITY if len(scores) > 1 or winner.keyword else 0.80
        if winner.similarity is None or winner.similarity < floor or best - runner_up < _MIN_MARGIN:
            return None
        return winner


async def retrieve_scenes(
    models: list[SearchModel], incoming: str, draft: str, semantic: SemanticIndex | None,
    *, global_ranking: bool = False,
) -> tuple[list[list[SceneHit]], str]:
    rankings = [keyword_matches(model, incoming, draft) for model in models]
    selected = [select_keyword_hits(hits, global_ranking=global_ranking) for hits in rankings]
    if semantic is None:
        return selected, "关键词"
    if not incoming.strip():
        return selected, "关键词（无已确认来话）"
    candidates = []
    for idx, (model, hits) in enumerate(zip(models, rankings)):
        has_clear_keyword = (
            hits and hits[0].confident
            and (len(hits) == 1 or hits[0].keyword >= 1.3 * hits[1].keyword)
        )
        if not has_clear_keyword and semantic.ready_for(model):
            candidates.append(idx)
    if not candidates:
        return selected, "关键词" if any(selected) else "关键词未命中，暂无可用的向量候选"
    vector = await semantic.query(incoming)
    if vector is None:
        return selected, "向量查询不可用，使用关键词"
    matched = False
    for idx in candidates:
        winner = semantic.match(models[idx], rankings[idx], vector)
        if winner is not None:
            other_hits = [
                hit for hit in selected[idx] if hit.confident and hit.scene is not winner.scene
            ] if global_ranking else []
            selected[idx] = [winner, *other_hits]
            matched = True
    mode = "关键词与向量混合检索" if matched else "向量候选相近或相关度不足，使用关键词"
    return selected, mode
