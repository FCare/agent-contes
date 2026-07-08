"""
ChromaDB vector store for the contes catalogue.

Single collection "contes" holding 3 granularities of documents,
distinguished by the "kind" metadata field:
- story_summary    : one per story (résumé général)
- period_summary   : one per ~3min window (mini-résumé)
- period_transcript: one per ~3min window (texte brut diarizé, pour garder
                     la diarization elle-même cherchable par embeddings)
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("/data/chroma")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def _get_ef():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _collection_handle():
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name="contes",
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _id(*parts: str) -> str:
    return hashlib.md5(":".join(parts).encode()).hexdigest()


def upsert_story_summary(story_id: int, title: str, author: str,
                          short_summary: str, long_summary: str,
                          total_duration_seconds: float) -> None:
    try:
        col = _collection_handle()
        doc = f"{title}\n{short_summary}\n{long_summary}"
        col.upsert(
            ids=[_id("story", str(story_id))],
            documents=[doc],
            metadatas=[{
                "kind": "story_summary",
                "story_id": story_id,
                "period_index": -1,
                "title": title,
                "author": author,
                "global_start_seconds": 0.0,
                "global_end_seconds": total_duration_seconds,
                "duration_int": int(total_duration_seconds),
            }],
        )
    except Exception as e:
        logger.error(f"ChromaDB upsert story_summary failed (story {story_id}): {e}")


def upsert_story_keywords(story_id: int, title: str, author: str,
                          keywords: list[str], total_duration_seconds: float) -> None:
    """Second, separate embedding computed only on a short keyword/theme list rather
    than the full summary paragraph — a literal detail ('un arbre à pain') gets diluted
    in a long prose embedding but stands out in a short keyword-list one."""
    if not keywords:
        return
    try:
        col = _collection_handle()
        col.upsert(
            ids=[_id("story_keywords", str(story_id))],
            documents=[", ".join(keywords)],
            metadatas=[{
                "kind": "story_keywords",
                "story_id": story_id,
                "period_index": -1,
                "title": title,
                "author": author,
                "global_start_seconds": 0.0,
                "global_end_seconds": total_duration_seconds,
                "duration_int": int(total_duration_seconds),
            }],
        )
    except Exception as e:
        logger.error(f"ChromaDB upsert story_keywords failed (story {story_id}): {e}")


def upsert_period(story_id: int, period_index: int, global_start_seconds: float,
                   global_end_seconds: float, summary_text: str, raw_transcript_text: str) -> None:
    try:
        col = _collection_handle()
        base_meta = {
            "story_id": story_id,
            "period_index": period_index,
            "global_start_seconds": global_start_seconds,
            "global_end_seconds": global_end_seconds,
        }
        ids, docs, metas = [], [], []
        if summary_text:
            ids.append(_id("period", str(story_id), str(period_index), "summary"))
            docs.append(summary_text)
            metas.append({**base_meta, "kind": "period_summary"})
        if raw_transcript_text:
            ids.append(_id("period", str(story_id), str(period_index), "transcript"))
            docs.append(raw_transcript_text)
            metas.append({**base_meta, "kind": "period_transcript"})
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
    except Exception as e:
        logger.error(f"ChromaDB upsert period failed (story {story_id}, period {period_index}): {e}")


def upsert_theme_class(class_id: int, label: str, description: str) -> None:
    """Classe thématique 'libre' (découverte depuis le contenu, pas un vocabulaire fixé
    à l'avance) — embeddée séparément pour router une requête libre ('des histoires de
    pirates') vers la classe la plus proche par similarité plutôt que par mot-clé exact."""
    try:
        col = _collection_handle()
        col.upsert(
            ids=[_id("theme_class", str(class_id))],
            documents=[f"{label}\n{description}"],
            metadatas=[{"kind": "theme_class", "class_id": class_id, "label": label}],
        )
    except Exception as e:
        logger.error(f"ChromaDB upsert theme_class failed (class {class_id}): {e}")


def search_theme_classes(query: str, n_results: int = 3) -> list[dict]:
    try:
        col = _collection_handle()
        count = col.count()
        if count == 0:
            return []
        results = col.query(query_texts=[query], n_results=min(n_results, count), where={"kind": "theme_class"})
        return _format_results(results)
    except Exception as e:
        logger.error(f"ChromaDB search_theme_classes failed: {e}")
        return []


def _format_results(results: dict) -> list[dict]:
    out = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({"content": doc, "metadata": meta, "score": round(1 - dist, 3)})
    return out


def search_stories(query: str, n_results: int = 5,
                    min_duration_seconds: float | None = None,
                    max_duration_seconds: float | None = None) -> list[dict]:
    """Searches both the summary embedding and the separate keywords embedding per
    story, then dedupes to the best-scoring document per story_id — a story can
    otherwise appear twice (once per kind) and crowd out other results."""
    try:
        col = _collection_handle()
        count = col.count()
        if count == 0:
            return []
        where: dict = {"kind": {"$in": ["story_summary", "story_keywords"]}}
        duration_clauses = []
        if min_duration_seconds is not None:
            duration_clauses.append({"duration_int": {"$gte": int(min_duration_seconds)}})
        if max_duration_seconds is not None:
            duration_clauses.append({"duration_int": {"$lte": int(max_duration_seconds)}})
        if duration_clauses:
            where = {"$and": [where, *duration_clauses]}
        fetch_n = min(count, n_results * 3)
        results = col.query(query_texts=[query], n_results=fetch_n, where=where)
        formatted = _format_results(results)

        best_by_story: dict[int, dict] = {}
        for r in formatted:
            sid = r["metadata"]["story_id"]
            if sid not in best_by_story or r["score"] > best_by_story[sid]["score"]:
                best_by_story[sid] = r
        deduped = sorted(best_by_story.values(), key=lambda r: r["score"], reverse=True)
        return deduped[:n_results]
    except Exception as e:
        logger.error(f"ChromaDB search_stories failed: {e}")
        return []


def search_moments(query: str, story_id: int | None = None, n_results: int = 5) -> list[dict]:
    try:
        col = _collection_handle()
        count = col.count()
        if count == 0:
            return []
        where: dict = {"kind": {"$in": ["period_summary", "period_transcript"]}}
        if story_id is not None:
            where = {"$and": [where, {"story_id": story_id}]}
        results = col.query(query_texts=[query], n_results=min(n_results, count), where=where)
        return _format_results(results)
    except Exception as e:
        logger.error(f"ChromaDB search_moments failed: {e}")
        return []
