import json
import logging

import aiosqlite

import chroma_store
import db

logger = logging.getLogger(__name__)


async def _get_speaker_map(conn: aiosqlite.Connection, story_id: int) -> dict:
    async with conn.execute(
        "SELECT track_id, speaker_label, character_name FROM speaker_map WHERE story_id = ?",
        (story_id,),
    ) as cur:
        return {(r["track_id"], r["speaker_label"]): r["character_name"] for r in await cur.fetchall()}


async def _get_story_segments(conn: aiosqlite.Connection, story_id: int) -> list[dict]:
    """Un segment = une phrase transcrite, avec sa position GLOBALE (toutes pistes
    confondues, via cumulative_start_seconds — même conversion que
    reference/summarize.py::_bucket_into_periods) et son locuteur résolu (nom de
    personnage si connu, sinon le label brut SPEAKER_XX) — voir
    chroma_store.upsert_segments, la granularité de recherche la plus fine."""
    speaker_map = await _get_speaker_map(conn, story_id)
    async with conn.execute(
        """
        SELECT ts.id, ts.text, ts.speaker_label, ts.start_seconds, ts.end_seconds,
               t.id AS track_id, t.cumulative_start_seconds
        FROM transcript_segments ts
        JOIN tracks t ON t.id = ts.track_id
        WHERE t.story_id = ?
        ORDER BY t.order_index, ts.start_seconds
        """,
        (story_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [
        {
            "segment_id": r["id"],
            "text": r["text"],
            "global_start_seconds": r["cumulative_start_seconds"] + r["start_seconds"],
            "global_end_seconds": r["cumulative_start_seconds"] + r["end_seconds"],
            "speaker": speaker_map.get((r["track_id"], r["speaker_label"]), r["speaker_label"]),
        }
        for r in rows
    ]


async def backfill_segment_embeddings(story_id: int | None = None) -> dict:
    """Ré-embedde les segments de transcription des histoires déjà 'ready' avant
    l'ajout de cette granularité de recherche (voir chroma_store.upsert_segments) —
    pour les nouvelles histoires, embed_pending s'en charge déjà à l'étape normale."""
    n_stories = 0
    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = "SELECT id FROM stories WHERE status = 'ready'"
        params: tuple = ()
        if story_id is not None:
            query += " AND id = ?"
            params = (story_id,)
        async with conn.execute(query, params) as cur:
            stories = await cur.fetchall()

        for story in stories:
            segments = await _get_story_segments(conn, story["id"])
            chroma_store.upsert_segments(story["id"], segments)
            n_stories += 1
            logger.info(f"backfill_segment_embeddings: histoire {story['id']} ok ({len(segments)} segments)")

    result = {"stories_done": n_stories}
    logger.info(f"backfill_segment_embeddings: {result}")
    return result


async def embed_pending(story_id: int | None = None) -> dict:
    n_stories = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = (
            "SELECT id, title, author, short_summary, long_summary, total_duration_seconds, keywords "
            "FROM stories WHERE status = 'summarized'"
        )
        params: tuple = ()
        if story_id is not None:
            query += " AND id = ?"
            params = (story_id,)
        async with conn.execute(query, params) as cur:
            stories = await cur.fetchall()

        for story in stories:
            chroma_store.upsert_story_summary(
                story["id"], story["title"], story["author"],
                story["short_summary"], story["long_summary"], story["total_duration_seconds"],
            )
            keywords = json.loads(story["keywords"]) if story["keywords"] else []
            chroma_store.upsert_story_keywords(
                story["id"], story["title"], story["author"], keywords, story["total_duration_seconds"],
            )

            async with conn.execute(
                "SELECT period_index, global_start_seconds, global_end_seconds, "
                "summary_text, raw_transcript_text FROM periods WHERE story_id = ?",
                (story["id"],),
            ) as cur2:
                periods = await cur2.fetchall()
            for p in periods:
                chroma_store.upsert_period(
                    story["id"], p["period_index"], p["global_start_seconds"], p["global_end_seconds"],
                    p["summary_text"], p["raw_transcript_text"],
                )

            segments = await _get_story_segments(conn, story["id"])
            chroma_store.upsert_segments(story["id"], segments)

            await conn.execute("UPDATE stories SET status = 'ready' WHERE id = ?", (story["id"],))
            await conn.commit()
            n_stories += 1
            logger.info(f"embed: histoire {story['id']} ok ({len(periods)} périodes)")

    result = {"stories_done": n_stories}
    logger.info(f"embed: {result}")
    return result
