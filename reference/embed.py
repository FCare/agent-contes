import json
import logging

import aiosqlite

import chroma_store
import db

logger = logging.getLogger(__name__)


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

            await conn.execute("UPDATE stories SET status = 'ready' WHERE id = ?", (story["id"],))
            await conn.commit()
            n_stories += 1
            logger.info(f"embed: histoire {story['id']} ok ({len(periods)} périodes)")

    result = {"stories_done": n_stories}
    logger.info(f"embed: {result}")
    return result
