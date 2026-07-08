import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import mutagen

import db

logger = logging.getLogger(__name__)

CONTES_ROOT = Path(os.environ.get("CONTES_ROOT", "/contes"))


async def compute_durations() -> dict:
    n_tracks = n_errors = n_stories = 0
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row

        async with conn.execute(
            "SELECT t.id, t.file_path FROM tracks t "
            "JOIN stories s ON s.id = t.story_id "
            "WHERE t.duration_seconds IS NULL AND s.status != 'excluded'"
        ) as cur:
            rows = await cur.fetchall()

        for row in rows:
            abs_path = CONTES_ROOT / row["file_path"]
            try:
                duration = mutagen.File(abs_path).info.length
            except Exception as e:
                logger.error(f"duration: échec lecture {abs_path}: {e}")
                n_errors += 1
                continue
            await conn.execute(
                "UPDATE tracks SET duration_seconds = ?, status = 'duration_known' WHERE id = ?",
                (duration, row["id"]),
            )
            n_tracks += 1
        await conn.commit()

        async with conn.execute(
            """
            SELECT s.id FROM stories s
            WHERE s.status = 'discovered'
              AND NOT EXISTS (
                  SELECT 1 FROM tracks t WHERE t.story_id = s.id AND t.duration_seconds IS NULL
              )
            """
        ) as cur:
            story_ids = [r["id"] for r in await cur.fetchall()]

        for story_id in story_ids:
            async with conn.execute(
                "SELECT id, duration_seconds FROM tracks WHERE story_id = ? ORDER BY order_index",
                (story_id,),
            ) as cur:
                tracks = await cur.fetchall()
            cumulative = 0.0
            for t in tracks:
                await conn.execute(
                    "UPDATE tracks SET cumulative_start_seconds = ? WHERE id = ?",
                    (cumulative, t["id"]),
                )
                cumulative += t["duration_seconds"]
            await conn.execute(
                "UPDATE stories SET total_duration_seconds = ?, status = 'tracks_catalogued', updated_at = ? WHERE id = ?",
                (cumulative, now, story_id),
            )
            n_stories += 1
        await conn.commit()

    result = {"tracks_measured": n_tracks, "errors": n_errors, "stories_catalogued": n_stories}
    logger.info(f"duration: {result}")
    return result
