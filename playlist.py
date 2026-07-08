import aiosqlite

import db


def _resolve(tracks: list[dict], target_seconds: float) -> tuple[int, float]:
    """Pure resolver: tracks sorted by order_index, each with duration_seconds
    and cumulative_start_seconds. Returns (order_index, offset_within_track)."""
    if not tracks:
        raise ValueError("Aucune piste")

    target = max(0.0, target_seconds)
    chosen = tracks[0]
    for t in tracks:
        if t["cumulative_start_seconds"] <= target:
            chosen = t
        else:
            break

    offset = min(target - chosen["cumulative_start_seconds"], chosen["duration_seconds"])
    return chosen["order_index"], offset


async def resolve_position(story_id: int, target_seconds: float) -> tuple[int, float]:
    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT order_index, duration_seconds, cumulative_start_seconds FROM tracks "
            "WHERE story_id = ? ORDER BY order_index",
            (story_id,),
        ) as cur:
            rows = await cur.fetchall()
    return _resolve([dict(r) for r in rows], target_seconds)
