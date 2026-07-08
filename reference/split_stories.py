"""Detects folders that are actually an anthology of independent stories
(e.g. "Contes de la rue Broca": 13 unrelated tales, one per track) rather
than a single continuous narrative split into chapters (e.g. Le Petit
Prince). Runs after transcription so the decision is based on each track's
actual transcribed opening, not just file/folder naming."""

import logging
import os
from datetime import datetime, timezone

import aiosqlite

import db
import llm
from . import scan

logger = logging.getLogger(__name__)

MAX_OPENING_CHARS = int(os.environ.get("MAX_OPENING_CHARS_FOR_SPLIT", "300"))

_SPLIT_TOOL = [{
    "type": "function",
    "function": {
        "name": "group_tracks_into_stories",
        "description": (
            "Regroupe une liste ordonnée de pistes audio (issues d'un même dossier) en une ou "
            "plusieurs histoires indépendantes, à partir du début transcrit de chaque piste"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "track_order_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "order_index des pistes qui forment une seule et même "
                                    "histoire continue, dans l'ordre"
                                ),
                            },
                            "title": {"type": "string", "description": "Titre de cette histoire"},
                        },
                        "required": ["track_order_indices", "title"],
                    },
                }
            },
            "required": ["groups"],
        },
    },
}]

_SYSTEM_PROMPT = (
    "Tu analyses les pistes audio d'un dossier de contes pour déterminer si elles forment "
    "UNE SEULE histoire continue (chapitres successifs, mêmes personnages, l'intrigue se "
    "poursuit d'une piste à l'autre — c'est le cas le plus fréquent) ou si ce dossier est en "
    "fait un RECUEIL de plusieurs histoires indépendantes (personnages différents, intrigues "
    "sans rapport, chaque piste raconte une histoire complète en elle-même — comme un recueil "
    "de contes classiques). Pour en juger, compare surtout la FIN de chaque piste avec le "
    "DÉBUT de la piste suivante : si l'intrigue/la scène se poursuit directement (mêmes "
    "personnages, même fil narratif), c'est la même histoire ; si la piste suivante repart "
    "sur une prémisse différente (nouveaux personnages, nouveau décor, une formule "
    "d'ouverture du type 'il était une fois' qui relance une histoire entièrement neuve), "
    "c'est une histoire indépendante. En cas de doute, privilégie une seule histoire "
    "continue. Réponds en appelant group_tracks_into_stories avec un groupe par histoire "
    "détectée, dans l'ordre, chaque groupe listant les order_index des pistes qui la "
    "composent et un titre adapté."
)


async def _get_track_boundaries(conn: aiosqlite.Connection, story_id: int) -> list[dict]:
    async with conn.execute(
        "SELECT id, order_index, file_path, duration_seconds FROM tracks "
        "WHERE story_id = ? ORDER BY order_index",
        (story_id,),
    ) as cur:
        tracks = await cur.fetchall()

    result = []
    for t in tracks:
        async with conn.execute(
            "SELECT text FROM transcript_segments WHERE track_id = ? ORDER BY start_seconds LIMIT 3",
            (t["id"],),
        ) as cur:
            opening_segs = await cur.fetchall()
        async with conn.execute(
            "SELECT text FROM transcript_segments WHERE track_id = ? ORDER BY start_seconds DESC LIMIT 3",
            (t["id"],),
        ) as cur:
            closing_segs = list(reversed(await cur.fetchall()))

        result.append({
            "track_id": t["id"],
            "order_index": t["order_index"],
            "duration_seconds": t["duration_seconds"],
            "title": scan.track_title(t["file_path"]),
            "opening": " ".join(s["text"] for s in opening_segs)[:MAX_OPENING_CHARS],
            "closing": " ".join(s["text"] for s in closing_segs)[-MAX_OPENING_CHARS:],
        })
    return result


async def _apply_split(conn: aiosqlite.Connection, story: dict, tracks: list[dict], groups: list[dict]) -> int:
    """Reassigns story_id/order_index/cumulative_start_seconds per detected group.
    Reuses the original story row for the first group; creates new rows for the rest."""
    now = datetime.now(timezone.utc).isoformat()
    by_index = {t["order_index"]: t for t in tracks}
    n_results = 0

    for i, group in enumerate(groups):
        group_tracks = [by_index[idx] for idx in group["track_order_indices"] if idx in by_index]
        if not group_tracks:
            continue
        n_results += 1

        if i == 0:
            target_story_id = story["id"]
            await conn.execute(
                "UPDATE stories SET title = ?, status = 'grouped', updated_at = ? WHERE id = ?",
                (group["title"], now, target_story_id),
            )
        else:
            cur = await conn.execute(
                "INSERT INTO stories (title, author, folder_path, merged_folders, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'grouped', ?, ?)",
                (group["title"], story["author"], f"{story['folder_path']}#{i}",
                 story["merged_folders"], now, now),
            )
            target_story_id = cur.lastrowid

        cumulative = 0.0
        for order_index, t in enumerate(group_tracks):
            await conn.execute(
                "UPDATE tracks SET story_id = ?, order_index = ?, cumulative_start_seconds = ? WHERE id = ?",
                (target_story_id, order_index, cumulative, t["track_id"]),
            )
            cumulative += t["duration_seconds"]

        await conn.execute(
            "UPDATE stories SET total_duration_seconds = ? WHERE id = ?", (cumulative, target_story_id),
        )

    return n_results


async def split_pending(story_id: int | None = None) -> dict:
    n_split = n_unchanged = n_errors = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = "SELECT * FROM stories WHERE status = 'transcribed'"
        params: tuple = ()
        if story_id is not None:
            query += " AND id = ?"
            params = (story_id,)
        async with conn.execute(query, params) as cur:
            stories = await cur.fetchall()

    for story in stories:
        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            tracks = await _get_track_boundaries(conn, story["id"])

        if len(tracks) <= 1:
            async with aiosqlite.connect(db.DB_PATH) as conn:
                await conn.execute("UPDATE stories SET status = 'grouped' WHERE id = ?", (story["id"],))
                await conn.commit()
            n_unchanged += 1
            continue

        blocks = "\n".join(
            f"[piste {t['order_index']}] \"{t['title']}\"\n"
            f"  début: {t['opening'] or '(silence)'}\n"
            f"  fin: {t['closing'] or '(silence)'}"
            for t in tracks
        )
        result = await llm.call_tool(
            system=_SYSTEM_PROMPT,
            user=f"DOSSIER: {story['title']} ({story['author']})\n\n{blocks}",
            tool=_SPLIT_TOOL,
        )
        groups = result.get("groups")
        if not groups:
            logger.error(f"split_stories: pas de réponse LLM pour l'histoire {story['id']}")
            n_errors += 1
            continue

        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            n_results = await _apply_split(conn, story, tracks, groups)
            await conn.commit()

        if n_results > 1:
            n_split += 1
            logger.info(f"split_stories: histoire {story['id']} scindée en {n_results} histoires")
        else:
            n_unchanged += 1
            logger.info(f"split_stories: histoire {story['id']} inchangée (une seule histoire)")

    result = {"split": n_split, "unchanged": n_unchanged, "errors": n_errors}
    logger.info(f"split_stories: {result}")
    return result
