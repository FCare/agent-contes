import json
import logging
import os

import aiosqlite

import chroma_store
import db
import llm

logger = logging.getLogger(__name__)

PERIOD_SECONDS = float(os.environ.get("PERIOD_SECONDS", "180"))
PERIODS_PER_LLM_CALL = int(os.environ.get("PERIODS_PER_LLM_CALL", "8"))

_PERIOD_SUMMARY_TOOL = [{
    "type": "function",
    "function": {
        "name": "summarize_periods",
        "description": "Résume chaque période de quelques minutes d'un conte à partir de son transcript diarizé",
        "parameters": {
            "type": "object",
            "properties": {
                "periods": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period_index": {"type": "integer"},
                            "summary": {
                                "type": "string",
                                "description": "1-2 phrases en français résumant ce qui se passe dans cette période",
                            },
                        },
                        "required": ["period_index", "summary"],
                    },
                }
            },
            "required": ["periods"],
        },
    },
}]

_STORY_SUMMARY_TOOL = [{
    "type": "function",
    "function": {
        "name": "summarize_story",
        "description": "Génère le résumé général d'un conte à partir de ses mini-résumés de période",
        "parameters": {
            "type": "object",
            "properties": {
                "short_summary": {
                    "type": "string",
                    "description": "1-2 phrases accrocheuses résumant l'histoire, pour aider à la choisir",
                },
                "long_summary": {
                    "type": "string",
                    "description": (
                        "Résumé complet en 5-8 phrases: de quoi ça parle, les personnages "
                        "principaux, ce qui se passe, le ton de l'histoire."
                    ),
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "5 à 15 mots ou courtes expressions représentant les personnages "
                        "(ex: 'sorcière', 'Bouma'), objets ou lieux marquants (ex: 'arbre à "
                        "pain', 'forêt'), et thèmes (ex: 'courage', 'partage') de l'histoire — "
                        "utilisés pour la recherche, à part du résumé"
                    ),
                },
            },
            "required": ["short_summary", "long_summary", "keywords"],
        },
    },
}]

_KEYWORDS_TOOL = [{
    "type": "function",
    "function": {
        "name": "extract_keywords",
        "description": "Extrait les mots-clés d'un conte à partir de ses mini-résumés de période",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "5 à 15 mots ou courtes expressions représentant les personnages, "
                        "objets ou lieux marquants, et thèmes de l'histoire "
                        "(ex: 'sorcière', 'arbre à pain', 'courage')"
                    ),
                }
            },
            "required": ["keywords"],
        },
    },
}]


async def _get_speaker_map(conn: aiosqlite.Connection, story_id: int) -> dict:
    # Clé (track_id, speaker_label) : la diarization tourne piste par piste, un label
    # brut seul n'est donc pas unique/stable à travers les pistes d'une même histoire.
    async with conn.execute(
        "SELECT track_id, speaker_label, character_name FROM speaker_map WHERE story_id = ?", (story_id,)
    ) as cur:
        return {(r["track_id"], r["speaker_label"]): r["character_name"] for r in await cur.fetchall()}


async def _bucket_into_periods(conn: aiosqlite.Connection, story_id: int) -> dict[int, dict]:
    """Group diarized segments into fixed-size (PERIOD_SECONDS) windows on the
    story-global timeline, using cumulative_start_seconds to convert
    track-relative segment times into story-relative ones."""
    speaker_map = await _get_speaker_map(conn, story_id)

    async with conn.execute(
        """
        SELECT ts.start_seconds, ts.end_seconds, ts.speaker_label, ts.text,
               t.id AS track_id, t.cumulative_start_seconds
        FROM transcript_segments ts
        JOIN tracks t ON t.id = ts.track_id
        WHERE t.story_id = ?
        ORDER BY t.order_index, ts.start_seconds
        """,
        (story_id,),
    ) as cur:
        rows = await cur.fetchall()

    periods: dict[int, dict] = {}
    for r in rows:
        global_start = r["cumulative_start_seconds"] + r["start_seconds"]
        period_index = int(global_start // PERIOD_SECONDS)
        p = periods.setdefault(period_index, {"lines": []})
        speaker = speaker_map.get((r["track_id"], r["speaker_label"]), r["speaker_label"])
        p["lines"].append(f"{speaker}: {r['text']}")
    return periods


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


_SUMMARIZE_BATCH_ATTEMPTS = 3


async def _summarize_batch(conn: aiosqlite.Connection, story_id: int, title: str,
                            periods: dict[int, dict], batch: list[int]) -> set[int]:
    """Tries a single batch, returns the set of period_index actually summarized."""
    blocks = []
    for idx in batch:
        start = idx * PERIOD_SECONDS
        end = start + PERIOD_SECONDS
        text = "\n".join(periods[idx]["lines"])
        blocks.append(f"=== PÉRIODE {idx} ({start:.0f}s-{end:.0f}s) ===\n{text}")

    result = await llm.call_tool(
        system=(
            "Tu résumes, période par période, le transcript diarizé d'un conte audio "
            "pour enfants. Chaque ligne du transcript est préfixée par le nom du "
            "personnage ou par \"Narrateur\". Pour chaque période fournie, résume en "
            "1-2 phrases ce qui s'y passe. Appelle summarize_periods avec autant "
            "d'entrées que de périodes fournies."
        ),
        user=f"TITRE: {title}\n\n" + "\n\n".join(blocks),
        tool=_PERIOD_SUMMARY_TOOL,
    )
    done = set()
    for entry in result.get("periods", []):
        idx = entry.get("period_index")
        summary = entry.get("summary")
        if idx not in periods or not summary:
            continue
        start = idx * PERIOD_SECONDS
        end = start + PERIOD_SECONDS
        await conn.execute(
            """
            INSERT INTO periods (story_id, period_index, global_start_seconds,
                                  global_end_seconds, raw_transcript_text, summary_text)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(story_id, period_index) DO UPDATE SET
                global_start_seconds = excluded.global_start_seconds,
                global_end_seconds = excluded.global_end_seconds,
                raw_transcript_text = excluded.raw_transcript_text,
                summary_text = excluded.summary_text
            """,
            (story_id, idx, start, end, "\n".join(periods[idx]["lines"]), summary),
        )
        await db.sync_period_fts(conn, story_id, idx, summary, "\n".join(periods[idx]["lines"]))
        done.add(idx)
    await conn.commit()
    return done


async def _summarize_periods(story_id: int, title: str, periods: dict[int, dict]) -> bool:
    """Renvoie False si au moins une période reste sans résumé après plusieurs tentatives
    — le LLM de ce backend peut répondre sans appeler l'outil malgré tool_choice="required"
    (constaté empiriquement), ce qui laissait silencieusement des trous de périodes
    entières dans des histoires marquées 'ready' sans qu'aucune erreur ne soit loggée."""
    indices = sorted(periods)
    all_ok = True
    async with aiosqlite.connect(db.DB_PATH) as conn:
        for batch in _chunks(indices, PERIODS_PER_LLM_CALL):
            remaining = list(batch)
            for attempt in range(1, _SUMMARIZE_BATCH_ATTEMPTS + 1):
                done = await _summarize_batch(conn, story_id, title, periods, remaining)
                remaining = [idx for idx in remaining if idx not in done]
                if not remaining:
                    break
                logger.warning(
                    f"summarize: histoire {story_id}, périodes {remaining} sans résumé "
                    f"(tentative {attempt}/{_SUMMARIZE_BATCH_ATTEMPTS}), nouvel essai"
                )
            if remaining:
                logger.error(
                    f"summarize: histoire {story_id}, périodes {remaining} définitivement "
                    f"sans résumé après {_SUMMARIZE_BATCH_ATTEMPTS} tentatives"
                )
                all_ok = False
    return all_ok


async def _summarize_story(conn: aiosqlite.Connection, story_id: int, title: str, author: str) -> bool:
    async with conn.execute(
        "SELECT period_index, summary_text FROM periods WHERE story_id = ? ORDER BY period_index",
        (story_id,),
    ) as cur:
        rows = await cur.fetchall()
    if not rows:
        return False

    periods_block = "\n".join(f"[période {r['period_index']}] {r['summary_text']}" for r in rows)
    result = await llm.call_tool(
        system=(
            "Tu écris le résumé général d'un conte audio pour enfants à partir de la suite "
            "chronologique de ses mini-résumés de période. Appelle summarize_story."
        ),
        user=f"TITRE: {title}\nAUTEUR/INTERPRÈTE: {author}\n\nMINI-RÉSUMÉS PAR PÉRIODE:\n{periods_block}",
        tool=_STORY_SUMMARY_TOOL,
    )
    short_summary = result.get("short_summary")
    long_summary = result.get("long_summary")
    keywords = result.get("keywords") or []
    if not short_summary or not long_summary:
        return False

    await conn.execute(
        "UPDATE stories SET short_summary = ?, long_summary = ?, keywords = ?, status = 'summarized' WHERE id = ?",
        (short_summary, long_summary, json.dumps(keywords, ensure_ascii=False), story_id),
    )
    await db.sync_story_fts(conn, story_id, title, short_summary, long_summary, keywords)
    await conn.commit()
    return True


async def _extract_keywords(conn: aiosqlite.Connection, story_id: int, title: str, author: str) -> list[str] | None:
    async with conn.execute(
        "SELECT period_index, summary_text FROM periods WHERE story_id = ? ORDER BY period_index",
        (story_id,),
    ) as cur:
        rows = await cur.fetchall()
    if not rows:
        return None

    periods_block = "\n".join(f"[période {r['period_index']}] {r['summary_text']}" for r in rows)
    result = await llm.call_tool(
        system=(
            "Tu extrais les mots-clés d'un conte audio pour enfants à partir de la suite "
            "chronologique de ses mini-résumés de période. Appelle extract_keywords."
        ),
        user=f"TITRE: {title}\nAUTEUR/INTERPRÈTE: {author}\n\nMINI-RÉSUMÉS PAR PÉRIODE:\n{periods_block}",
        tool=_KEYWORDS_TOOL,
    )
    return result.get("keywords") or None


async def backfill_keywords(story_id: int | None = None) -> dict:
    """For stories already 'ready' from before keywords existed: extract and embed
    them without touching the already-generated summary."""
    n_done = n_errors = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = "SELECT id, title, author, total_duration_seconds FROM stories WHERE status = 'ready' AND (keywords IS NULL OR keywords = '')"
        params: tuple = ()
        if story_id is not None:
            query += " AND id = ?"
            params = (story_id,)
        async with conn.execute(query, params) as cur:
            stories = await cur.fetchall()

    for story in stories:
        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            keywords = await _extract_keywords(conn, story["id"], story["title"], story["author"])

        if not keywords:
            logger.error(f"backfill_keywords: pas de réponse LLM pour l'histoire {story['id']}")
            n_errors += 1
            continue

        async with aiosqlite.connect(db.DB_PATH) as conn:
            await conn.execute(
                "UPDATE stories SET keywords = ? WHERE id = ?",
                (json.dumps(keywords, ensure_ascii=False), story["id"]),
            )
            await conn.commit()

        chroma_store.upsert_story_keywords(
            story["id"], story["title"], story["author"], keywords, story["total_duration_seconds"],
        )
        n_done += 1
        logger.info(f"backfill_keywords: histoire {story['id']} ok ({len(keywords)} mots-clés)")

    result = {"done": n_done, "errors": n_errors}
    logger.info(f"backfill_keywords: {result}")
    return result


async def backfill_fts(story_id: int | None = None) -> dict:
    """Pure-SQL migration: (re)populates stories_fts/periods_fts for stories already
    'ready' from before the FTS fallback existed, using data already in the DB."""
    n_stories = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = (
            "SELECT id, title, short_summary, long_summary, keywords FROM stories WHERE status = 'ready'"
        )
        params: tuple = ()
        if story_id is not None:
            query += " AND id = ?"
            params = (story_id,)
        async with conn.execute(query, params) as cur:
            stories = await cur.fetchall()

        for story in stories:
            keywords = json.loads(story["keywords"]) if story["keywords"] else []
            await db.sync_story_fts(
                conn, story["id"], story["title"], story["short_summary"], story["long_summary"], keywords
            )

            async with conn.execute(
                "SELECT period_index, summary_text, raw_transcript_text FROM periods WHERE story_id = ?",
                (story["id"],),
            ) as cur2:
                periods = await cur2.fetchall()
            for p in periods:
                await db.sync_period_fts(
                    conn, story["id"], p["period_index"], p["summary_text"], p["raw_transcript_text"]
                )
            n_stories += 1
        await conn.commit()

    result = {"stories_done": n_stories}
    logger.info(f"backfill_fts: {result}")
    return result


async def summarize_pending(story_id: int | None = None) -> dict:
    n_stories = n_errors = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = "SELECT id, title, author FROM stories WHERE status = 'speakers_identified'"
        params: tuple = ()
        if story_id is not None:
            query += " AND id = ?"
            params = (story_id,)
        async with conn.execute(query, params) as cur:
            stories = await cur.fetchall()

    for story in stories:
        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            periods = await _bucket_into_periods(conn, story["id"])

        if not periods:
            logger.error(f"summarize: aucune période pour l'histoire {story['id']}")
            n_errors += 1
            continue

        periods_ok = await _summarize_periods(story["id"], story["title"], periods)
        if not periods_ok:
            # Ne PAS finaliser (status reste 'speakers_identified') : une histoire avec des
            # périodes manquantes ne doit jamais atteindre 'ready' silencieusement — le
            # prochain passage de summarize_pending retentera automatiquement les périodes
            # manquantes plutôt que de figer un catalogue incomplet.
            n_errors += 1
            logger.error(f"summarize: histoire {story['id']} incomplète, finalisation reportée")
            continue

        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            ok = await _summarize_story(conn, story["id"], story["title"], story["author"])

        if ok:
            n_stories += 1
            logger.info(f"summarize: histoire {story['id']} ok ({len(periods)} périodes)")
        else:
            n_errors += 1
            logger.error(f"summarize: échec résumé général pour l'histoire {story['id']}")

    result = {"stories_done": n_stories, "errors": n_errors}
    logger.info(f"summarize: {result}")
    return result
