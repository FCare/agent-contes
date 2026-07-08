import logging
import os
from itertools import groupby

import aiosqlite

import db
import llm

logger = logging.getLogger(__name__)

# Taille max (en caractères) de transcript envoyée au LLM en un seul appel. Une histoire
# plus longue est découpée en plusieurs chunks séquentiels, TOUJOURS le long des
# frontières de piste (jamais au milieu d'une piste) : la diarization tourne piste par
# piste (voir reference/transcribe.py), donc un label SPEAKER_XX n'est stable qu'AU SEIN
# d'une même piste — le réutiliser d'une piste à l'autre désignerait potentiellement un
# personnage différent. Un chunk peut contenir plusieurs pistes courtes ; si une seule
# piste dépasse déjà le budget, elle est découpée seule (la stabilité du label est alors
# toujours garantie, puisqu'on reste à l'intérieur d'une unique diarization).
CHUNK_CHARS_FOR_SPEAKER_ID = int(os.environ.get("CHUNK_CHARS_FOR_SPEAKER_ID", "12000"))

_SPEAKER_MAP_TOOL = [{
    "type": "function",
    "function": {
        "name": "identify_speakers",
        "description": "Identifie à qui correspond chaque label de locuteur (SPEAKER_XX) dans un extrait de conte transcrit et diarizé",
        "parameters": {
            "type": "object",
            "properties": {
                "speakers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "track_id": {
                                "type": "integer",
                                "description": "Identifiant de piste tel qu'indiqué dans l'en-tête '=== PISTE N ==='",
                            },
                            "speaker_label": {"type": "string", "description": "Label brut, ex: SPEAKER_00"},
                            "character_name": {
                                "type": "string",
                                "description": (
                                    "Nom du personnage identifié à partir du contenu et des tournures de "
                                    "dialogue (ex: 'le Loup', 'la Reine'), ou \"Narrateur\" si cette voix "
                                    "porte la narration. Réutilise EXACTEMENT le même nom qu'un personnage "
                                    "déjà identifié dans une piste précédente s'il s'agit de lui à nouveau "
                                    "(reconnaissable au contenu, pas au label — les labels ne se répètent "
                                    "pas d'une piste à l'autre)."
                                ),
                            },
                        },
                        "required": ["track_id", "speaker_label", "character_name"],
                    },
                }
            },
            "required": ["speakers"],
        },
    },
}]


async def _get_story_lines(conn: aiosqlite.Connection, story_id: int) -> list[tuple[int, str, str]]:
    """Renvoie (track_id, speaker_label, text), ordonné par piste puis par temps —
    l'ordre des pistes est important : les frontières de piste bornent les chunks."""
    async with conn.execute(
        """
        SELECT t.id AS track_id, ts.speaker_label, ts.text
        FROM transcript_segments ts
        JOIN tracks t ON t.id = ts.track_id
        WHERE t.story_id = ?
        ORDER BY t.order_index, ts.start_seconds
        """,
        (story_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [(r["track_id"], r["speaker_label"], r["text"]) for r in rows]


def _chunk_by_track(lines: list[tuple[int, str, str]], max_chars: int) -> list[list[tuple[int, str, str]]]:
    """Groups whole tracks into chunks bounded by max_chars — never splits a track
    across two chunks unless that single track alone exceeds the budget."""
    track_groups = [(tid, list(items)) for tid, items in groupby(lines, key=lambda l: l[0])]

    chunks: list[list[tuple[int, str, str]]] = []
    current: list[tuple[int, str, str]] = []
    current_len = 0

    for track_id, track_lines in track_groups:
        track_len = sum(len(label) + len(text) + 4 for _, label, text in track_lines)
        if current and current_len + track_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        if track_len > max_chars:
            # Une seule piste dépasse déjà le budget à elle seule : la découper en
            # sous-chunks internes reste sûr (même diarization, labels toujours stables).
            if current:
                chunks.append(current)
                current = []
                current_len = 0
            sub: list[tuple[int, str, str]] = []
            sub_len = 0
            for line in track_lines:
                line_len = len(line[1]) + len(line[2]) + 4
                if sub and sub_len + line_len > max_chars:
                    chunks.append(sub)
                    sub = []
                    sub_len = 0
                sub.append(line)
                sub_len += line_len
            if sub:
                chunks.append(sub)
            continue
        current.extend(track_lines)
        current_len += track_len

    if current:
        chunks.append(current)
    return chunks


def _format_chunk(chunk: list[tuple[int, str, str]]) -> str:
    blocks = []
    for track_id, group in groupby(chunk, key=lambda l: l[0]):
        lines = "\n".join(f"[{label}] {text}" for _, label, text in group)
        blocks.append(f"=== PISTE {track_id} ===\n{lines}")
    return "\n\n".join(blocks)


async def identify_speakers_pending(story_id: int | None = None) -> dict:
    n_stories = n_errors = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = "SELECT id, title FROM stories WHERE status = 'grouped'"
        params: tuple = ()
        if story_id is not None:
            query += " AND id = ?"
            params = (story_id,)
        async with conn.execute(query, params) as cur:
            stories = await cur.fetchall()

        for story in stories:
            lines = await _get_story_lines(conn, story["id"])
            if not lines:
                continue

            chunks = _chunk_by_track(lines, CHUNK_CHARS_FOR_SPEAKER_ID)
            # (track_id, speaker_label) -> character_name — clé complète car un label
            # brut seul n'est pas unique à travers les pistes.
            mapping: dict[tuple[int, str], str] = {}
            known_characters: set[str] = set()
            chunk_failed = False

            for chunk_index, chunk in enumerate(chunks):
                transcript = _format_chunk(chunk)
                if known_characters:
                    known_block = (
                        "Personnages déjà identifiés dans les parties précédentes de CETTE "
                        "histoire (réutilise le même nom si l'un d'eux reprend la parole ici, "
                        "reconnaissable au contenu de son discours, PAS au label qui n'est "
                        "jamais le même d'une piste à l'autre) : " + ", ".join(sorted(known_characters))
                    )
                else:
                    known_block = "Aucun personnage identifié pour l'instant (tout début de l'histoire)."

                result = await llm.call_tool(
                    system=(
                        "Tu analyses un extrait du transcript diarizé d'un conte audio pour enfants, "
                        "découpé en plusieurs parties successives. Chaque partie contient une ou "
                        "plusieurs pistes audio, annoncées par '=== PISTE N ==='. Un label de "
                        "locuteur entre crochets (ex: [SPEAKER_00]) N'EST STABLE QU'AU SEIN DE SA "
                        "PISTE : le même label dans une autre piste peut désigner un personnage "
                        "totalement différent — ne présume JAMAIS qu'ils correspondent, base-toi "
                        "uniquement sur le contenu du discours pour reconnaître un personnage déjà "
                        "vu. Identifie qui parle derrière chaque label de chaque piste fournie. La "
                        "voix qui raconte l'histoire (hors dialogues) doit être étiquetée "
                        "\"Narrateur\". Indique le track_id exact de la piste pour chaque entrée. "
                        "Appelle identify_speakers."
                    ),
                    user=(
                        f"TITRE: {story['title']}\n\n{known_block}\n\n"
                        f"TRANSCRIPT — partie {chunk_index + 1}/{len(chunks)}:\n{transcript}"
                    ),
                    tool=_SPEAKER_MAP_TOOL,
                )
                speakers = result.get("speakers", [])
                if not speakers:
                    logger.error(
                        f"identify_speakers: histoire {story['id']}, aucune réponse LLM pour la "
                        f"partie {chunk_index + 1}/{len(chunks)}"
                    )
                    chunk_failed = True
                    continue
                for entry in speakers:
                    track_id = entry.get("track_id")
                    label = entry.get("speaker_label")
                    name = entry.get("character_name")
                    if not track_id or not label or not name:
                        continue
                    mapping[(track_id, label)] = name
                    known_characters.add(name)

            if not mapping:
                logger.error(f"identify_speakers: aucun locuteur identifié pour l'histoire {story['id']}")
                n_errors += 1
                continue
            if chunk_failed:
                # Histoire partiellement traitée : ne pas la marquer 'speakers_identified',
                # le prochain passage retentera les parties manquantes plutôt que de figer
                # un mapping locuteur incomplet.
                logger.error(f"identify_speakers: histoire {story['id']} incomplète, finalisation reportée")
                n_errors += 1
                continue

            for (track_id, label), name in mapping.items():
                await conn.execute(
                    "INSERT INTO speaker_map (story_id, track_id, speaker_label, character_name) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(story_id, track_id, speaker_label) DO UPDATE SET "
                    "character_name = excluded.character_name",
                    (story["id"], track_id, label, name),
                )
            await conn.execute("UPDATE stories SET status = 'speakers_identified' WHERE id = ?", (story["id"],))
            await conn.commit()
            n_stories += 1
            logger.info(
                f"identify_speakers: histoire {story['id']} ok ({len(mapping)} locuteurs, "
                f"{len(chunks)} partie(s), {len(known_characters)} personnages)"
            )

    result = {"stories_done": n_stories, "errors": n_errors}
    logger.info(f"identify_speakers: {result}")
    return result
