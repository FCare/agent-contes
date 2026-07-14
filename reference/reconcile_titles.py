"""Reconciles anthology sub-story titles against the real published table of
contents, retrieved via web search — split_stories.py only has each track's
first ~300 transcribed characters to guess a title from, which is often wrong
or generic (observed: the reused first-group row of "Contes de la rue Broca"
kept the raw folder name "La rue broca" as its title instead of a real story
title). Runs once per anthology (a group of sibling stories sharing the same
base folder, e.g. folder_path 'La rue broca', 'La rue broca#1', ... 'La rue
broca#12'), not once per story, to keep the web search cost bounded."""

import asyncio
import json
import logging
import os
import re

import aiosqlite
from rapidfuzz import fuzz, process

import chroma_store
import db
import llm
from . import web_search_client

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = float(os.environ.get("TITLE_RECONCILE_THRESHOLD", "70"))
# Au-delà d'un certain nombre d'histoires, le risque de confondre deux titres RÉELS mais
# voisins augmente fortement (observé en pratique sur les 18 Fables de La Fontaine : 'Le
# Lièvre et la Tortue' réassigné à 'Le Lièvre et les Grenouilles', deux fables distinctes
# partageant juste le mot 'lièvre') — le matching par similarité de texte seul n'a pas assez
# de signal pour ça sur un grand recueil aux titres courts et thématiquement proches. Limité
# aux petits recueils, où ce risque de collision reste faible (validé sur Rue Broca, 13
# histoires).
MAX_ANTHOLOGY_SIZE = int(os.environ.get("TITLE_RECONCILE_MAX_SIZE", "15"))

_PLAN_TOOL = [{
    "type": "function",
    "function": {
        "name": "plan_toc_search",
        "description": (
            "Décide la ou les requêtes de recherche web pour retrouver la table des "
            "matières officielle de ce recueil de contes"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "1 à 2 requêtes ciblées (ex: 'Pierre Gripari Contes de la rue Broca "
                        "liste des histoires'). Liste vide si le nom de dossier est trop "
                        "générique pour identifier une œuvre précise."
                    ),
                },
            },
            "required": ["queries"],
        },
    },
}]

_PLAN_SYSTEM_PROMPT = (
    "Tu prépares une recherche web pour retrouver la table des matières officielle d'un "
    "recueil de contes audio, à partir du nom brut de son dossier et, si connu, de son "
    "auteur littéraire. Le but : obtenir la vraie liste ordonnée des titres des histoires "
    "qui composent ce recueil, pour corriger des titres devinés à partir d'une "
    "transcription audio partielle. Une seule requête ciblée suffit généralement (titre du "
    "recueil + auteur + 'liste des histoires' ou 'table des matières'). Liste vide si le "
    "nom de dossier ne correspond visiblement à aucune œuvre identifiable."
)

_EXTRACT_TOOL = [{
    "type": "function",
    "function": {
        "name": "extract_table_of_contents",
        "description": (
            "Extrait la liste ordonnée des titres d'histoires d'un recueil à partir de "
            "rapports de recherche web"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Titres des histoires du recueil, dans l'ordre si connu. Liste vide "
                        "si les rapports ne permettent pas d'identifier une table des "
                        "matières fiable."
                    ),
                },
            },
            "required": ["titles"],
        },
    },
}]

_EXTRACT_SYSTEM_PROMPT = (
    "Tu extrais la table des matières d'un recueil de contes à partir de rapports de "
    "recherche web. RÈGLE ABSOLUE : n'inclus que des titres explicitement listés dans les "
    "rapports fournis — n'invente jamais un titre à partir de connaissances générales, et "
    "ne complète jamais une liste partielle avec des titres plausibles non confirmés par le "
    "texte. "
    "MÉFIANCE PARTICULIÈRE (observé en pratique, source d'erreurs) : un rapport peut "
    "provenir d'une thèse/mémoire universitaire (sommaire académique : 'INTRODUCTION', "
    "'I. LIRE LES FABLES', 'I.4.1. Le corbeau et le renard et la persuasion'...) ou d'un "
    "index bibliographique numéroté catalographiant des CENTAINES de contes traditionnels "
    "sans rapport avec CE recueil précis (ex: '202. La princesse-cane grise' dans un index "
    "général de contes russes). Ces sommaires/index NE SONT PAS la table des matières du "
    "recueil audio recherché même s'ils contiennent des mots proches — si le rapport a "
    "cette forme (numérotation académique du type 'I.4.1.', sections comme "
    "'INTRODUCTION'/'ANNEXES'/'BIBLIOGRAPHIE', ou une liste numérotée qui dépasse largement "
    "le nombre d'histoires attendu), réponds liste vide plutôt que d'en extraire des titres."
)


def _base_folder(folder_path: str) -> str:
    return re.sub(r"#\d+$", "", folder_path)


async def _find_anthology_groups(
    conn: aiosqlite.Connection, story_id: int | None,
) -> dict[tuple, list[dict]]:
    async with conn.execute(
        "SELECT id, title, author, folder_path, literary_author FROM stories "
        "WHERE status IN ('grouped', 'ready')"
    ) as cur:
        rows = [dict(r) for r in await cur.fetchall()]

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["author"], _base_folder(r["folder_path"]))
        groups.setdefault(key, []).append(r)

    # Un recueil a forcément >= 2 histoires (sinon rien à réconcilier).
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if story_id is not None:
        groups = {k: v for k, v in groups.items() if any(m["id"] == story_id for m in v)}
    return groups


async def _reconcile_group(
    conn: aiosqlite.Connection, author: str, base_folder: str, members: list[dict],
) -> int:
    if len(members) > MAX_ANTHOLOGY_SIZE:
        logger.info(
            f"reconcile_titles: {base_folder!r} ignoré ({len(members)} histoires > "
            f"{MAX_ANTHOLOGY_SIZE}, risque de collision entre titres proches trop élevé)"
        )
        return 0

    literary_author = next((m["literary_author"] for m in members if m.get("literary_author")), "")
    context = (
        f"NOM DE DOSSIER: {base_folder}\n"
        f"AUTEUR LITTÉRAIRE CONNU: {literary_author or '(inconnu)'}\n"
        f"NOMBRE D'HISTOIRES DÉTECTÉES: {len(members)}"
    )

    plan = await llm.call_tool(system=_PLAN_SYSTEM_PROMPT, user=context, tool=_PLAN_TOOL)
    queries = [q for q in (plan.get("queries") or []) if q.strip()][:2]
    if not queries:
        logger.info(f"reconcile_titles: aucune requête pertinente pour {base_folder!r}")
        return 0

    results = await asyncio.gather(*[web_search_client.search(q) for q in queries])
    reports = [(q, (r.get("report") or "").strip()) for q, r in zip(queries, results)]
    reports = [(q, r) for q, r in reports if r]
    if not reports:
        logger.info(f"reconcile_titles: pas de résultat web pour {base_folder!r}")
        return 0

    reports_block = "\n\n".join(f"=== REQUÊTE: {q} ===\n{r}" for q, r in reports)
    extracted = await llm.call_tool(
        system=_EXTRACT_SYSTEM_PROMPT, user=f"{context}\n\n{reports_block}", tool=_EXTRACT_TOOL,
    )
    canonical_titles = [t.strip() for t in (extracted.get("titles") or []) if t.strip()]
    if not canonical_titles:
        logger.info(f"reconcile_titles: aucune table des matières fiable extraite pour {base_folder!r}")
        return 0

    logger.info(
        f"reconcile_titles: {len(canonical_titles)} titre(s) canonique(s) pour "
        f"{base_folder!r}: {canonical_titles}"
    )

    async def _apply(m: dict, canonical_title: str) -> None:
        await conn.execute("UPDATE stories SET title = ? WHERE id = ?", (canonical_title, m["id"]))
        async with conn.execute(
            "SELECT short_summary, long_summary, keywords, total_duration_seconds "
            "FROM stories WHERE id = ?",
            (m["id"],),
        ) as cur:
            row = await cur.fetchone()
        keywords = json.loads(row["keywords"]) if row["keywords"] else []
        await db.sync_story_fts(
            conn, m["id"], canonical_title, row["short_summary"] or "", row["long_summary"] or "", keywords,
        )
        chroma_store.upsert_story_summary(
            m["id"], canonical_title, author, row["short_summary"] or "", row["long_summary"] or "",
            row["total_duration_seconds"] or 0.0,
        )
        if keywords:
            chroma_store.upsert_story_keywords(
                m["id"], canonical_title, author, keywords, row["total_duration_seconds"] or 0.0,
            )
        logger.info(f"reconcile_titles: histoire {m['id']} {m['title']!r} -> {canonical_title!r}")

    # Un titre canonique n'est utilisé qu'une seule fois (retiré de 'available' une fois
    # assigné) pour éviter que deux histoires distinctes matchent le même titre.
    available = {t: db.normalize_text(t) for t in canonical_titles}
    unmatched: list[dict] = []
    n_updated = 0
    for m in members:
        if not available:
            unmatched.append(m)
            continue
        norm_current = db.normalize_text(m["title"])
        match = process.extractOne(norm_current, available, scorer=fuzz.ratio, score_cutoff=MATCH_THRESHOLD)
        if not match:
            unmatched.append(m)
            continue
        canonical_title = match[2]
        del available[canonical_title]
        if canonical_title == m["title"]:
            continue
        await _apply(m, canonical_title)
        n_updated += 1

    # Repli par élimination : un titre resté générique (ex: la ligne de base réutilisée par
    # split_stories garde parfois le nom brut du dossier, cf. "La rue broca") ne ressemble
    # textuellement à AUCUN titre canonique et ne peut donc jamais matcher par score flou —
    # mais s'il ne reste exactement qu'un titre canonique inutilisé pour exactement une
    # histoire non assignée, c'est nécessairement la bonne correspondance.
    if len(unmatched) == 1 and len(available) == 1:
        m = unmatched[0]
        canonical_title = next(iter(available))
        logger.info(f"reconcile_titles: assignation par élimination pour l'histoire {m['id']}")
        await _apply(m, canonical_title)
        n_updated += 1

    return n_updated


async def reconcile_pending(story_id: int | None = None) -> dict:
    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        groups = await _find_anthology_groups(conn, story_id)

    n_groups = 0
    n_updated_total = 0
    for (author, base_folder), members in groups.items():
        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            try:
                n_updated = await _reconcile_group(conn, author, base_folder, members)
                await conn.commit()
            except Exception as e:
                logger.error(f"reconcile_titles: échec pour {base_folder!r}: {e}")
                n_updated = 0
        n_groups += 1
        n_updated_total += n_updated

    result = {"groups": n_groups, "titles_updated": n_updated_total}
    logger.info(f"reconcile_titles: {result}")
    return result
