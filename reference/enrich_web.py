"""Enriches each story with real-world metadata found via agent-web-search:
the original literary author/date of the tale, and the real/corrected name
of the audio narrator — both distinct from stories.author, which is just the
raw, sometimes misspelled folder name (e.g. "Roman Boringher").

Rather than firing fixed query templates, an LLM first looks at what's
already known (summary, keywords, raw folder name, any prior enrichment) and
decides what's actually missing and worth a web search — it may plan zero,
one, or several targeted queries, or skip entirely if nothing looks missing."""

import asyncio
import json
import logging

import aiosqlite

import chroma_store
import db
import llm
from . import web_search_client

logger = logging.getLogger(__name__)

_PLAN_TOOL = [{
    "type": "function",
    "function": {
        "name": "plan_web_searches",
        "description": "Décide quelles recherches web combleraient les informations manquantes sur ce conte",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "0 à 3 requêtes de recherche web, chacune ciblée sur une info "
                        "manquante précise (ex: auteur littéraire original si inconnu, "
                        "identité réelle du narrateur si le nom de dossier ressemble à un "
                        "nom de personne potentiellement mal orthographié). Liste vide si "
                        "rien de plausible ne manque ou si le nom de dossier est déjà "
                        "générique ('Interprète inconnu' par ex., rien à vérifier)."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "1 phrase expliquant ce qui manque, ou pourquoi rien ne manque",
                },
            },
            "required": ["queries", "reasoning"],
        },
    },
}]

_PLAN_SYSTEM_PROMPT = (
    "Tu prépares l'enrichissement de la fiche d'un conte audio à partir de recherches "
    "web. On te donne ce qu'on sait déjà sur l'histoire. Détermine quelles recherches "
    "combleraient un vrai manque, sans jamais répéter une info déjà connue avec "
    "confiance. Deux angles typiques : (1) l'auteur littéraire original de l'histoire, "
    "si absent ; (2) l'identité réelle du narrateur de cette version audio, seulement "
    "si le nom de dossier fourni ressemble à un nom de personne propre (pas un terme "
    "générique) et pourrait être mal orthographié. N'invente pas de piste artificielle "
    "s'il n'y a rien de plausible à chercher."
)

_EXTRACT_TOOL = [{
    "type": "function",
    "function": {
        "name": "extract_literary_info",
        "description": "Extrait l'auteur original, la date de publication et le narrateur audio à partir de rapports de recherche web",
        "parameters": {
            "type": "object",
            "properties": {
                "original_author": {
                    "type": ["string", "null"],
                    "description": (
                        "Nom de l'auteur original de l'histoire (ex: 'Charles Perrault', "
                        "'Roald Dahl'), ou null si les résultats ne permettent pas de "
                        "l'identifier avec confiance. Ne jamais inventer ni deviner."
                    ),
                },
                "publication_year": {
                    "type": ["string", "null"],
                    "description": (
                        "Année ou période de publication originale (ex: '1697', 'XIXe "
                        "siècle'), ou null si inconnue. Ne jamais inventer ni deviner."
                    ),
                },
                "narrator_name": {
                    "type": ["string", "null"],
                    "description": (
                        "Nom correct (orthographe corrigée si besoin) de la personne qui "
                        "lit cette version audio précise, uniquement si confirmé par les "
                        "résultats (ex: dossier 'Roman Boringher' -> actrice 'Romane "
                        "Bohringer'). Null si non confirmé — ne jamais déduire du seul nom "
                        "de dossier sans confirmation dans les résultats."
                    ),
                },
                "notes": {
                    "type": "string",
                    "description": (
                        "1-2 phrases de contexte fiable trouvé dans les résultats ; "
                        "chaîne vide si rien de fiable."
                    ),
                },
            },
            "required": ["original_author", "publication_year", "narrator_name", "notes"],
        },
    },
}]

_EXTRACT_SYSTEM_PROMPT = (
    "Tu extrais des informations factuelles sur un conte à partir de rapports de "
    "recherche web, un par requête planifiée. RÈGLE ABSOLUE : si un rapport ne permet "
    "pas d'identifier clairement une information avec confiance, réponds null pour ce "
    "champ — ne devine jamais, n'utilise jamais tes connaissances générales à la place "
    "du rapport fourni. Un conte populaire/anonyme sans auteur identifiable, ou un "
    "narrateur non identifiable, sont des résultats normaux."
)


def _story_context(story: dict) -> str:
    keywords = json.loads(story["keywords"]) if story.get("keywords") else []
    return (
        f"TITRE: {story['title']}\n"
        f"NOM BRUT DU DOSSIER AUDIO: {story['author']}\n"
        f"RÉSUMÉ: {story.get('short_summary') or '(pas de résumé)'}\n"
        f"MOTS-CLÉS DÉJÀ CONNUS: {', '.join(keywords) if keywords else '(aucun)'}\n"
        f"AUTEUR LITTÉRAIRE DÉJÀ CONNU: {story.get('literary_author') or '(inconnu)'}\n"
        f"NARRATEUR DÉJÀ CONFIRMÉ: {story.get('narrator') or '(inconnu)'}"
    )


async def _enrich_one(conn: aiosqlite.Connection, story: dict) -> bool:
    plan = await llm.call_tool(
        system=_PLAN_SYSTEM_PROMPT,
        user=_story_context(story),
        tool=_PLAN_TOOL,
    )
    queries = [q for q in (plan.get("queries") or []) if q.strip()][:3]

    if not queries:
        await conn.execute("UPDATE stories SET literary_author = '', narrator = '' WHERE id = ?", (story["id"],))
        await conn.commit()
        return True

    results = await asyncio.gather(
        *[web_search_client.search(q, categories="general", detail_level=2) for q in queries]
    )
    reports = [(q, (r.get("report") or "").strip()) for q, r in zip(queries, results)]
    reports = [(q, r) for q, r in reports if r]
    all_sources = [s for r in results for s in (r.get("sources") or [])]

    if not reports:
        await conn.execute("UPDATE stories SET literary_author = '', narrator = '' WHERE id = ?", (story["id"],))
        await conn.commit()
        return True

    reports_block = "\n\n".join(f"=== REQUÊTE: {q} ===\n{r}" for q, r in reports)
    extracted = await llm.call_tool(
        system=_EXTRACT_SYSTEM_PROMPT,
        user=f"{_story_context(story)}\n\n{reports_block}",
        tool=_EXTRACT_TOOL,
    )
    if not extracted:
        return False

    def _clean(value) -> str:
        # Le LLM renvoie parfois la chaîne littérale "null" au lieu d'un vrai null JSON.
        value = (value or "").strip()
        return "" if value.lower() in ("null", "none", "n/a") else value

    original_author = _clean(extracted.get("original_author"))
    narrator_name = _clean(extracted.get("narrator_name"))
    publication_year = _clean(extracted.get("publication_year")) or None
    notes = extracted.get("notes") or ""
    literary_info = json.dumps(
        {"publication_year": publication_year, "notes": notes, "sources": all_sources},
        ensure_ascii=False,
    )

    await conn.execute(
        "UPDATE stories SET literary_author = ?, narrator = ?, literary_info = ? WHERE id = ?",
        (original_author, narrator_name, literary_info, story["id"]),
    )

    new_terms = [t for t in (original_author, narrator_name) if t]
    if new_terms:
        # Réutilise l'infra mots-clés existante (FTS + embedding dédié) plutôt que
        # d'ajouter un nouveau canal de recherche : ces noms deviennent des mots-clés
        # de plus, cherchables exactement comme les autres.
        async with conn.execute("SELECT keywords, short_summary, long_summary, author FROM stories WHERE id = ?",
                                 (story["id"],)) as cur:
            row = await cur.fetchone()
        keywords = json.loads(row["keywords"]) if row["keywords"] else []
        for term in new_terms:
            if term not in keywords:
                keywords.append(term)
        await conn.execute("UPDATE stories SET keywords = ? WHERE id = ?",
                            (json.dumps(keywords, ensure_ascii=False), story["id"]))
        await db.sync_story_fts(conn, story["id"], story["title"], row["short_summary"], row["long_summary"], keywords)
        chroma_store.upsert_story_keywords(
            story["id"], story["title"], row["author"], keywords, story["total_duration_seconds"],
        )

    await conn.commit()
    return True


async def enrich_pending(story_id: int | None = None, limit: int | None = None) -> dict:
    n_done = n_errors = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = (
            "SELECT id, title, author, short_summary, keywords, literary_author, narrator, "
            "total_duration_seconds FROM stories WHERE status = 'ready' AND literary_author IS NULL"
        )
        params: tuple = ()
        if story_id is not None:
            query += " AND id = ?"
            params = (story_id,)
        query += " ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            params = params + (limit,)
        async with conn.execute(query, params) as cur:
            stories = await cur.fetchall()

    for story in stories:
        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            try:
                ok = await _enrich_one(conn, dict(story))
            except Exception as e:
                logger.error(f"enrich_web: échec pour l'histoire {story['id']}: {e}")
                ok = False
        if ok:
            n_done += 1
            logger.info(f"enrich_web: histoire {story['id']} ok")
        else:
            n_errors += 1

    result = {"done": n_done, "errors": n_errors}
    logger.info(f"enrich_web: {result}")
    return result
