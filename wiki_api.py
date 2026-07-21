"""Recherche sémantique pour le wiki statique (voir reference/build_wiki.py) :
réutilise chroma_store.py tel quel, aucun nouvel index — le wiki n'a pas
vocation à dupliquer la recherche de contes_tools.py, seulement à l'exposer
à un client HTTP simple (fetch JS depuis la page d'accueil du wiki, voir
wiki_theme/search.js)."""

import logging

from fastapi import APIRouter

import chroma_store
import db
from reference.build_wiki import _story_slug
from reference.summarize import PERIOD_SECONDS

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_LIMIT = 8


@router.get("/api/wiki/search")
async def search(q: str = "", limit: int = DEFAULT_LIMIT) -> dict:
    query = q.strip()
    if not query:
        return {"stories": [], "moments": []}

    semantic_stories = chroma_store.search_stories(query, n_results=limit)
    semantic_moments = chroma_store.search_moments(query, n_results=limit)

    story_ids = {r["metadata"]["story_id"] for r in semantic_stories} | {
        r["metadata"]["story_id"] for r in semantic_moments
    }
    # Titre relu depuis SQLite (pas les métadonnées Chroma, potentiellement figées
    # depuis le dernier embed — voir docs/known-issues.md) : le slug généré ici doit
    # correspondre EXACTEMENT à celui produit par build_wiki.py au même moment, qui
    # part toujours du titre SQLite actuel. Filtré sur status='ready' : Chroma peut
    # contenir des histoires 'missing'/'excluded' non nettoyées (docs/known-issues.md),
    # pour lesquelles aucune page wiki n'a été générée — un lien vers elles serait un 404.
    titles: dict[int, str] = {}
    for sid in story_ids:
        story = await db.get_story(sid)
        if story and story["status"] == "ready":
            titles[sid] = story["title"]

    stories = []
    for r in semantic_stories:
        sid = r["metadata"]["story_id"]
        title = titles.get(sid)
        if not title:
            continue
        stories.append({
            "story_id": sid,
            "title": title,
            "summary": r["content"],
            "score": r["score"],
            "url": f"/wiki/histoires/{_story_slug(title, sid)}/",
        })

    moments = []
    for r in semantic_moments:
        sid = r["metadata"]["story_id"]
        title = titles.get(sid)
        if not title:
            continue
        # Calculé depuis global_start_seconds plutôt que lu tel quel dans les
        # métadonnées : un résultat "segment_transcript" (une phrase précise, voir
        # chroma_store.upsert_segments) a period_index=-1 en métadonnées (n'appartient
        # pas lui-même à une période) — on cible alors l'ancre du chapitre qui le
        # contient, la page ne portant pas d'ancre par phrase individuelle.
        period_index = int(r["metadata"]["global_start_seconds"] // PERIOD_SECONDS)
        moments.append({
            "story_id": sid,
            "title": title,
            "excerpt": r["content"],
            "score": r["score"],
            "url": f"/wiki/histoires/{_story_slug(title, sid)}/#periode-{period_index}",
        })

    return {"stories": stories, "moments": moments}
