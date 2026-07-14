import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from nexus_client import NexusClient

import contes_tools
import db
import playlist
from reference import classify
from reference import pipeline as pipeline_stage
from reference import scan as scan_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VK_URL = os.environ["VK_URL"]
MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
SERVICE_USERNAME = os.environ["MQTT_SERVICE_USERNAME"]
SERVICE_API_KEY = os.environ["MQTT_SERVICE_API_KEY"]
CONTES_ROOT = Path(os.environ.get("CONTES_ROOT", "/contes"))

AGENT_NAME = "contes"
_subscribed_sessions: set[str] = set()
DAILY_ORPHAN_CHECK_HOUR = int(os.environ.get("DAILY_ORPHAN_CHECK_HOUR", "4"))  # heure locale Europe/Paris

app = FastAPI(title="contes-agent")


async def _daily_catalog_sync_loop() -> None:
    """Chaque jour vers DAILY_ORPHAN_CHECK_HOUR (heure de Paris), synchronise le catalogue
    avec le disque dans les deux sens :
    - nouveaux contes : pipeline complet (scan, durée, transcription, découpage,
      identification des locuteurs, résumé, embeddings) sur tout ce qui n'a pas encore
      été traité — only_new=True pour ne pas retraiter ce qui l'est déjà ;
    - contes supprimés : marqués 'missing' (voir reference.scan.mark_missing), pour
      qu'ils disparaissent des recherches sans perdre leurs données si le disque
      réapparaît (ex: point de montage temporairement indisponible)."""
    tz = ZoneInfo("Europe/Paris")
    while True:
        now = datetime.now(tz)
        next_run = now.replace(hour=DAILY_ORPHAN_CHECK_HOUR, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        await asyncio.sleep((next_run - now).total_seconds())
        logger.info("Synchronisation quotidienne du catalogue de contes...")
        try:
            await pipeline_stage.run(stage="all", only_new=True, story_id=None, limit=None)
            logger.info("Traitement des nouveaux contes terminé")
        except Exception as e:
            logger.error(f"Traitement des nouveaux contes échoué: {e}")
        try:
            result = await scan_stage.mark_missing()
            logger.info(f"Vérification orphelins terminée: {result}")
        except Exception as e:
            logger.error(f"Vérification orphelins échouée: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


_MEDIA_TYPES = {".mp3": "audio/mpeg", ".ogg": "audio/ogg"}


@app.get("/stream/{track_id}")
async def stream_track(track_id: int):
    track = await db.get_track(track_id)
    if not track:
        raise HTTPException(404, "Piste introuvable")
    abs_path = CONTES_ROOT / track["file_path"]
    if not abs_path.is_file():
        raise HTTPException(404, "Fichier introuvable sur disque")
    media_type = _MEDIA_TYPES.get(abs_path.suffix.lower(), "application/octet-stream")
    return FileResponse(abs_path, media_type=media_type)


@app.get("/playlist/{story_id}")
async def debug_playlist(story_id: int, request: Request, at: float = 0):
    """Debug route exercising playlist.py directly, outside the MQTT flow."""
    story = await db.get_story(story_id)
    if not story:
        raise HTTPException(404, "Histoire introuvable")
    tracks = await db.get_tracks_for_story(story_id)
    if not tracks:
        raise HTTPException(404, "Aucune piste pour cette histoire")

    start_index, start_offset = playlist._resolve(tracks, at)
    base = str(request.base_url)
    return {
        "story_id": story_id,
        "title": story["title"],
        "tracks": [
            {"order_index": t["order_index"], "url": f"{base}stream/{t['id']}",
             "duration_seconds": t["duration_seconds"]}
            for t in tracks
        ],
        "start_index": start_index,
        "start_offset_seconds": start_offset,
    }


# ---------------------------------------------------------------------------
# MQTT agent
# ---------------------------------------------------------------------------

_REQUEST_DESCRIPTION = (
    "Catalogue de contes audio pour enfants — chercher une histoire ou un passage précis, "
    "obtenir un résumé, démarrer la lecture, ou sauvegarder la position. Types : "
    "search_contes → point d'entrée UNIQUE pour toute recherche (query obligatoire = tous "
    "les mots pertinents : titre, thème/ambiance, auteur/narrateur ET/OU détail recherché "
    "dans le contenu, ex: 'la recette de la potion dans George Bouillon', 'le moment où le "
    "loup arrive') — pas besoin de distinguer avant d'appeler, renvoie à la fois histoires "
    "candidates ET passages précis ; story_id optionnel pour restreindre à une histoire ; "
    "min/max_duration_minutes optionnels. "
    "list_stories → UNIQUEMENT pour parcourir sans thème ni critère précis ('la liste des "
    "contes', PAS pour une demande avec un thème, utiliser search_contes dans ce cas) ; "
    "page de 15 max triée par titre, range_start pour paginer (jamais range_end en même "
    "temps, calculé automatiquement) ; min/max_duration_minutes, "
    f"age_range ({', '.join(classify.AGE_RANGES)}), mood ({', '.join(classify.MOOD_TAGS)}) "
    "optionnels. Si 'toutes les histoires' sans aucun critère, demande d'abord thème/âge/"
    "ambiance/durée (voir list_themes) plutôt que d'appeler à l'aveugle. À l'oral, ne cite "
    "par défaut que 3 à 5 exemples et le total, PAS toute la liste — SAUF demande EXPLICITE "
    "de liste complète/entière, auquel cas énumérer vraiment tout (enchaîner range_start "
    "croissant si 'truncated' jusqu'à couvrir 'total_stories'). "
    "list_themes → classes thématiques du catalogue (aucun paramètre) — utile pour 'quels "
    "thèmes as-tu ?' ou pour proposer des pistes avant une recherche par thème. "
    "story_details → résumé complet et découpage par période (story_id). "
    "get_playlist → à appeler une fois l'histoire choisie pour démarrer sa lecture "
    "(story_id ; resume=true pour reprendre où on s'était arrêté ; ou start_at_seconds pour "
    "un instant précis, ex: obtenu via search_contes). "
    "save_bookmark → sauvegarder la position actuelle (story_id, position_seconds)."
)

_RESULT_DESCRIPTION = (
    "IMPORTANT pour la réponse orale, quel que soit le type de requête : quand tu "
    "présentes une histoire trouvée, cite TOUJOURS son titre exact tel qu'il apparaît "
    "dans 'title' — ne le remplace jamais par une paraphrase du contenu ou du résumé "
    "('une histoire avec un géant et une enfant' au lieu de 'Le BGG'). Le titre vient "
    "TOUJOURS en premier ; tu peux ajouter une courte accroche après, mais jamais à sa "
    "place. "
    "Résultat de la requête contes. search_contes → deux champs : 'stories' (histoires "
    "candidates) et 'moments' (passages précis, dans l'histoire ou dans tout le "
    "catalogue), triés par pertinence décroissante — regarder les DEUX, la réponse à "
    "une question de détail (ex: une recette, un événement précis) est presque toujours "
    "dans 'moments', pas dans le résumé général d'une entrée de 'stories'. Chaque "
    "résultat a un 'match_type' : 'author' (narrateur/auteur reconnu), 'semantic' "
    "(similarité de sens), 'keyword' (mot littéral de la requête) ou 'theme' (proche "
    "d'une classe thématique du catalogue) — les quatre sont valides, aucun n'est de "
    "moindre qualité. Si les deux champs sont vides, alors seulement il n'y a aucune "
    "correspondance. "
    "list_stories → 'stories' (au plus 15, titre/auteur/durée seulement), "
    "'total_stories' (nombre réel total pour les critères donnés), 'range_start'/"
    "'range_end' (bornes effectivement renvoyées), 'truncated' (true s'il reste des "
    "histoires au-delà de 'range_end') — par défaut ne pas toutes les citer à l'oral, "
    "en nommer 3 à 5 en exemple et proposer de préciser un critère (search_contes) ou "
    "de demander la page suivante ; SAUF si l'utilisateur a explicitement demandé la "
    "liste complète, auquel cas toutes les citer (en enchaînant les pages si "
    "'truncated' est true, voir ci-dessus). "
    "list_themes → 'themes' (label + description de chaque classe thématique du "
    "catalogue) — s'en servir pour suggérer des pistes concrètes à l'utilisateur. "
    "story_details → résumé, durée, découpage par période. "
    "get_playlist → tracks (liste ordonnée d'URLs de pistes à streamer en HTTP), "
    "start_index (index de la piste de départ dans 'tracks'), start_offset_seconds "
    "(temps de départ dans cette piste) — à transmettre tel quel au lecteur audio local. "
    "save_bookmark → {ok: true}."
)

_REQUEST_FORMAT = {
    "type": "search_contes | list_stories | list_themes | story_details | get_playlist | save_bookmark",
    "query": "(search_contes) tous les mots pertinents : titre, thème, auteur/narrateur, et/ou détail recherché",
    "story_id": "(search_contes, optionnel, restreint la recherche de passages à cette histoire ; "
                "story_details/get_playlist/save_bookmark requis)",
    "min_duration_minutes": "(search_contes ou list_stories, optionnel)",
    "max_duration_minutes": "(search_contes ou list_stories, optionnel)",
    "age_range": f"(list_stories, optionnel) un parmi : {', '.join(classify.AGE_RANGES)}",
    "mood": f"(list_stories, optionnel) un parmi : {', '.join(classify.MOOD_TAGS)}",
    "range_start": "(list_stories, optionnel, défaut 1) première histoire de la page (triée par titre)",
    "range_end": "(list_stories, optionnel, défaut = range_start + 14) dernière histoire de la page, page limitée à 15",
    "resume": "(get_playlist, optionnel) reprendre à la dernière position",
    "start_at_seconds": "(get_playlist, optionnel) temps de départ explicite",
    "position_seconds": "(save_bookmark, requis)",
}


async def on_user_connected(topic: str, payload) -> None:
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    password = payload.get("password")
    session_id = payload.get("session_id")
    private_topics = payload.get("private_topics", [])

    if not username or not password or not session_id:
        return

    agent_topics_topic = None
    for entry in private_topics:
        for t in entry.get("topics", []):
            if t["topic"].endswith("/agent_topics"):
                agent_topics_topic = t["topic"]
                break

    if not agent_topics_topic:
        logger.warning(f"[{username}] agent_topics introuvable, skip")
        return

    request_topic = f"users/{username}/{session_id}/contes/request"
    result_topic = f"users/{username}/{session_id}/contes/result"

    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    await nexus.publish(agent_topics_topic, [{
        "agent": AGENT_NAME,
        "topics": [
            {
                "topic": request_topic,
                "description": _REQUEST_DESCRIPTION,
                "access": "write",
                "response_topic": result_topic,
                "format": _REQUEST_FORMAT,
            },
            {
                "topic": result_topic,
                "description": _RESULT_DESCRIPTION,
                "access": "read",
                "format": {"results": "[...]", "tracks": "[...]"},
            },
        ],
    }])
    logger.info(f"[{username}/{session_id}] Topics contes déclarés")

    if session_id in _subscribed_sessions:
        return
    _subscribed_sessions.add(session_id)

    async def on_contes_request(t: str, p) -> None:
        if not isinstance(p, dict):
            return
        req_type = p.get("type", "").strip()
        logger.info(f"[{username}] Requête contes: {p}")
        result = await contes_tools.dispatch(req_type, p)
        reply_to = p.get("reply_to", result_topic)
        await nexus.publish(reply_to, result)
        logger.info(f"[{username}] Réponse contes publiée sur {reply_to}")

    nexus.subscribe(request_topic, on_contes_request)
    nexus.start_listening()
    logger.info(f"[{username}/{session_id}] Abonné à {request_topic}")


async def main() -> None:
    await db.init_db()
    asyncio.create_task(_daily_catalog_sync_loop())

    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.start_listening()
    logger.info("Agent contes démarré")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
