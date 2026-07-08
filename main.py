import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from nexus_client import NexusClient

import contes_tools
import db
import playlist
from reference import classify

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

app = FastAPI(title="contes-agent")


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
    "Catalogue de contes audio pour enfants. Utiliser pour toute demande liée aux "
    "contes/histoires à raconter : en chercher un, retrouver un détail/passage précis "
    "dans un ou tous les contes, obtenir le résumé complet d'une histoire, démarrer sa "
    "lecture, ou sauvegarder où on s'est arrêté. Types disponibles: "
    "search_contes → point d'entrée UNIQUE pour toute recherche, qu'il s'agisse de "
    "trouver une histoire ou d'un détail précis dedans (query obligatoire) : mets "
    "simplement tous les mots pertinents de la demande dans query — titre, thème/"
    "ambiance, nom de narrateur/auteur, ET/OU détail factuel recherché dans le contenu "
    "('la recette de la potion dans George Bouillon', 'le moment où le loup arrive', "
    "'les contes de Richard Bohringer', 'une histoire de sorcière') — inutile de "
    "distinguer ces cas avant d'appeler, l'agent cherche sur toutes les pistes à la "
    "fois (y compris par similarité avec les classes thématiques du catalogue) et "
    "renvoie à la fois des histoires candidates ET des passages précis correspondants ; "
    "ajouter story_id si l'histoire est déjà choisie pour restreindre la recherche de "
    "passages à celle-ci ; min/max_duration_minutes optionnels ; "
    "list_stories → à utiliser UNIQUEMENT quand l'utilisateur veut parcourir/découvrir "
    "le catalogue sans thème ni critère précis ('donne-moi la liste des contes', "
    "'qu'est-ce que tu as comme histoires ?') — PAS pour une demande avec un thème "
    "(utiliser search_contes dans ce cas) ; chaque appel renvoie une page d'au plus 15 "
    "histoires (triées par titre), jamais plus ; range_start (nombre entier) optionnel "
    "pour avancer page par page plutôt que de repartir du début à chaque fois (ex: "
    "range_start=16 pour la page suivante après une première page 1-15) — NE PAS "
    "fournir range_end en même temps que range_start pour une page suivante, il est "
    "calculé automatiquement (15 histoires à partir de range_start) ; min/max_duration_minutes, age_range "
    f"({', '.join(classify.AGE_RANGES)}), mood ({', '.join(classify.MOOD_TAGS)}) "
    "optionnels pour filtrer. Si l'utilisateur demande 'toutes les histoires' ou "
    "équivalent sans aucun critère, il vaut mieux d'abord lui demander s'il a un thème, "
    "un âge, une ambiance ou une durée en tête (voir list_themes) plutôt que d'appeler "
    "list_stories à l'aveugle. IMPORTANT pour la réponse orale : par défaut, ne PAS "
    "énumérer toutes les histoires reçues une par une — n'en citer que 3 à 5 à titre "
    "d'exemple, mentionner le nombre total, et proposer de préciser un critère ou de "
    "voir la suite. EXCEPTION : si l'utilisateur demande EXPLICITEMENT la liste "
    "complète/entière/toutes les histoires ('donne-moi la liste complète', 'liste-les "
    "toutes', 'je veux tout voir'), les énumérer réellement toutes — si 'truncated' est "
    "true, enchaîner les appels list_stories avec range_start croissant (16, 31, ...) "
    "jusqu'à couvrir 'total_stories' avant de répondre, plutôt que de s'arrêter à la "
    "première page. "
    "list_themes → liste les classes thématiques découvertes dans le catalogue (aucun "
    "paramètre) — utile pour répondre à 'quels thèmes as-tu ?' ou pour proposer des "
    "pistes concrètes à l'utilisateur avant une recherche par thème ; "
    "story_details → résumé complet et découpage par période d'une histoire (story_id) ; "
    "get_playlist → à appeler une fois l'histoire choisie pour démarrer sa lecture "
    "(story_id ; resume=true pour reprendre où on s'était arrêté, ou recommencer au début si "
    "l'histoire était presque terminée ; ou start_at_seconds pour démarrer à un instant "
    "précis, ex: obtenu via search_contes) ; "
    "save_bookmark → sauvegarder la position de lecture actuelle (story_id, position_seconds)."
)

_RESULT_DESCRIPTION = (
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

    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.start_listening()
    logger.info("Agent contes démarré")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
