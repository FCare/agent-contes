"""Business logic behind each 'type' handled by the contes MQTT agent.
Each function takes the request payload dict and returns the structured
result payload to publish back on the result topic."""

import json
import logging
import os

import chroma_store
import db
import playlist

logger = logging.getLogger(__name__)

# Une histoire à moins de X secondes de sa fin est considérée comme terminée :
# reprendre la playlist repart du début plutôt que de rejouer les dernières secondes.
BOOKMARK_FINISH_THRESHOLD_SECONDS = float(os.environ.get("BOOKMARK_FINISH_THRESHOLD_SECONDS", "30"))

# Scores synthétiques donnés aux correspondances hors embeddings : un score manquant
# ("null") pouvant être interprété par un LLM consommateur comme "peu fiable" et ignoré,
# on leur donne toujours un score explicite plutôt que rien. Le lexical (FTS) reste
# volontairement dans la fourchette moyenne d'une similarité sémantique plausible — un
# terme rare/littéral qui matche doit apparaître, mais sans écraser systématiquement un
# meilleur résultat sémantique : il comble les trous que le sémantique rate, il ne les
# domine pas par défaut (voir _search_stories/_search_moments : jamais utilisé pour
# remplacer une entrée déjà trouvée par une autre voie, quelle qu'elle soit).
KEYWORD_MATCH_SCORE = 0.55
AUTHOR_MATCH_SCORE = 1.0

# Sous ce seuil de similarité, la classe thématique la plus proche n'est plus
# pertinente pour la requête — mieux vaut ne rien ajouter que du bruit.
THEME_CLASS_SIMILARITY_THRESHOLD = 0.3
THEME_CLASS_TOP_N = 2


async def _search_stories(query: str, top_k: int, min_minutes, max_minutes) -> list[dict]:
    def _duration_ok(seconds: float) -> bool:
        if min_minutes and seconds < min_minutes * 60:
            return False
        if max_minutes and seconds > max_minutes * 60:
            return False
        return True

    # Indexé par story_id : une même histoire peut ressortir par plusieurs voies
    # (auteur, sémantique, lexical) — on garde la meilleure, jamais un doublon.
    by_story: dict[int, dict] = {}

    # Auteur/narrateur : son nom vient du dossier, il n'apparaît presque jamais dans le
    # contenu de l'histoire — on le cherche donc directement dans la requête brute,
    # priorité maximale puisqu'un nom propre qui matche est un signal quasi certain.
    for m in await db.list_stories_by_author(query, limit=top_k):
        if not _duration_ok(m["total_duration_seconds"]):
            continue
        by_story[m["id"]] = {
            "story_id": m["id"],
            "title": m["title"],
            "author": m["author"],
            "duration_seconds": m["total_duration_seconds"],
            "summary": m["short_summary"],
            "score": AUTHOR_MATCH_SCORE,
            "match_type": "author",
        }

    semantic = chroma_store.search_stories(
        query,
        n_results=top_k,
        min_duration_seconds=min_minutes * 60 if min_minutes else None,
        max_duration_seconds=max_minutes * 60 if max_minutes else None,
    )
    for r in semantic:
        sid = r["metadata"]["story_id"]
        if sid in by_story:
            continue  # déjà trouvé par auteur — signal plus fort, on ne le dégrade pas
        by_story[sid] = {
            "story_id": sid,
            "title": r["metadata"]["title"],
            "author": r["metadata"]["author"],
            "duration_seconds": r["metadata"]["global_end_seconds"],
            "summary": r["content"],
            "score": r["score"],
            "match_type": "semantic",
        }

    # Fallback lexical : un détail littéral et rare ("un arbre à pain") peut être dilué
    # dans un vecteur sémantique ; une correspondance mot-à-mot le retrouve à coup sûr.
    # Comble les trous que les voies précédentes ratent — ne remplace jamais une entrée
    # déjà trouvée (auteur, sémantique, thème), même si son score synthétique est plus
    # élevé : sinon un mot générique de la requête écrase de bien meilleurs résultats.
    for r in await db.search_stories_fts(query, limit=top_k):
        if r["story_id"] in by_story:
            continue
        if not _duration_ok(r["total_duration_seconds"]):
            continue
        keywords = json.loads(r["keywords"]) if r["keywords"] else []
        # Le résumé court seul peut ne pas mentionner littéralement le terme recherché
        # (ex: "arbre à pain" n'apparaît que dans le résumé long / les mots-clés) — sans
        # ça, le LLM consommateur ne voit pas pourquoi ce résultat est pertinent.
        summary = f"{r['short_summary']}\n{r['long_summary']}"
        if keywords:
            summary += f"\nMots-clés: {', '.join(keywords)}"
        by_story[r["story_id"]] = {
            "story_id": r["story_id"],
            "title": r["title"],
            "author": r["author"],
            "duration_seconds": r["total_duration_seconds"],
            "summary": summary,
            "score": KEYWORD_MATCH_SCORE,
            "match_type": "keyword",
        }

    # Classes thématiques "libres" (découvertes depuis le catalogue, pas un vocabulaire
    # fixé à l'avance) : une requête vague ("des histoires de pirates") peut ne matcher
    # aucun mot-clé littéral ni thème précis en embedding phrase-à-phrase, mais tomber
    # near d'une classe entière — on ajoute alors ses membres en complément, jamais en
    # écrasant un résultat déjà trouvé par une voie plus fiable.
    for cls in chroma_store.search_theme_classes(query, n_results=THEME_CLASS_TOP_N):
        if cls["score"] < THEME_CLASS_SIMILARITY_THRESHOLD:
            continue
        class_id = cls["metadata"]["class_id"]
        for m in await db.stories_by_theme_class(class_id, limit=top_k):
            if m["id"] in by_story:
                continue
            if not _duration_ok(m["total_duration_seconds"]):
                continue
            by_story[m["id"]] = {
                "story_id": m["id"],
                "title": m["title"],
                "author": m["author"],
                "duration_seconds": m["total_duration_seconds"],
                "summary": m["short_summary"],
                "score": cls["score"],
                "match_type": "theme",
            }

    results = sorted(by_story.values(), key=lambda r: r["score"], reverse=True)
    return results[:top_k]


async def _search_moments(query: str, story_id: int | None, top_k: int) -> list[dict]:
    semantic = chroma_store.search_moments(query, story_id=story_id, n_results=top_k)

    # Indexé par (story_id, global_start_seconds) : le sémantique est ajouté en premier,
    # le lexical ne comble ensuite que les trous qu'il rate (voir boucle plus bas).
    by_moment: dict[tuple, dict] = {}
    for r in semantic:
        key = (r["metadata"]["story_id"], r["metadata"]["global_start_seconds"])
        by_moment[key] = {
            "story_id": key[0],
            "global_start_seconds": key[1],
            "excerpt": r["content"],
            "score": r["score"],
            "match_type": "semantic",
        }

    for r in await db.search_moments_fts(query, story_id=story_id, limit=top_k):
        key = (r["story_id"], r["global_start_seconds"])
        if key in by_moment:
            continue  # comble les trous du sémantique, ne l'écrase jamais
        # Le mot cherché peut être dans le texte brut diarizé plutôt que dans le
        # mini-résumé de la période — inclure les deux pour que ce soit visible.
        excerpt = f"{r['summary_text']}\n{r['raw_transcript_text']}"
        by_moment[key] = {
            "story_id": key[0],
            "global_start_seconds": key[1],
            "excerpt": excerpt,
            "score": KEYWORD_MATCH_SCORE,
            "match_type": "keyword",
        }

    results = sorted(by_moment.values(), key=lambda r: r["score"], reverse=True)
    return results[:top_k]


async def search_contes(args: dict) -> dict:
    """Point d'entrée de recherche unique : un seul champ query (thème, titre, nom
    d'auteur/narrateur, détail précis, ou un mélange) — l'appelant n'a pas à savoir/
    deviner quel type de question c'est ni quel outil appeler. On renvoie toujours à
    la fois des histoires candidates ET des passages précis correspondants, pour ne
    jamais rater une réponse simplement parce qu'un LLM appelant a choisi de chercher
    "une histoire" plutôt qu'un "détail" (ou l'inverse)."""
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "query requis"}

    story_id = args.get("story_id")
    min_minutes = args.get("min_duration_minutes")
    max_minutes = args.get("max_duration_minutes")
    stories_top_k = args.get("stories_top_k", 5)
    moments_top_k = args.get("moments_top_k", 8)

    # Si l'histoire est déjà connue (story_id fourni), pas la peine de rechercher
    # quelle histoire choisir — uniquement les passages précis, restreints à celle-ci.
    stories = [] if story_id else await _search_stories(query, stories_top_k, min_minutes, max_minutes)
    moments = await _search_moments(query, story_id, moments_top_k)

    return {"stories": stories, "moments": moments}


# Cap dur par appel, quelle que soit la tranche demandée : sur un catalogue de ~300
# histoires, un appel sans tranche (ou une tranche trop large) a fait déborder le
# contexte du LLM appelant et tronqué sa réponse en plein milieu (finish_reason=length,
# ~27k tokens de prompt rien que pour les résumés). On ne renvoie donc jamais le résumé
# ici (juste titre/auteur/durée) ET on plafonne la taille d'une page à 15 — jamais
# silencieusement : 'truncated' indique explicitement qu'il reste des histoires au-delà.
LIST_STORIES_PAGE_SIZE = 15


async def list_stories(args: dict) -> dict:
    """Parcours du catalogue sans requête précise : l'utilisateur navigue plutôt que de
    chercher un titre/thème donné ('donne-moi la liste des contes disponibles'). Sans
    range_start/range_end, part du début du catalogue (trié par titre) ; l'appelant peut
    avancer via ces deux paramètres pour parcourir le catalogue page par page (15 à la
    fois) ; min/max_duration_minutes, age_range, mood optionnels pour filtrer."""
    min_minutes = args.get("min_duration_minutes")
    max_minutes = args.get("max_duration_minutes")
    min_seconds = min_minutes * 60 if min_minutes else None
    max_seconds = max_minutes * 60 if max_minutes else None
    age_range = args.get("age_range")
    mood = args.get("mood")

    total = await db.count_ready_stories(min_duration_seconds=min_seconds, max_duration_seconds=max_seconds,
                                          age_range=age_range, mood=mood)
    range_start = max(args.get("range_start") or 1, 1)
    range_end = args.get("range_end")
    if range_end is not None and range_end < range_start:
        # Vu en pratique : l'appelant avance range_start pour la page suivante mais
        # renvoie par erreur le range_end de la page précédente (ex: range_start=16,
        # range_end=15 laissé inchangé) — une tranche incohérente doit retomber sur la
        # taille de page par défaut plutôt que de produire silencieusement 0 résultat.
        logger.warning(f"list_stories: range_end ({range_end}) < range_start ({range_start}), ignoré")
        range_end = None
    offset = range_start - 1
    requested = (range_end - range_start + 1) if range_end else LIST_STORIES_PAGE_SIZE
    limit = min(requested, LIST_STORIES_PAGE_SIZE)
    stories = await db.sample_stories(limit=limit, offset=offset,
                                       min_duration_seconds=min_seconds, max_duration_seconds=max_seconds,
                                       age_range=age_range, mood=mood)
    actual_range_end = range_start + len(stories) - 1 if stories else range_start
    return {
        "total_stories": total,
        "range_start": range_start,
        "range_end": actual_range_end,
        "truncated": actual_range_end < total,
        "stories": [
            {
                "story_id": s["id"],
                "title": s["title"],
                "author": s["author"],
                "duration_seconds": s["total_duration_seconds"],
            }
            for s in stories
        ],
    }


async def list_themes(args: dict) -> dict:
    """Liste les classes thématiques découvertes dans le catalogue (voir
    reference/classify.py) — sert à répondre à 'quels thèmes tu as ?' ou à proposer des
    pistes à l'utilisateur avant une recherche par thème."""
    classes = await db.get_theme_classes()
    return {"themes": [{"label": c["label"], "description": c["description"]} for c in classes]}


async def _resolve_story_id(args: dict) -> tuple[int | None, dict | None]:
    """Renvoie (story_id, None) si résolu, sinon (None, {"error": ...}). Accepte un
    entier direct, ou une chaîne de titre approximative — garde-fou pour le cas où
    l'appelant devine un titre à la place d'un vrai story_id plutôt que d'appeler
    search_contes d'abord (observé en pratique : 'story_id': 'le petit prince')."""
    story_id = args.get("story_id")
    if story_id is None:
        return None, {"error": "story_id requis"}
    if isinstance(story_id, int):
        return story_id, None
    resolved = await db.find_story_id_by_title(str(story_id))
    if resolved is not None:
        return resolved, None
    return None, {"error": "histoire introuvable"}


async def story_details(args: dict) -> dict:
    story_id, error = await _resolve_story_id(args)
    if error:
        return error
    story = await db.get_story(story_id)
    if not story:
        return {"error": "histoire introuvable"}
    periods = await db.get_periods_for_story(story_id)
    bookmark = await db.get_bookmark(story_id)
    return {
        "story_id": story["id"],
        "title": story["title"],
        "author": story["author"],
        "long_summary": story["long_summary"],
        "total_duration_seconds": story["total_duration_seconds"],
        "periods": [
            {"start_seconds": p["global_start_seconds"], "end_seconds": p["global_end_seconds"],
             "summary": p["summary_text"]}
            for p in periods
        ],
        "bookmark_position_seconds": bookmark["position_seconds"] if bookmark else None,
    }


async def get_playlist(args: dict) -> dict:
    story_id, error = await _resolve_story_id(args)
    if error:
        return error
    story = await db.get_story(story_id)
    if not story:
        return {"error": "histoire introuvable"}
    tracks = await db.get_tracks_for_story(story_id)
    if not tracks:
        return {"error": "aucune piste pour cette histoire"}

    target_seconds = args.get("start_at_seconds")
    if args.get("resume"):
        bookmark = await db.get_bookmark(story_id)
        if bookmark:
            remaining = story["total_duration_seconds"] - bookmark["position_seconds"]
            if remaining <= BOOKMARK_FINISH_THRESHOLD_SECONDS:
                await db.delete_bookmark(story_id)
                target_seconds = 0
            else:
                target_seconds = bookmark["position_seconds"]
        else:
            target_seconds = 0
    if target_seconds is None:
        target_seconds = 0

    start_index, start_offset = playlist._resolve(tracks, target_seconds)
    return {
        "story_id": story_id,
        "title": story["title"],
        "tracks": [
            {"order_index": t["order_index"], "url": f"/stream/{t['id']}", "duration_seconds": t["duration_seconds"]}
            for t in tracks
        ],
        "start_index": start_index,
        "start_offset_seconds": start_offset,
    }


async def save_bookmark(args: dict) -> dict:
    position_seconds = args.get("position_seconds")
    if position_seconds is None:
        return {"error": "position_seconds requis"}
    story_id, error = await _resolve_story_id(args)
    if error:
        return error
    await db.save_bookmark(story_id, position_seconds)
    return {"ok": True}


_HANDLERS = {
    "search_contes": search_contes,
    "list_stories": list_stories,
    "list_themes": list_themes,
    "story_details": story_details,
    "get_playlist": get_playlist,
    "save_bookmark": save_bookmark,
}


# Champs numériques attendus par au moins un handler ci-dessus. Le schéma d'outil partagé
# entre agents (mqtt_step._send_write_tool_update) n'exprime que des descriptions texte,
# pas de types JSON par champ — le LLM appelant sérialise donc parfois un nombre en chaîne
# ("16" plutôt que 16), ce qui fait planter toute comparaison/arithmétique en aval avec un
# TypeError peu explicite (déjà observé : range_start/range_end passés en str). On corrige
# ici, une seule fois, avant que args n'atteigne un handler quelconque, plutôt que de
# dupliquer une conversion défensive dans chacun.
_NUMERIC_ARG_KEYS = {
    "story_id", "min_duration_minutes", "max_duration_minutes", "stories_top_k",
    "moments_top_k", "range_start", "range_end", "start_at_seconds", "position_seconds",
}


def _coerce_numeric_args(args: dict) -> dict:
    coerced = dict(args)
    for key in _NUMERIC_ARG_KEYS & coerced.keys():
        value = coerced[key]
        if isinstance(value, str):
            try:
                coerced[key] = int(value)
            except ValueError:
                try:
                    coerced[key] = float(value)
                except ValueError:
                    logger.warning(f"contes_tools: valeur non numérique pour {key!r}: {value!r}, laissée telle quelle")
    return coerced


async def dispatch(req_type: str, args: dict) -> dict:
    # Le champ 'type' est censé être obligatoire, mais le LLM appelant l'omet souvent
    # malgré la consigne explicite (constaté en pratique sur environ un quart des
    # appels) — plutôt que de renvoyer une erreur qui pousse le modèle à parfois
    # halluciner une réponse plausible, on déduit un type par défaut raisonnable à
    # partir de ce qui EST fourni : une vraie recherche si 'query' est présent, sinon
    # un simple parcours du catalogue (list_stories n'a aucun champ obligatoire, donc
    # ne peut jamais échouer de la même façon).
    if not req_type:
        req_type = "search_contes" if (args.get("query") or "").strip() else "list_stories"
        logger.warning(f"contes_tools: 'type' manquant, déduit à {req_type!r} depuis les autres paramètres: {args}")

    handler = _HANDLERS.get(req_type)
    if not handler:
        return {"error": f"type inconnu: {req_type}. Disponibles: {', '.join(_HANDLERS)}"}
    try:
        return await handler(_coerce_numeric_args(args))
    except Exception as e:
        logger.error(f"contes_tools: {req_type} a échoué: {e}")
        return {"error": str(e)}
