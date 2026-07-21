import logging
import os
from datetime import datetime, timezone

import aiosqlite
import numpy as np

import db
import llm
from reference import speaker_voice_eval as voice_eval

logger = logging.getLogger(__name__)

# Seuils d'escalade utilisés quand une identification échoue à atteindre une confiance
# haute : on suppose alors que le cluster mélange plusieurs narrateurs et on le re-scinde
# à un seuil plus strict, avant de refaire une identification par sous-groupe. Validé
# manuellement sur 2 clusters mal fusionnés à 0.65 (voir speaker_voice_eval.CLUSTER_THRESHOLD) :
# la scission à 0.75 puis 0.80 sépare proprement des narrateurs mélangés, sans jamais
# re-scinder les clusters déjà corrects (ceux-là obtiennent 'haute' dès le premier passage).
ESCALATION_THRESHOLDS = [
    float(t) for t in os.environ.get("NARRATOR_ESCALATION_THRESHOLDS", "0.75,0.80").split(",")
]

IDENTIFY_TOOL = [{
    "type": "function",
    "function": {
        "name": "identify_narrator",
        "description": "Déduit l'identité probable du narrateur à partir des métadonnées des histoires où sa voix apparaît",
        "parameters": {
            "type": "object",
            "properties": {
                "inferred_name": {"type": "string", "description": "Nom ou identité probable, ou chaîne vide si aucun indice exploitable"},
                "confidence": {"type": "string", "enum": ["faible", "moyenne", "haute"]},
                "is_professional": {"type": "boolean", "description": "true si comédien/narrateur professionnel identifiable, false si amateur/familial"},
                "reasoning": {"type": "string", "description": "Justification courte, 1-2 phrases"},
            },
            "required": ["inferred_name", "confidence", "is_professional", "reasoning"],
        },
    },
}]

# Convention de dossiers observée dans le catalogue, expliquée explicitement au LLM plutôt
# que laissée à déduire : le dossier de premier niveau désigne souvent un narrateur par
# défaut (label choisi par la personne qui a organisé sa collection), et un sous-dossier
# intermédiaire nomme un narrateur différent quand il y en a un. Sans cette explication le
# LLM confond parfois le TITRE d'un conte avec un nom de narrateur (ex: "Georges_Bouillon",
# titre d'un Roald Dahl, pris pour le nom du narrateur) — d'où la mise en garde dédiée.
SYSTEM_PROMPT = (
    "Tu es un expert en catalogues audio. La collection est organisée en dossiers : le dossier "
    "de premier niveau est souvent un LABEL choisi par la personne qui a organisé la collection "
    "pour désigner le narrateur par défaut de ce sous-ensemble. Certains titres ont un "
    "SOUS-DOSSIER intermédiaire nommant explicitement un narrateur différent du narrateur par "
    "défaut (ex: un comédien invité). L'ABSENCE de sous-dossier nommé signifie que le titre est "
    "raconté par le narrateur par défaut désigné par le dossier de premier niveau. Le TITRE de "
    "l'histoire (dernier segment du chemin) n'est jamais un nom de narrateur, même s'il "
    "ressemble à un nom propre (c'est un titre de conte). Si les indices sont insuffisants pour "
    "distinguer une identité précise, dis-le franchement (confiance faible, inferred_name vide ou "
    "très générique) plutôt que d'inventer."
)


def _split_folder_segments(folder_path: str) -> list[str]:
    return folder_path.split("/")[:-1]


def _format_story(title: str, folder_path: str) -> str:
    return f'- titre="{title}", dossiers_intermediaires={_split_folder_segments(folder_path)}'


async def _load_embeddings(conn: aiosqlite.Connection) -> dict[tuple[int, int, str], np.ndarray]:
    async with conn.execute(
        "SELECT story_id, track_id, speaker_label, embedding FROM eval_voice_embeddings"
    ) as cur:
        rows = await cur.fetchall()
    result = {}
    for r in rows:
        raw = np.frombuffer(r["embedding"], dtype=np.float32)
        result[(r["story_id"], r["track_id"], r["speaker_label"])] = raw / np.linalg.norm(raw)
    return result


async def _load_story_meta(conn: aiosqlite.Connection) -> dict[int, dict]:
    async with conn.execute("SELECT id, title, folder_path FROM stories") as cur:
        rows = await cur.fetchall()
    return {r["id"]: {"title": r["title"], "folder_path": r["folder_path"]} for r in rows}


def _find_counter_examples(
    cluster: dict, all_clusters: list[dict], rows_by_story: dict[int, dict], max_examples: int = 4
) -> list[str]:
    """Cherche, parmi les AUTRES clusters acoustiques, des titres partageant le même dossier
    de PREMIER NIVEAU que le cluster en cours d'identification, MAIS avec un sous-dossier
    intermédiaire supplémentaire (folder_path plus profond) — signe d'un narrateur nommé
    explicitement, différent du narrateur par défaut. Le simple partage d'auteur déclaré ne
    suffit pas comme filtre : observé en pratique, ça remontait des titres du MÊME narrateur
    (pas de sous-dossier non plus) comme "contre-exemples", ce qui n'apprend rien au LLM —
    d'où le filtre explicite sur la profondeur. Validé manuellement : avec un vrai contre-
    exemple (sous-dossier nommé du type ".../Jacques Alric/..."), le LLM comprend que
    "Histoires Beau frere Anne" est un label de collection avec narrateur par défaut, pas un
    prénom, et répond "Beau-frère d'Anne" au lieu de "Anne"."""
    own_top_folders = {
        rows_by_story[m["story_id"]]["folder_path"].split("/")[0]
        for m in cluster["appearances"] if m["story_id"] in rows_by_story
    }
    if not own_top_folders:
        return []
    examples: list[str] = []
    seen_titles: set[str] = set()
    for other in all_clusters:
        if other is cluster:
            continue
        for a in other["appearances"]:
            meta = rows_by_story.get(a["story_id"])
            if not meta or meta["title"] in seen_titles:
                continue
            segments = meta["folder_path"].split("/")[:-1]
            if len(segments) <= 1 or segments[0] not in own_top_folders:
                continue  # pas de sous-dossier nommé, ou dossier racine différent : pas exploitable
            seen_titles.add(meta["title"])
            examples.append(_format_story(meta["title"], meta["folder_path"]))
            break  # un seul titre par cluster suffit à illustrer le contraste
        if len(examples) >= max_examples:
            break
    return examples[:max_examples]


async def _identify(
    members: list[dict], rows_by_story: dict[int, dict], counter_examples: list[str] | None = None
) -> dict:
    seen: dict[int, dict] = {}
    for m in members:
        seen[m["story_id"]] = rows_by_story[m["story_id"]]
    rows = list(seen.values())
    authors = sorted({m["author"] for m in members})
    roles = sorted({m["character_name"] for m in members})

    user_content = (
        f"Voix (regroupement acoustique) — apparaît dans {len(members)} extraits, {len(rows)} histoires, "
        f"auteurs déclarés vus: {authors}, rôles vus: {roles[:8]} :\n"
        + "\n".join(_format_story(r["title"], r["folder_path"]) for r in rows)
    )
    if counter_examples:
        user_content += (
            "\n\nPour comparaison, titres partageant un même label de premier niveau mais "
            "appartenant à un cluster acoustique DIFFÉRENT (donc probablement un narrateur "
            "différent — ne pas les inclure dans la déduction ci-dessus) :\n"
            + "\n".join(counter_examples)
        )
    user_content += "\n\nDéduis l'identité probable de cette voix."

    raw = await llm.call_tool(system=SYSTEM_PROMPT, user=user_content, tool=IDENTIFY_TOOL, max_tokens=500)

    # Le tool_choice="required" côté LLM ne garantit pas que le modèle remplisse TOUS les
    # champs déclarés "required" du schéma (observé en pratique : is_professional absent
    # sur un appel après ~50 réussis) — on retombe sur des valeurs sûres plutôt que de
    # laisser un KeyError faire échouer tout le run (et perdre les identités déjà calculées).
    missing = [k for k in ("inferred_name", "confidence", "is_professional", "reasoning") if k not in raw]
    if missing:
        logger.warning(f"narrator_identity: réponse LLM incomplète (champs manquants: {missing}) — {raw}")
    return {
        "inferred_name": raw.get("inferred_name") or "",
        "confidence": raw.get("confidence") or "faible",
        "is_professional": bool(raw.get("is_professional", False)),
        "reasoning": raw.get("reasoning") or "",
    }


async def _resolve(
    members: list[dict], rows_by_story: dict[int, dict], counter_examples: list[str] | None = None,
    level: int = 0,
) -> list[dict]:
    """Retourne une liste de groupes finaux (feuilles de l'arbre d'escalade), chacun avec
    son identité déduite. Une voix jamais recroisée ailleurs (cluster à 1 membre) n'a rien
    à déduire d'un recoupement — ignorée plutôt qu'envoyée au LLM sans matière. Les
    contre-exemples sont calculés une fois sur le cluster d'origine (voir run()) et restent
    valables à travers l'escalade : ils comparent au dossier de premier niveau, pas au
    sous-groupe courant."""
    if len(members) < 2:
        return []

    identity = await _identify(members, rows_by_story, counter_examples)
    if identity["confidence"] == "haute" or level >= len(ESCALATION_THRESHOLDS):
        return [{"members": members, "identity": identity}]

    subgroups = voice_eval._cluster_by_embedding(members, ESCALATION_THRESHOLDS[level])
    if len(subgroups) <= 1:
        # La scission n'a rien séparé : inutile d'insister, on garde le résultat actuel
        # plutôt que de reboucler indéfiniment sur le même groupe non scindable.
        return [{"members": members, "identity": identity}]

    leaves = []
    for group in subgroups:
        leaves.extend(await _resolve(group, rows_by_story, counter_examples, level + 1))
    if not leaves:
        # Tous les sous-groupes sont retombés à 1 membre (voix isolées) : on garde le
        # résultat du niveau courant plutôt que de perdre l'information.
        return [{"members": members, "identity": identity}]
    return leaves


async def _persist_leaf(
    conn: aiosqlite.Connection, leaf: dict, threshold: float, verified_story_ids: set[int]
) -> int:
    """Insère UNE identité déduite et, si confiance haute, pousse en production sur
    stories.narrator (écrase une valeur existante : voir décision explicite — le clustering
    acoustique fait foi sur CE fichier précis, une valeur déjà présente peut venir d'un
    enrichissement web qui s'est trompé d'édition/support, cas observé sur Le_BGG). Commit
    immédiat par leaf plutôt qu'en fin de run entier : un run porte sur ~200+ clusters et
    autant d'appels LLM, un seul échec (réseau, réponse malformée) en cours de route ne doit
    pas faire perdre tout le travail déjà accompli avant lui.

    verified_story_ids exclut les histoires ayant un casting de référence IMMUABLE (voir
    db.story_cast_verified, alimenté manuellement ou via reference/extract_cast_from_media.py) :
    ni leurs membres ni leur stories.narrator ne doivent être touchés par le clustering
    acoustique, même si l'identité déduite couvre aussi d'autres histoires non protégées
    (cas réel observé : un cluster "Jean Rochefort" regroupait à tort des pistes de Pinocchio,
    en réalité interprété par Anouk Grinberg selon les tags ID3/casting confirmé)."""
    identity = leaf["identity"]
    now = datetime.now(timezone.utc).isoformat()
    members = [m for m in leaf["members"] if m["story_id"] not in verified_story_ids]
    if not members:
        return 0

    cur = await conn.execute(
        "INSERT INTO narrator_identities "
        "(inferred_name, confidence, is_professional, reasoning, cluster_threshold, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            identity["inferred_name"], identity["confidence"], int(identity["is_professional"]),
            identity["reasoning"], threshold, now,
        ),
    )
    identity_id = cur.lastrowid
    await conn.executemany(
        "INSERT INTO narrator_identity_members (identity_id, story_id, track_id, speaker_label) "
        "VALUES (?, ?, ?, ?)",
        [(identity_id, m["story_id"], m["track_id"], m["speaker_label"]) for m in members],
    )

    n_pushed = 0
    if identity["confidence"] == "haute" and identity["inferred_name"]:
        story_ids = sorted({m["story_id"] for m in members})
        await conn.executemany(
            "UPDATE stories SET narrator = ?, updated_at = ? WHERE id = ?",
            [(identity["inferred_name"], now, sid) for sid in story_ids],
        )
        n_pushed = len(story_ids)

    await conn.commit()
    return n_pushed


async def run(threshold: float | None = None, persist: bool = True) -> dict:
    """Point d'entrée : reprend le clustering acoustique (speaker_voice_eval.cluster_voices),
    déduit une identité par cluster via LLM (avec escalade de seuil pour les clusters mal
    fusionnés — voir _resolve), et par défaut persiste le résultat au fur et à mesure : trace
    complète dans narrator_identities/_members, et pour les identités à confiance haute, mise
    à jour de stories.narrator (le champ de production, distinct de stories.author). Le
    contenu précédent de narrator_identities est vidé en préambule (le clustering n'étant pas
    stable d'une exécution à l'autre, le garder entre deux runs n'a pas de sens) — mais si le
    run est interrompu en cours de route, tout ce qui a été traité AVANT l'interruption reste
    acquis, contrairement à une purge+réécriture en un seul bloc final."""
    threshold = threshold if threshold is not None else voice_eval.CLUSTER_THRESHOLD
    cluster_result = await voice_eval.cluster_voices(threshold=threshold)
    if "error" in cluster_result:
        return cluster_result

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        embeddings = await _load_embeddings(conn)
        rows_by_story = await _load_story_meta(conn)

        # Histoires à casting de référence vérifié (voir db.story_cast_verified) : le
        # clustering acoustique ne doit jamais les réassigner, ni dans
        # narrator_identity_members ni sur stories.narrator - voir _persist_leaf.
        async with conn.execute("SELECT DISTINCT story_id FROM story_cast_verified") as cur:
            verified_story_ids = {row[0] for row in await cur.fetchall()}
        if verified_story_ids:
            logger.info(
                f"narrator_identity: {len(verified_story_ids)} histoire(s) à casting vérifié, "
                f"exclue(s) du clustering acoustique"
            )

        if persist:
            await conn.execute("DELETE FROM narrator_identity_members")
            await conn.execute("DELETE FROM narrator_identities")
            await conn.commit()

        leaves: list[dict] = []
        n_pushed = 0
        multi_clusters = [c for c in cluster_result["clusters"] if c["size"] >= 2]
        for i, cluster in enumerate(multi_clusters, start=1):
            members = []
            for a in cluster["appearances"]:
                key = (a["story_id"], a["track_id"], a["speaker_label"])
                emb = embeddings.get(key)
                if emb is None:
                    continue
                members.append({**a, "emb": emb})
            # Cherche parmi TOUS les clusters, y compris ceux à 1 seul membre : un narrateur
            # invité peut n'apparaître qu'une seule fois dans tout le catalogue, et serait
            # exclu à tort si on ne regardait que les clusters multi-membres.
            counter_examples = _find_counter_examples(cluster, cluster_result["clusters"], rows_by_story)
            cluster_leaves = await _resolve(members, rows_by_story, counter_examples)
            leaves.extend(cluster_leaves)
            if persist:
                for leaf in cluster_leaves:
                    n_pushed += await _persist_leaf(conn, leaf, threshold, verified_story_ids)
            logger.info(f"narrator_identity: cluster {i}/{len(multi_clusters)} traité ({len(cluster_leaves)} identités)")

    result = {
        "n_source_clusters": cluster_result["n_multi_member_clusters"],
        "n_identities": len(leaves),
        "n_haute": sum(1 for l in leaves if l["identity"]["confidence"] == "haute"),
        "n_stories_updated": n_pushed,
        "identities": [
            {
                "inferred_name": l["identity"]["inferred_name"],
                "confidence": l["identity"]["confidence"],
                "is_professional": l["identity"]["is_professional"],
                "reasoning": l["identity"]["reasoning"],
                "size": len(l["members"]),
                "titles": sorted({m["title"] for m in l["members"]}),
            }
            for l in leaves
        ],
    }
    logger.info(
        f"narrator_identity run: {result['n_source_clusters']} clusters -> {result['n_identities']} identités "
        f"({result['n_haute']} haute confiance, {n_pushed} histoires mises à jour)"
    )
    return result
