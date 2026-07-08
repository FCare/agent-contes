"""Classifies each story two ways, to make browsing/filtering reliable:

1. Fixed traits (age_range, mood_tags) — a small vocabulary decided up front,
   assigned per story by an LLM call. Exact-filterable (like duration), no
   ambiguity: the calling agent already knows the valid values.

2. Free/open theme classes — discovered FROM the catalogue's actual content
   rather than a vocabulary picked in advance. Three passes: propose a raw,
   story-specific theme label per story; consolidate all raw labels into a
   small set of canonical classes (avoids ~1 class per story); assign each
   story to its closest canonical class. Each class is embedded in Chroma so
   a free-text theme query ("des histoires de pirates") can be routed to the
   nearest class by similarity rather than an exact/keyword match.
"""

import asyncio
import logging

import chroma_store
import db
import llm

logger = logging.getLogger(__name__)

AGE_RANGES = ["tout-petit", "petit", "enfant", "grand-enfant"]
MOOD_TAGS = [
    "peur", "aventure", "humour", "tendresse", "animaux", "magie",
    "amitie", "educatif", "quotidien", "classique", "voyage", "mystere",
]

# Cible pour la consolidation des classes libres : assez pour discriminer un
# catalogue de ~300 histoires, assez peu pour rester compréhensible et éviter
# les classes quasi uniques à une seule histoire.
MIN_THEME_CLASSES = 8
MAX_THEME_CLASSES = 25

_TRAITS_TOOL = [{
    "type": "function",
    "function": {
        "name": "classify_traits",
        "description": "Assigne une tranche d'âge et des tags d'ambiance à un conte, depuis un vocabulaire fixe",
        "parameters": {
            "type": "object",
            "properties": {
                "age_range": {
                    "type": "string",
                    "enum": AGE_RANGES,
                    "description": (
                        "tout-petit (0-3 ans, très simple/répétitif) ; petit (3-6 ans) ; "
                        "enfant (6-9 ans) ; grand-enfant (9 ans et plus, plus complexe/long)."
                    ),
                },
                "mood_tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": MOOD_TAGS},
                    "description": "1 à 4 tags d'ambiance parmi la liste, ceux qui dominent vraiment l'histoire.",
                },
            },
            "required": ["age_range", "mood_tags"],
        },
    },
}]

_RAW_THEME_TOOL = [{
    "type": "function",
    "function": {
        "name": "propose_theme",
        "description": "Propose un thème court et spécifique pour ce conte précis",
        "parameters": {
            "type": "object",
            "properties": {
                "theme_label": {
                    "type": "string",
                    "description": (
                        "3 à 6 mots capturant ce qui distingue CETTE histoire (ex: "
                        "'loup rusé et grand-mère', 'apprentissage de l'alphabet', "
                        "'colonie pénitentiaire pour enfants'). Pas une catégorie générique."
                    ),
                },
            },
            "required": ["theme_label"],
        },
    },
}]


def _consolidate_tool(min_classes: int, max_classes: int) -> list:
    return [{
        "type": "function",
        "function": {
            "name": "consolidate_theme_classes",
            "description": "Regroupe des thèmes bruts, un par histoire, en un petit ensemble de classes communes",
            "parameters": {
                "type": "object",
                "properties": {
                    "classes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Nom court de la classe (2-4 mots)"},
                                "description": {
                                    "type": "string",
                                    "description": "1 phrase décrivant ce que couvre cette classe",
                                },
                            },
                            "required": ["label", "description"],
                        },
                        "description": (
                            f"Entre {min_classes} et {max_classes} classes couvrant TOUS les thèmes bruts "
                            "fournis. Fusionne les thèmes isolés/proches dans la classe la plus pertinente "
                            "plutôt que de créer une classe pour un cas unique."
                        ),
                    },
                },
                "required": ["classes"],
            },
        },
    }]


def _assign_tool(valid_labels: list[str]) -> list:
    return [{
        "type": "function",
        "function": {
            "name": "assign_theme_classes",
            "description": "Choisit zéro, une ou plusieurs classes thématiques pertinentes pour ce texte",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_labels": {
                        "type": "array",
                        "items": {"type": "string", "enum": valid_labels},
                        "description": (
                            "0 à 3 classes parmi la liste, celles qui correspondent vraiment à "
                            "ce texte. Une histoire peut légitimement relever de plusieurs classes "
                            "(ex: à la fois 'magie et enchantements' et 'peur et créatures "
                            "fantastiques') — renvoie un tableau vide si aucune ne correspond "
                            "clairement à CE texte précis, n'en force jamais une par défaut."
                        ),
                    },
                },
                "required": ["class_labels"],
            },
        },
    }]


# Longueur max (caractères) de texte brut envoyée en un seul appel LLM, calée sur la
# fenêtre de contexte du backend — au-delà, l'histoire est découpée en plusieurs appels
# (voir _raw_text_chunks). La classification lit ainsi le texte réel de l'histoire plutôt
# que ses résumés, qui édulcorent volontiers les détails marquants (violence, peur) au
# profit d'un ton narratif plus neutre (ex: une décapitation résumée en simple
# "confrontation").
RAW_CONTEXT_MAX_CHARS = 45000

# Cap dur sur le nombre de classes libres accumulées par histoire via l'union des
# extraits — au-delà, l'utilité de la classification (filtrer/naviguer) diminue.
MAX_THEME_CLASSES_PER_STORY = 5


# Nombre d'histoires traitées en parallèle par les fonctions *_pending — les appels LLM
# eux-mêmes restent bornés par llm._LLM_SEMAPHORE (vLLM fait le vrai batching côté
# serveur, voir llm.py) ; cette limite évite juste d'ouvrir des centaines de tâches
# asyncio à la fois pour les gros lots (~300 histoires).
_STORY_CONCURRENCY = 8


async def _run_concurrently(stories: list[dict], process) -> dict:
    sem = asyncio.Semaphore(_STORY_CONCURRENCY)

    async def _bounded(story: dict) -> bool:
        async with sem:
            return await process(story)

    results = await asyncio.gather(*[_bounded(s) for s in stories])
    n_done = sum(1 for r in results if r)
    return {"done": n_done, "errors": len(results) - n_done}


async def _raw_text_chunks(story_id: int) -> list[str]:
    """Regroupe le texte brut diarizé des périodes d'une histoire en un minimum de blocs
    de taille RAW_CONTEXT_MAX_CHARS chacun, dans l'ordre chronologique."""
    periods = await db.get_raw_periods_for_story(story_id)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for p in periods:
        text = p.get("raw_transcript_text") or ""
        if not text:
            continue
        if current and current_len + len(text) > RAW_CONTEXT_MAX_CHARS:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(text)
        current_len += len(text)
    if current:
        chunks.append("\n".join(current))
    return chunks


async def classify_traits_pending(story_id: int | None = None, limit: int | None = None) -> dict:
    stories = await db.stories_missing_traits(story_id=story_id, limit=limit)

    async def _process(story: dict) -> bool:
        chunks = await _raw_text_chunks(story["id"])
        if not chunks:
            logger.error(f"classify_traits: aucun texte brut disponible pour l'histoire {story['id']}")
            return False

        best_age_idx = -1
        mood_tags: set[str] = set()
        for i, chunk in enumerate(chunks):
            extrait = f" (extrait {i + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            result = await llm.call_tool(
                system=(
                    "Tu classes un conte audio pour enfants selon un vocabulaire fixe, à partir du "
                    "texte réel (transcrit, avec les noms de locuteurs) de l'histoire — pas d'un "
                    "résumé. Choisis la tranche d'âge la plus adaptée et les tags d'ambiance qui "
                    f"dominent vraiment ce texte{extrait}. Base-toi sur ce qui se passe réellement "
                    "(y compris les détails marquants comme la violence ou la peur), pas sur le seul "
                    "ton général de l'histoire."
                ),
                user=f"TITRE: {story['title']}\n\nTEXTE{extrait}:\n{chunk}",
                tool=_TRAITS_TOOL,
            )
            age_range = result.get("age_range")
            raw_mood_tags = result.get("mood_tags") or []
            # Le grammar-constrained decoding du backend LLM ne garantit pas toujours le
            # respect strict d'un enum imbriqué dans un array — filtrer plutôt que de
            # stocker une valeur hors vocabulaire qui casserait le filtre exact plus tard.
            valid_mood = [m for m in raw_mood_tags if m in MOOD_TAGS]
            dropped = set(raw_mood_tags) - set(valid_mood)
            if dropped:
                logger.warning(f"classify_traits: histoire {story['id']}, tags hors vocabulaire ignorés: {dropped}")
            mood_tags.update(valid_mood)
            if age_range in AGE_RANGES:
                # La tranche d'âge retenue est la plus mature vue sur tout le texte : un
                # seul passage plus intense justifie de relever la tranche pour l'histoire
                # entière, même si le reste est plus doux.
                best_age_idx = max(best_age_idx, AGE_RANGES.index(age_range))
            else:
                logger.warning(f"classify_traits: histoire {story['id']}, extrait {i + 1}, age_range invalide: {age_range!r}")

        if best_age_idx < 0:
            logger.error(f"classify_traits: aucune tranche d'âge valide pour l'histoire {story['id']}")
            return False
        age_range = AGE_RANGES[best_age_idx]
        sorted_mood = sorted(mood_tags)
        await db.set_story_traits(story["id"], age_range, sorted_mood)
        logger.info(f"classify_traits: histoire {story['id']} -> {age_range} / {sorted_mood}")
        return True

    result = await _run_concurrently(stories, _process)
    logger.info(f"classify_traits: {result}")
    return result


async def propose_raw_themes_pending(story_id: int | None = None, limit: int | None = None) -> dict:
    stories = await db.stories_missing_raw_theme(story_id=story_id, limit=limit)

    async def _process(story: dict) -> bool:
        chunks = await _raw_text_chunks(story["id"])
        if not chunks:
            logger.error(f"propose_raw_themes: aucun texte brut disponible pour l'histoire {story['id']}")
            return False

        # Sur plusieurs extraits, le thème est affiné de proche en proche (le thème
        # provisoire est réinjecté comme contexte du suivant) plutôt que déduit d'un seul
        # extrait au hasard — même schéma que le chaînage utilisé pour identifier les
        # personnages sur les histoires multi-pistes. Les extraits d'UNE histoire restent
        # traités en séquence (ce chaînage l'exige) ; c'est le traitement de plusieurs
        # histoires EN PARALLÈLE (voir _run_concurrently) qui exploite le batching vLLM.
        label = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                system = (
                    "Tu résumes en un thème court et SPÉCIFIQUE ce qui distingue ce conte des autres "
                    "— pas une catégorie générique ('conte pour enfant'), mais ce qui le rend "
                    "reconnaissable (personnages, situation, morale, univers, y compris un moment "
                    "marquant comme la peur ou la violence s'il y en a). Base-toi sur le texte réel "
                    "de l'histoire, pas sur un résumé édulcoré."
                )
                user = f"TITRE: {story['title']}\n\nTEXTE (début de l'histoire):\n{chunk}"
            else:
                system = (
                    "Voici la SUITE du même conte, ainsi que le thème provisoire déjà identifié à "
                    "partir du début. Affine ce thème pour qu'il reflète l'histoire ENTIÈRE (pas "
                    "seulement ce nouvel extrait) — en particulier si cette suite révèle un élément "
                    "marquant (rebondissement, peur, violence...) absent du début, il doit apparaître "
                    "dans le thème."
                )
                user = f"TITRE: {story['title']}\n\nTHÈME PROVISOIRE: {label}\n\nSUITE DU TEXTE:\n{chunk}"
            result = await llm.call_tool(system=system, user=user, tool=_RAW_THEME_TOOL)
            new_label = (result.get("theme_label") or "").strip()
            if new_label:
                label = new_label

        if not label:
            logger.error(f"propose_raw_themes: aucune réponse LLM pour l'histoire {story['id']}")
            return False
        await db.set_raw_theme_label(story["id"], label)
        logger.info(f"propose_raw_themes: histoire {story['id']} -> {label!r}")
        return True

    result = await _run_concurrently(stories, _process)
    logger.info(f"propose_raw_themes: {result}")
    return result


async def consolidate_theme_classes() -> dict:
    """Repart de zéro (voir db.replace_theme_classes) : à relancer après avoir ajouté
    de nouvelles histoires avec propose_raw_themes_pending, pour que les classes
    reflètent tout le catalogue actuel plutôt que seulement le premier lot traité."""
    raw = await db.all_raw_theme_labels()
    if not raw:
        return {"classes": 0, "reason": "aucun theme_label brut disponible"}

    listing = "\n".join(f"{r['id']}. {r['title']} — {r['raw_theme_label']}" for r in raw)
    result = await llm.call_tool(
        system=(
            "Voici les thèmes bruts, un par conte, d'un catalogue de contes audio pour enfants. "
            "Regroupe-les en classes thématiques communes qui ont un vrai sens pour naviguer le "
            "catalogue (ex: 'ruse animale', 'peur et créatures effrayantes', 'famille et tendresse'). "
            "Chaque conte doit pouvoir être rattaché à l'une des classes proposées."
        ),
        user=listing,
        tool=_consolidate_tool(MIN_THEME_CLASSES, MAX_THEME_CLASSES),
        max_tokens=4000,
    )
    classes = result.get("classes") or []
    if not classes:
        logger.error("consolidate_theme_classes: aucune réponse LLM")
        return {"classes": 0, "reason": "échec LLM"}

    label_to_id = await db.replace_theme_classes(classes)
    for c in classes:
        class_id = label_to_id[c["label"]]
        chroma_store.upsert_theme_class(class_id, c["label"], c["description"])

    logger.info(f"consolidate_theme_classes: {len(classes)} classes créées")
    return {"classes": len(classes)}


async def assign_theme_classes_pending(story_id: int | None = None, limit: int | None = None) -> dict:
    classes = await db.get_theme_classes()
    if not classes:
        return {"done": 0, "errors": 0, "reason": "aucune classe — lancer consolidate_theme_classes d'abord"}
    label_to_id = {c["label"]: c["id"] for c in classes}
    tool = _assign_tool(list(label_to_id))
    classes_listing = "\n".join(f"- {c['label']}: {c['description']}" for c in classes)

    stories = await db.stories_missing_theme_assignment(story_id=story_id, limit=limit)

    async def _process(story: dict) -> bool:
        chunks = await _raw_text_chunks(story["id"])
        if not chunks:
            logger.error(f"assign_theme_classes: aucun texte brut disponible pour l'histoire {story['id']}")
            return False

        # Une classe est retenue dès qu'UN extrait la justifie (union), pas seulement si
        # elle ressort d'un résumé global qui peut avoir édulcoré ce passage précis —
        # dict pour dédoublonner tout en gardant l'ordre de première apparition.
        chosen: dict[int, str] = {}
        for i, chunk in enumerate(chunks):
            extrait = f" (extrait {i + 1}/{len(chunks)} du texte)" if len(chunks) > 1 else ""
            result = await llm.call_tool(
                system=(
                    "Choisis, parmi les classes thématiques ci-dessous, celles qui correspondent "
                    f"vraiment à ce texte{extrait} — une, plusieurs, ou aucune si rien ne correspond "
                    "clairement à CET extrait précis. Base-toi sur ce qui se passe réellement dans le "
                    "texte, pas sur un résumé : un passage effrayant ou violent doit être reconnu même "
                    "si le reste de l'histoire relève d'un autre registre."
                ),
                user=f"CLASSES DISPONIBLES:\n{classes_listing}\n\nTITRE: {story['title']}\n"
                     f"THÈME BRUT: {story['raw_theme_label']}\n\nTEXTE{extrait}:\n{chunk}",
                tool=tool,
            )
            chosen_labels = result.get("class_labels") or []
            dropped = set(chosen_labels) - set(label_to_id)
            if dropped:
                logger.warning(f"assign_theme_classes: histoire {story['id']}, classes hors vocabulaire ignorées: {dropped}")
            for label in chosen_labels:
                if label in label_to_id:
                    chosen.setdefault(label_to_id[label], label)

        if not chosen:
            logger.error(f"assign_theme_classes: aucune classe retenue pour l'histoire {story['id']}")
            return False
        final_ids = list(chosen.keys())[:MAX_THEME_CLASSES_PER_STORY]
        await db.set_story_theme_classes(story["id"], final_ids)
        logger.info(f"assign_theme_classes: histoire {story['id']} -> {[chosen[i] for i in final_ids]}")
        return True

    result = await _run_concurrently(stories, _process)
    logger.info(f"assign_theme_classes: {result}")
    return result
