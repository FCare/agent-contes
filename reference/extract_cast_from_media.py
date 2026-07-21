"""
Extraction du casting de référence depuis les métadonnées embarquées dans les
fichiers audio - alimente story_cast_verified (voir db.py), IMMUABLE par le
clustering acoustique (reference/narrator_identity.py, qui reconstruit tout à
chaque exécution et peut se tromper - cas réel observé : Pinocchio assigné à
tort à "Jean Rochefort" alors que le vrai casting, "Anouk Grinberg" en tête,
est écrit noir sur blanc dans le tag ID3 TPE1 de 17 des 18 pistes).

Deux sources, dans l'ordre (la première qui donne un résultat exploitable
l'emporte) :
1. Tag ID3 TPE1 (Performer) - vote majoritaire sur toutes les pistes de
   l'histoire, une piste isolée mal taguée (générique "Interprète inconnu")
   ne doit pas invalider un signal par ailleurs cohérent sur le reste de
   l'album.
2. Pochette embarquée (APIC), si présente - un LLM vision lit les noms
   d'interprètes visibles sur l'image, s'il y en a. Repli seulement : sur ce
   catalogue, aucune piste échantillonnée n'a d'artwork embarqué (0/60), donc
   cette voie ne se déclenchera probablement jamais en pratique, mais reste
   nécessaire pour les histoires sans TPE1 exploitable.

Usage : python -m reference.extract_cast_from_media [--story-id ID] [--limit N]
"""
import argparse
import asyncio
import base64
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

import mutagen

import db
import llm

logger = logging.getLogger(__name__)

CONTES_ROOT = Path(os.environ.get("CONTES_ROOT", "/contes"))

_PLACEHOLDER_MARKERS = ("inconnu", "unknown")
_NON_NAME_TOKENS = {"etc", "etc.", "et al", "et al.", "et autres", ""}

# Valeurs vues en pratique dans TPE1 qui ne sont PAS des noms d'interprète : label de
# collection/série, nom de site web, placeholder générique - comparaison en minuscules.
# "Contes" à lui seul polluait 56 histoires sur un premier run en masse.
_JUNK_VALUES = {
    "contes", "various artists", "les belles histoires", "mes premiers j'aime lire",
    "les enfantastiques", "interprète inconnu", "album inconnu",
    # Auteur littéraire de la série "Contes de la rue Broca" (stories.author),
    # placé à tort dans TPE1 (Performer) au lieu du champ auteur - sans cette
    # exclusion, le vote majoritaire ID3 masque le fallback artwork sur ces
    # fichiers, qui ONT une pochette embarquée (voir story 82, la rue broca#1..12).
    "pierre gripari",
    # Mêmes cas : auteurs littéraires connus retrouvés dans TPE1 au lieu du
    # narrateur réel (stories 3, 55, 69 - aucune pochette disponible pour
    # celles-ci, donc pas de repli possible, mieux vaut aucune donnée qu'une
    # donnée fausse). "carrol"/"lewis" : "Carrol, Lewis" (Lewis Carroll,
    # format "Nom, Prénom") coupé à tort en deux par le split sur virgule.
    "antoine de saint-exupéry", "mary shelley", "roald dahl", "carrol", "lewis",
}
# Préfixes français encadrant un vrai nom ("Lu par Richard Bohringer" -> "Richard
# Bohringer") à retirer avant de garder la valeur.
_NAME_PREFIXES = ("lu par ", "raconté par ", "raconte par ", "interprété par ", "interprete par ")

_CAST_EXTRACT_TOOL = [{
    "type": "function",
    "function": {
        "name": "extract_cast",
        "description": (
            "Extrait la liste des interprètes/voix visibles sur cette pochette "
            "d'histoire audio, si le texte en est lisible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Noms propres des interprètes/voix identifiés sur l'image, "
                        "un par élément. Liste vide si aucun nom n'est lisible."
                    ),
                },
            },
            "required": ["names"],
        },
    },
}]


def _is_junk_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in _JUNK_VALUES or lowered in _NON_NAME_TOKENS:
        return True
    if ".com" in lowered or "www." in lowered:
        return True
    # "d'après Rabelais" / "d'apres Rabelais" : décrit la source LITTÉRAIRE adaptée
    # (voir stories.literary_author), jamais l'interprète - confusion documentée
    # ailleurs dans le code (db.get_narrator_info).
    if lowered.startswith("d'après ") or lowered.startswith("d'apres "):
        return True
    return False


def _split_cast_names(raw: str) -> list[str]:
    """'Lu par Anouk Grinberg, J-P Cassel; Zabou Breitman, Etc.' -> noms
    individuels, en retirant les préfixes français encadrants et en filtrant
    les marqueurs génériques/valeurs connues comme n'étant pas des noms
    propres (voir _JUNK_VALUES, _is_junk_name)."""
    raw = raw.replace("&", ",").replace(";", ",")
    names = []
    for part in re.split(r",", raw):
        name = part.strip(" .")
        lowered = name.lower()
        for prefix in _NAME_PREFIXES:
            if lowered.startswith(prefix):
                name = name[len(prefix):].strip(" .")
                break
        if not name or _is_junk_name(name):
            continue
        names.append(name)
    return names


async def _extract_from_id3(story_id: int) -> list[str] | None:
    tracks = await db.get_tracks_for_story(story_id)
    votes = Counter()
    for t in tracks:
        abs_path = CONTES_ROOT / t["file_path"]
        try:
            f = mutagen.File(abs_path)
        except Exception as e:
            logger.debug(f"extract_cast_from_media: échec lecture {abs_path}: {e}")
            continue
        if not f or not f.tags:
            continue
        tpe1 = f.tags.get("TPE1")
        if not tpe1 or not tpe1.text:
            continue
        text = str(tpe1.text[0]).strip()
        if not text or any(marker in text.lower() for marker in _PLACEHOLDER_MARKERS):
            continue
        votes[text] += 1
    if not votes:
        return None
    best_text, _ = votes.most_common(1)[0]
    names = _split_cast_names(best_text)
    return names or None


async def _extract_from_artwork(story_id: int) -> list[str] | None:
    tracks = await db.get_tracks_for_story(story_id)
    for t in tracks:
        abs_path = CONTES_ROOT / t["file_path"]
        try:
            f = mutagen.File(abs_path)
        except Exception:
            continue
        if not f or not f.tags:
            continue
        apic_key = next((k for k in f.tags.keys() if k.startswith("APIC")), None)
        if not apic_key:
            continue
        apic = f.tags[apic_key]
        b64 = base64.b64encode(apic.data).decode("ascii")
        mime = apic.mime or "image/jpeg"
        result = await llm.call_tool(
            system=(
                "Tu identifies les interprètes/voix mentionnés sur une pochette "
                "d'histoire audio, quand le texte en est lisible."
            ),
            user=[
                {
                    "type": "text",
                    "text": "Quels sont les noms des interprètes/voix visibles sur cette pochette, s'il y en a ?",
                },
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
            tool=_CAST_EXTRACT_TOOL,
        )
        names = [n.strip() for n in result.get("names", []) if n and n.strip() and not _is_junk_name(n.strip())]
        if names:
            return names
    return None


async def process_story(story_id: int) -> bool:
    """Retourne True si un casting a été extrait et persisté pour cette histoire."""
    if await db.has_verified_cast(story_id):
        logger.debug(f"extract_cast_from_media: histoire {story_id} déjà vérifiée, ignorée")
        return False

    names = await _extract_from_id3(story_id)
    source = "id3_metadata"
    if not names:
        names = await _extract_from_artwork(story_id)
        source = "artwork_vision"
    if not names:
        return False

    # Le premier nom cité est conventionnellement le narrateur principal (voir
    # le format observé : "Anouk Grinberg, J-P Cassel, Zabou Breitman, Etc.") -
    # heuristique raisonnable mais non garantie, jamais promue si source
    # incertaine (l'utilisateur peut toujours corriger via une entrée 'human',
    # prioritaire — voir db._SOURCE_PRIORITY).
    for i, name in enumerate(names):
        await db.add_verified_cast_member(
            story_id, name, role=None, is_narrator=(i == 0), source=source
        )
    logger.info(f"extract_cast_from_media: histoire {story_id} — casting extrait via {source}: {names}")
    return True


async def run(story_id: int | None = None, limit: int | None = None) -> dict:
    story_ids = [story_id] if story_id else await db.get_all_story_ids()
    if limit:
        story_ids = story_ids[:limit]

    n_processed = 0
    n_extracted = 0
    for sid in story_ids:
        n_processed += 1
        if await process_story(sid):
            n_extracted += 1

    result = {"n_processed": n_processed, "n_extracted": n_extracted}
    logger.info(f"extract_cast_from_media: {n_processed} histoire(s) examinée(s), {n_extracted} casting(s) extrait(s)")
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--story-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(db.init_db())
    asyncio.run(run(story_id=args.story_id, limit=args.limit))


if __name__ == "__main__":
    main()
