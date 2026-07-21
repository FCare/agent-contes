import argparse
import asyncio
import logging
import sys

import db
from . import build_wiki as build_wiki_stage
from . import classify as classify_stage
from . import duration as duration_stage
from . import embed as embed_stage
from . import enrich_web as enrich_web_stage
from . import extract_cast_from_media as extract_cast_stage
from . import identify_speakers as identify_speakers_stage
from . import narrator_identity as narrator_identity_stage
from . import reconcile_titles as reconcile_titles_stage
from . import scan as scan_stage
from . import speaker_voice_eval as speaker_voice_eval_stage
from . import split_stories as split_stories_stage
from . import summarize as summarize_stage
from . import transcribe as transcribe_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

STAGES = ["scan", "duration", "transcribe", "split_stories", "identify_speakers", "summarize", "embed"]
# Ni l'une ni l'autre ne font partie de "all" : backfill_* sont des migrations
# ponctuelles pour les histoires déjà 'ready' avant l'ajout d'une fonctionnalité ;
# enrich_web sollicite un service externe (recherche web) et doit rester un choix
# explicite, pas un effet de bord automatique de chaque histoire nouvellement traitée.
# classify_* : à lancer dans l'ordre traits -> themes -> consolidate_themes -> assign_themes
# (consolidate_themes doit être relancé après tout ajout d'histoires pour que les
# classes reflètent le catalogue à jour, donc reste un choix explicite).
EXTRA_STAGES = [
    "backfill_keywords", "backfill_fts", "backfill_segment_embeddings", "enrich_web", "reconcile_titles",
    "classify_traits", "propose_themes", "consolidate_themes", "assign_themes",
    # Expérimental, hors production : compare le mapping locuteur actuel (LLM sur le
    # texte du transcript, retenu comme référence) à une approche alternative par
    # empreinte vocale ECAPA-TDNN — voir reference/speaker_voice_eval.py. Choix explicite
    # uniquement, jamais un effet de bord de "all".
    "eval_speaker_voice",
    # Suite de eval_speaker_voice : déduit une identité de narrateur par cluster acoustique
    # via LLM, et pousse le résultat en production sur stories.narrator quand la confiance
    # est haute — voir reference/narrator_identity.py. Choix explicite, jamais un effet de
    # bord de "all" ni de "eval_speaker_voice" (dépend de son cache d'embeddings mais reste
    # une étape distincte, ré-exécutable indépendamment). Exclut automatiquement les
    # histoires à casting de référence vérifié (voir extract_cast_media ci-dessous).
    "narrator_identity",
    # Casting de référence extrait des tags ID3 (TPE1)/artwork embarqué - voir
    # reference/extract_cast_from_media.py. Source plus fiable que le clustering
    # acoustique quand disponible (texte structuré plutôt qu'inférence), et surtout
    # IMMUABLE : une fois une histoire traitée ici, narrator_identity ne la touche
    # plus jamais. Choix explicite, jamais un effet de bord de "all".
    "extract_cast_media",
    # Génère le wiki statique (voir reference/build_wiki.py) depuis l'état actuel
    # de la base — aucun coût LLM, régénération complète à chaque exécution.
    # Aussi appelée automatiquement en fin de synchro quotidienne (main.py,
    # _daily_catalog_sync_loop), donc rejouable manuellement ici pour un
    # rebuild à la demande sans attendre la prochaine synchro.
    "build_wiki",
]


async def run(stage: str, only_new: bool, story_id: int | None, limit: int | None) -> None:
    await db.init_db()
    if stage in ("scan", "all"):
        await scan_stage.scan(only_new=only_new)
    if stage in ("duration", "all"):
        await duration_stage.compute_durations()
    if stage in ("transcribe", "all"):
        await transcribe_stage.transcribe_pending(story_id=story_id)
    if stage in ("split_stories", "all"):
        await split_stories_stage.split_pending(story_id=story_id)
    if stage in ("identify_speakers", "all"):
        await identify_speakers_stage.identify_speakers_pending(story_id=story_id)
    if stage in ("summarize", "all"):
        await summarize_stage.summarize_pending(story_id=story_id)
    if stage in ("embed", "all"):
        await embed_stage.embed_pending(story_id=story_id)
    if stage == "backfill_keywords":
        await summarize_stage.backfill_keywords(story_id=story_id)
    if stage == "backfill_fts":
        await summarize_stage.backfill_fts(story_id=story_id)
    if stage == "backfill_segment_embeddings":
        result = await embed_stage.backfill_segment_embeddings(story_id=story_id)
        logger.info(f"backfill_segment_embeddings: {result}")
    if stage == "enrich_web":
        await enrich_web_stage.enrich_pending(story_id=story_id, limit=limit)
    if stage == "reconcile_titles":
        await reconcile_titles_stage.reconcile_pending(story_id=story_id)
    if stage == "classify_traits":
        await classify_stage.classify_traits_pending(story_id=story_id, limit=limit)
    if stage == "propose_themes":
        await classify_stage.propose_raw_themes_pending(story_id=story_id, limit=limit)
    if stage == "consolidate_themes":
        await classify_stage.consolidate_theme_classes()
    if stage == "assign_themes":
        await classify_stage.assign_theme_classes_pending(story_id=story_id, limit=limit)
    if stage == "eval_speaker_voice":
        result = await speaker_voice_eval_stage.run(limit=limit)
        logger.info(f"eval_speaker_voice: {result}")
    if stage == "narrator_identity":
        result = await narrator_identity_stage.run()
        logger.info(f"narrator_identity: {result}")
    if stage == "extract_cast_media":
        result = await extract_cast_stage.run(story_id=story_id, limit=limit)
        logger.info(f"extract_cast_media: {result}")
    if stage == "build_wiki":
        result = await build_wiki_stage.run()
        logger.info(f"build_wiki: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de référencement des contes")
    parser.add_argument("--stage", choices=[*STAGES, *EXTRA_STAGES, "all"], default="all")
    parser.add_argument("--only-new", action="store_true",
                         help="Ignore les histoires déjà présentes en base lors du scan")
    parser.add_argument("--story-id", type=int, default=None,
                         help="Limiter transcribe/identify_speakers/summarize/embed/enrich_web/classify_* à une seule histoire")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limiter enrich_web/classify_traits/propose_themes/assign_themes à N histoires (traitement par lots)")
    args = parser.parse_args()
    asyncio.run(run(args.stage, args.only_new, args.story_id, args.limit))


if __name__ == "__main__":
    main()
