import logging
import os
from pathlib import Path

import aiosqlite

import db

logger = logging.getLogger(__name__)

CONTES_ROOT = Path(os.environ.get("CONTES_ROOT", "/contes"))
HF_TOKEN = os.environ.get("HF_TOKEN")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
# Épinglé sur la version dont la licence a été acceptée (le défaut whisperx
# pointe vers "pyannote/speaker-diarization-community-1", un dépôt différent).
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

_asr_model = None
_align_model = None
_align_metadata = None
_diarize_pipeline = None


def _reset_models_to_cpu():
    """Abandonne les modèles déjà chargés en VRAM et force CPU/int8 pour tous les
    chargements suivants. Un modèle déjà chargé sur cuda ne peut pas être basculé
    simplement en changeant WHISPER_DEVICE — il faut le recharger."""
    global _asr_model, _align_model, _align_metadata, _diarize_pipeline
    global WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    _asr_model = None
    _align_model = None
    _align_metadata = None
    _diarize_pipeline = None
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"
    import torch
    torch.cuda.empty_cache()


def _load_with_cuda_fallback(load_fn, label: str):
    """Tente le chargement sur WHISPER_DEVICE (cuda par défaut) ; si la VRAM est
    insuffisante, bascule sur CPU pour ce modèle et pour tous les chargements suivants de
    cette exécution. Les services persistants du serveur (vLLM, moshi-server) gardent leur
    VRAM allouée en permanence — contrairement à une contention ponctuelle, attendre ou
    réessayer plus tard ne change rien, donc sans ce repli la synchronisation nocturne du
    catalogue échouait systématiquement dès qu'un autre service GPU tournait, ce qui est le
    cas en continu sur cette machine. int8_float16 n'étant pas supporté sur CPU, on bascule
    aussi WHISPER_COMPUTE_TYPE sur 'int8' dans ce cas."""
    global WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    try:
        return load_fn(WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or WHISPER_DEVICE == "cpu":
            raise
        logger.warning(f"{label}: VRAM insuffisante sur {WHISPER_DEVICE}, repli sur CPU pour cette exécution: {e}")
        WHISPER_DEVICE = "cpu"
        WHISPER_COMPUTE_TYPE = "int8"
        import torch
        torch.cuda.empty_cache()
        return load_fn(WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)


def _get_asr_model():
    global _asr_model
    if _asr_model is None:
        import whisperx

        def _load(device, compute_type):
            logger.info(f"Chargement whisper {WHISPER_MODEL} ({compute_type}) sur {device}")
            return whisperx.load_model(WHISPER_MODEL, device, compute_type=compute_type, language="fr")

        _asr_model = _load_with_cuda_fallback(_load, "whisper ASR")
    return _asr_model


def _get_align_model():
    global _align_model, _align_metadata
    if _align_model is None:
        import whisperx

        def _load(device, _compute_type):
            logger.info("Chargement du modèle d'alignement fr")
            return whisperx.load_align_model(language_code="fr", device=device)

        _align_model, _align_metadata = _load_with_cuda_fallback(_load, "modèle d'alignement")
    return _align_model, _align_metadata


def _get_diarize_pipeline():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        import whisperx.diarize

        def _load(device, _compute_type):
            logger.info(f"Chargement du pipeline de diarization pyannote ({DIARIZATION_MODEL})")
            return whisperx.diarize.DiarizationPipeline(model_name=DIARIZATION_MODEL, token=HF_TOKEN, device=device)

        _diarize_pipeline = _load_with_cuda_fallback(_load, "diarization")
    return _diarize_pipeline


def _transcribe_track(abs_path: Path) -> list[dict]:
    """ASR + alignement + diarization sur une piste. Retourne des segments
    {start, end, speaker, text} avec des secondes relatives à la piste."""
    import whisperx
    import whisperx.diarize

    audio = whisperx.load_audio(str(abs_path))
    asr_result = _get_asr_model().transcribe(audio, batch_size=WHISPER_BATCH_SIZE, language="fr")

    align_model, align_metadata = _get_align_model()
    aligned = whisperx.align(asr_result["segments"], align_model, align_metadata, audio, WHISPER_DEVICE)

    diarize_df = _get_diarize_pipeline()(audio)
    result = whisperx.diarize.assign_word_speakers(diarize_df, aligned)

    return [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]


async def transcribe_pending(story_id: int | None = None) -> dict:
    n_tracks = n_errors = n_stories = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        query = (
            "SELECT t.id, t.file_path FROM tracks t JOIN stories s ON s.id = t.story_id "
            "WHERE t.status != 'transcribed' AND s.status = 'tracks_catalogued' "
        )
        params: tuple = ()
        if story_id is not None:
            query += "AND t.story_id = ? "
            params = (story_id,)
        query += "ORDER BY t.story_id, t.order_index"
        async with conn.execute(query, params) as cur:
            rows = await cur.fetchall()

        for row in rows:
            abs_path = CONTES_ROOT / row["file_path"]
            try:
                segments = _transcribe_track(abs_path)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and WHISPER_DEVICE != "cpu":
                    # L'OOM peut survenir pendant l'inférence elle-même (transcribe/align/
                    # diarize), pas seulement au chargement — un modèle déjà en VRAM ne
                    # bascule pas tout seul, il faut l'abandonner et le recharger sur CPU.
                    logger.warning(f"transcribe: VRAM insuffisante sur {abs_path}, repli sur CPU pour le reste de cette exécution: {e}")
                    _reset_models_to_cpu()
                    try:
                        segments = _transcribe_track(abs_path)
                    except Exception as e2:
                        logger.error(f"transcribe: échec {abs_path} même après repli CPU: {e2}")
                        n_errors += 1
                        continue
                else:
                    logger.error(f"transcribe: échec {abs_path}: {e}")
                    n_errors += 1
                    continue
            except Exception as e:
                logger.error(f"transcribe: échec {abs_path}: {e}")
                n_errors += 1
                continue

            for seg in segments:
                await conn.execute(
                    "INSERT INTO transcript_segments (track_id, start_seconds, end_seconds, speaker_label, text) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (row["id"], seg["start"], seg["end"], seg["speaker"], seg["text"]),
                )
            await conn.execute("UPDATE tracks SET status = 'transcribed' WHERE id = ?", (row["id"],))
            await conn.commit()
            n_tracks += 1
            logger.info(f"transcribe: piste {row['id']} ok ({len(segments)} segments)")

        done_query = (
            "SELECT s.id FROM stories s WHERE s.status = 'tracks_catalogued' "
            "AND NOT EXISTS (SELECT 1 FROM tracks t WHERE t.story_id = s.id AND t.status != 'transcribed') "
        )
        done_params: tuple = ()
        if story_id is not None:
            done_query += "AND s.id = ? "
            done_params = (story_id,)
        async with conn.execute(done_query, done_params) as cur:
            story_ids = [r["id"] for r in await cur.fetchall()]
        for sid in story_ids:
            await conn.execute("UPDATE stories SET status = 'transcribed' WHERE id = ?", (sid,))
            n_stories += 1
        await conn.commit()

    result = {"tracks_transcribed": n_tracks, "errors": n_errors, "stories_done": n_stories}
    logger.info(f"transcribe: {result}")
    return result
