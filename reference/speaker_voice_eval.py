import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import numpy as np

import db

logger = logging.getLogger(__name__)

CONTES_ROOT = Path(os.environ.get("CONTES_ROOT", "/contes"))
# Assez pour un embedding stable sans charger tout un rôle sur une histoire entière — au
# delà, le gain de stabilité est marginal face au coût de calcul.
MAX_SEGMENT_SECONDS = float(os.environ.get("EVAL_MAX_SEGMENT_SECONDS", "60"))
# Similarité cosinus minimale pour considérer deux échantillons de voix comme la même
# personne. Calibré par validation manuelle sur un narrateur confirmé couvrant 14 titres
# (James_La_Grosse_Peche, Le_BGG, Contes_De_La_Rue_Broca, Scooby-Doo, etc., tous listés
# "Histoires Beau frere Anne" ou apparentés) : à 0.65 (liaison moyenne), les 109 voix de
# ce narrateur se regroupent en un seul cluster, sans mélanger les 4 titres confirmés
# appartenir à un narrateur différent (Ivanhoe, Sindbad le marin, Histoire de Lustucru,
# Les heros fabuleux du moyen age). Un seuil plus haut (0.85) fragmente ce même narrateur
# en dizaines de singletons — la variabilité interne d'une voix (personnages différents,
# séances distinctes) dépasse parfois l'écart qu'on voudrait utiliser comme frontière.
CLUSTER_THRESHOLD = float(os.environ.get("EVAL_CLUSTER_THRESHOLD", "0.65"))
SAMPLE_RATE = 16000

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier
        logger.info("eval: chargement speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN)")
        _encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/data/speechbrain_ecapa",
        )
    return _encoder


async def _all_voice_entries(conn: aiosqlite.Connection) -> list[dict]:
    """Une ligne par voix atomique réellement détectée par la diarization — (story_id,
    track_id, speaker_label) — quel que soit le rôle (Narrateur ou personnage)."""
    async with conn.execute(
        """
        SELECT sm.story_id, sm.track_id, sm.speaker_label, sm.character_name,
               s.title, s.author, t.file_path
        FROM speaker_map sm
        JOIN stories s ON s.id = sm.story_id
        JOIN tracks t ON t.id = sm.track_id
        WHERE s.status IN ('speakers_identified', 'summarized', 'ready')
        """
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def _segment_times(conn: aiosqlite.Connection, track_id: int, speaker_label: str) -> list[tuple[float, float]]:
    async with conn.execute(
        "SELECT start_seconds, end_seconds FROM transcript_segments "
        "WHERE track_id = ? AND speaker_label = ? ORDER BY start_seconds",
        (track_id, speaker_label),
    ) as cur:
        rows = await cur.fetchall()
    return [(r["start_seconds"], r["end_seconds"]) for r in rows]


def _extract_segment_audio(file_path: str, times: list[tuple[float, float]]) -> tuple[np.ndarray, float]:
    """Concatène les segments d'UNE seule voix atomique (un (track_id, speaker_label)),
    jusqu'à MAX_SEGMENT_SECONDS."""
    import whisperx

    audio_full = whisperx.load_audio(str(CONTES_ROOT / file_path))
    chunks: list[np.ndarray] = []
    total = 0.0
    for start, end in times:
        if total >= MAX_SEGMENT_SECONDS:
            break
        remaining = MAX_SEGMENT_SECONDS - total
        s_idx = int(start * SAMPLE_RATE)
        e_idx = min(int(end * SAMPLE_RATE), s_idx + int(remaining * SAMPLE_RATE))
        if e_idx <= s_idx:
            continue
        chunks.append(audio_full[s_idx:e_idx])
        total += (e_idx - s_idx) / SAMPLE_RATE

    if not chunks:
        return np.array([], dtype=np.float32), 0.0
    return np.concatenate(chunks), total


def _embed(audio: np.ndarray) -> np.ndarray:
    import torch

    signal = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        embedding = _get_encoder().encode_batch(signal)
    return embedding.squeeze().cpu().numpy()


async def compute_voice_embeddings(limit: int | None = None) -> dict:
    """Calcule et met en cache (table eval_voice_embeddings) l'empreinte vocale de CHAQUE
    voix atomique (track_id, speaker_label) détectée par la diarization, quel que soit le
    rôle — étape lente (chargement audio + inférence), séparée du clustering pour ne pas
    recalculer à chaque nouvelle tentative de regroupement."""
    n_done = n_skipped = n_errors = 0
    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        entries = await _all_voice_entries(conn)
        async with conn.execute("SELECT story_id, track_id, speaker_label FROM eval_voice_embeddings") as cur:
            already = {(r["story_id"], r["track_id"], r["speaker_label"]) for r in await cur.fetchall()}

        for e in entries:
            key = (e["story_id"], e["track_id"], e["speaker_label"])
            if key in already:
                n_skipped += 1
                continue
            if limit is not None and n_done >= limit:
                break
            try:
                times = await _segment_times(conn, e["track_id"], e["speaker_label"])
                audio, seconds = _extract_segment_audio(e["file_path"], times)
                if seconds < 2.0:
                    logger.warning(
                        f"eval: histoire {e['story_id']} piste {e['track_id']} {e['speaker_label']} "
                        f"({e['character_name']}) — trop peu d'audio ({seconds:.1f}s), ignoré"
                    )
                    n_errors += 1
                    continue
                embedding = _embed(audio)
                await conn.execute(
                    "INSERT OR REPLACE INTO eval_voice_embeddings "
                    "(story_id, track_id, speaker_label, character_name, embedding, seconds, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        e["story_id"], e["track_id"], e["speaker_label"], e["character_name"],
                        embedding.astype(np.float32).tobytes(), seconds,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                await conn.commit()
                n_done += 1
                logger.info(
                    f"eval: histoire {e['story_id']} piste {e['track_id']} {e['speaker_label']} "
                    f"({e['character_name']}) — embedding calculé sur {seconds:.1f}s"
                )
            except Exception as ex:
                logger.error(f"eval: échec {e['story_id']}/{e['track_id']}/{e['speaker_label']}: {ex}")
                n_errors += 1

    result = {"computed": n_done, "skipped_cached": n_skipped, "errors": n_errors}
    logger.info(f"eval compute_voice_embeddings: {result}")
    return result


def _cluster_by_embedding(members: list[dict], threshold: float) -> list[list[dict]]:
    """Regroupe des voix (dicts contenant une clé 'emb' = embedding normalisé) par liaison
    moyenne sur similarité cosinus. Factorisé hors de cluster_voices() pour être réutilisé
    par reference.narrator_identity, qui re-scinde localement un cluster à un seuil plus
    strict quand le LLM n'arrive pas à en déduire une identité avec confiance haute (signe
    que le cluster mélange probablement plusieurs narrateurs — voir ce module)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(members) < 2:
        return [members] if members else []

    embs = np.stack([m["emb"] for m in members])
    sim = embs @ embs.T
    dist = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    groups: dict[int, list[dict]] = {}
    for member, label in zip(members, labels):
        groups.setdefault(int(label), []).append(member)
    return list(groups.values())


async def cluster_voices(threshold: float = CLUSTER_THRESHOLD) -> dict:
    """Regroupe TOUTES les voix atomiques en cache par pure similarité d'embedding —
    aucune métadonnée (auteur, personnage) n'intervient dans le regroupement lui-même.
    Une fois les clusters formés, chacun est annoté avec les infos déjà connues sur ses
    membres (histoire, rôle, auteur déclaré) : c'est cette étape, et seulement celle-ci,
    qui répond à 'qui est-ce'. Un cluster à un seul membre est une voix pas encore
    recroisée ailleurs dans le catalogue.

    Liaison MOYENNE (scipy, method='average') — pas liaison simple, qui chaîne (union-find
    dès qu'UNE SEULE paire dépasse le seuil : deux personnes sans rapport se retrouvent dans
    le même cluster via un intermédiaire, observé en pratique — un cluster de 135 membres
    mélangeant des auteurs sans aucun rapport). Pas liaison complète non plus : elle exige
    que TOUTE paire dans un cluster dépasse le seuil, ce qui casse un vrai narrateur dès
    qu'UNE SEULE de ses apparitions est un peu atypique (personnage différent, séance
    d'enregistrement différente) — validé en pratique : à seuil 0.85 un narrateur confirmé
    sur 14 titres (233 voix) se fragmente en dizaines de singletons. La liaison moyenne
    (distance moyenne entre TOUTES les paires inter-clusters) tolère cette variabilité
    interne sans pour autant chaîner — validée manuellement : à 0.65 elle regroupe les 109
    voix retenues de ce même narrateur en un seul cluster, sans y mélanger les 4 titres
    confirmés appartenir à quelqu'un d'autre (voir commentaire sur CLUSTER_THRESHOLD)."""
    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            """
            SELECT e.story_id, e.track_id, e.speaker_label, e.character_name, e.embedding,
                   s.title, s.author
            FROM eval_voice_embeddings e JOIN stories s ON s.id = e.story_id
            """
        ) as cur:
            rows = await cur.fetchall()

    if len(rows) < 2:
        return {"error": "pas assez de voix en cache — lancer compute_voice_embeddings d'abord"}

    n = len(rows)
    entries = []
    for r in rows:
        raw = np.frombuffer(r["embedding"], dtype=np.float32)
        entries.append({**dict(r), "emb": raw / np.linalg.norm(raw)})

    groups = _cluster_by_embedding(entries, threshold)

    clusters = []
    for members in groups:
        appearances = [
            {
                "story_id": m["story_id"],
                "title": m["title"],
                "author": m["author"],
                "character_name": m["character_name"],
                "track_id": m["track_id"],
                "speaker_label": m["speaker_label"],
            }
            for m in members
        ]
        authors = sorted({a["author"] for a in appearances})
        roles = sorted({a["character_name"] for a in appearances})
        clusters.append({
            "size": len(members),
            "authors_seen": authors,
            "roles_seen": roles,
            "appearances": appearances,
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)

    result = {
        "total_voices": n,
        "threshold": threshold,
        "n_clusters": len(clusters),
        "n_multi_member_clusters": sum(1 for c in clusters if c["size"] >= 2),
        "clusters": clusters,
    }
    logger.info(
        f"eval cluster_voices: {n} voix -> {len(clusters)} clusters "
        f"({result['n_multi_member_clusters']} avec >=2 membres)"
    )
    return result


async def run(limit: int | None = None, threshold: float = CLUSTER_THRESHOLD) -> dict:
    """Point d'entrée unique : calcule les embeddings manquants puis regroupe."""
    compute_result = await compute_voice_embeddings(limit=limit)
    cluster_result = await cluster_voices(threshold=threshold)
    return {"compute": compute_result, "clusters": cluster_result}
