#!/usr/bin/env bash
# Réindexe le catalogue contes-agent (scan + pipeline complet, only_new) en libérant
# temporairement la VRAM du GPU 0 partagé : voxcpm2-api en est le plus gros
# consommateur (~10,7 Go sur 16 Go), suffisant pour empêcher whisperx de charger et le
# faire basculer sur CPU (voir reference/transcribe.py::_load_with_cuda_fallback) —
# ~7-8 min/piste en CPU contre quelques secondes sur GPU.
#
# voxcpm2-api est TOUJOURS redémarré en sortie (succès, échec, Ctrl-C) via le trap
# ci-dessous, pour ne jamais laisser ce service resté arrêté par accident.
set -uo pipefail

CONTAINER=contes-agent
VOXCPM_CONTAINER=voxcpm2-api

restart_voxcpm() {
    echo "[reindex] Redémarrage de ${VOXCPM_CONTAINER}..."
    docker start "${VOXCPM_CONTAINER}"
}
trap restart_voxcpm EXIT

echo "[reindex] Arrêt de ${VOXCPM_CONTAINER} pour libérer de la VRAM..."
docker stop "${VOXCPM_CONTAINER}"

# Un run précédent (lancé manuellement en mode CPU) peut encore tourner — on le
# stoppe pour repartir sur un chargement GPU propre plutôt que de laisser deux
# processus se disputer les mêmes histoires 'discovered'/'tracks_catalogued'.
EXISTING_PID=$(docker exec "$CONTAINER" pgrep -f "python -m reference.pipeline" || true)
if [ -n "$EXISTING_PID" ]; then
    echo "[reindex] Arrêt du run existant (pid ${EXISTING_PID})..."
    docker exec "$CONTAINER" kill "$EXISTING_PID"
    sleep 2
fi

echo "[reindex] Lancement du pipeline (scan + duration + transcribe + split_stories + identify_speakers + summarize + embed, only_new)..."
docker exec "$CONTAINER" python -m reference.pipeline --stage all --only-new
PIPELINE_EXIT=$?

echo "[reindex] Régénération du wiki..."
docker exec "$CONTAINER" python -m reference.pipeline --stage build_wiki

echo "[reindex] Terminé (code pipeline: ${PIPELINE_EXIT})."
exit "$PIPELINE_EXIT"
