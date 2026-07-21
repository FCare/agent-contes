# Documentation des bases de données — Contes Agent

Index de la documentation générée le 2026-07-19, vérifiée contre l'instance
réelle du conteneur `contes-agent` (schéma + statistiques de remplissage),
pas seulement le code source.

Le système repose sur **deux bases complémentaires** :

1. **[database-sqlite.md](./database-sqlite.md)** — `contes.db` (SQLite),
   source de vérité : histoires, pistes audio, transcriptions, résumés,
   personnages, casting, thèmes, bookmarks. 12 tables métier + 2 index FTS5.
2. **[database-chroma.md](./database-chroma.md)** — collection ChromaDB,
   index de recherche sémantique **dérivé** de SQLite (jamais de donnée
   source unique côté Chroma).

Voir aussi :
- **[concepts-personnes.md](./concepts-personnes.md)** — clarifie les 5
  notions distinctes (auteur brut, auteur littéraire, narrateur, voix,
  personnage) qui reviennent dans presque toutes les tables et qui sont
  fréquemment confondues.
- **[known-issues.md](./known-issues.md)** — 3 anomalies constatées sur les
  données réelles (contenu web inapproprié, désynchronisation Chroma/SQLite
  sur les histoires disparues et sur les classes thématiques). Non
  corrigées, documentées pour référence.

## Chiffres clés (2026-07-19)

- 340 histoires en base, dont 295 `ready` (visibles en recherche), 32
  `missing`, 12 `excluded`, 1 `grouped`.
- 1382 pistes audio, 47 402 segments de transcription, 1502 périodes de ~3 min.
- 3681 documents vectoriels dans Chroma.
- 10 classes thématiques actives ; 216 entrées de casting vérifié.

## Comment cette doc a été produite

Schéma lu depuis `db.py::init_db()` (CREATE/ALTER TABLE) et
`chroma_store.py` (métadonnées d'upsert), puis **vérifié en interrogeant
l'instance réelle** (`docker exec contes-agent python3 ...`, lecture seule)
pour confirmer le schéma exact et mesurer le taux de remplissage réel de
chaque champ optionnel. Toute évolution du schéma (nouvelle colonne,
nouvelle table) doit être répercutée ici.
