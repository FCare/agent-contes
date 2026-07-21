# Base SQLite — `contes.db`

Source de vérité principale de l'agent. Fichier SQLite unique, chemin `DB_PATH`
(défaut `/data/contes.db`, monté depuis un volume Docker). Schéma défini et
créé par `db.py::init_db()` (appelé à chaque démarrage, idempotent —
`CREATE TABLE IF NOT EXISTS` + `ALTER TABLE ... ADD COLUMN` pour les
migrations). Vérifié contre l'instance réelle du conteneur `contes-agent` le
2026-07-19 : **340 histoires** au total (295 `ready`, 32 `missing`, 12
`excluded`, 1 `grouped`).

## Vue d'ensemble des tables

| Table | Rôle | Lignes (réel) |
|---|---|---|
| `stories` | Une ligne par histoire (l'entité centrale) | 340 |
| `tracks` | Les fichiers audio physiques qui composent une histoire | 1382 |
| `transcript_segments` | Transcription diarizée brute, phrase par phrase | 47 402 |
| `periods` | Découpage en fenêtres de ~3 min avec résumé | 1502 |
| `speaker_map` | Mapping label de diarization → nom de personnage | 2211 |
| `bookmarks` | Position de lecture en cours par histoire | 0 |
| `theme_classes` | Classes thématiques consolidées (vocabulaire découvert) | 10 |
| `story_theme_classes` | Association N-N histoire ↔ thème | 871 |
| `story_cast_verified` | Casting de référence, immuable, source fiable | 216 |
| `narrator_identities` | Identité de narrateur inférée par clustering acoustique + LLM | 312 |
| `narrator_identity_members` | Membres (piste/label) de chaque identité inférée | 1661 |
| `eval_voice_embeddings` | Empreintes vocales ECAPA-TDNN (expérimental) | 2106 |
| `stories_fts` / `periods_fts` | Index plein texte FTS5 (fallback lexical) | dérivées |

---

## `stories`

Une ligne = une histoire (peut regrouper plusieurs pistes audio et plusieurs
dossiers physiques d'origine — voir `merged_folders`).

| Champ | Type | Rempli (ready) | Description |
|---|---|---|---|
| `id` | INTEGER PK | 340/340 | Identifiant unique. |
| `title` | TEXT NOT NULL | 340/340 | Titre de l'histoire. Dérivé du nom de dossier au scan, corrigé ensuite par `reconcile_titles.py` (recherche web) si nécessaire. |
| `author` | TEXT NOT NULL | 340/340 | Valeur **brute** tirée du nom du dossier parent. Ambiguë par construction : désigne tantôt le narrateur, tantôt un label de collection, parfois (piège observé) l'auteur littéraire glissé dans le nom du dossier. **Jamais** utilisée directement comme narrateur ou auteur littéraire fiable — sert seulement de repli de dernier recours. |
| `folder_path` | TEXT UNIQUE NOT NULL | 340/340 | Chemin relatif à `CONTES_ROOT` du dossier source. Unique — si `split_stories` sépare un dossier en plusieurs histoires, suffixé `#1`, `#2`, etc. |
| `merged_folders` | TEXT (JSON array) NOT NULL | 340/340 | Liste des dossiers physiques fusionnés en une seule histoire (ex: CD1/CD2 d'un même livre audio). Exemple réel : `["Carrol, Lewis/Alice au Pays des Merveilles [Disc 1]", "...[Disc 2]"]`. |
| `status` | TEXT NOT NULL, défaut `'discovered'` | — | État d'avancement dans le pipeline de traitement. Voir [section dédiée](#cycle-de-vie-status) plus bas. Seul `'ready'` est exposé aux recherches utilisateur. |
| `short_summary` | TEXT | 295/295 | Résumé court (1-2 phrases), généré par `reference/summarize.py`. |
| `long_summary` | TEXT | 295/295 | Résumé détaillé, même origine. |
| `total_duration_seconds` | REAL | — | Durée totale calculée par `reference/duration.py` (somme des pistes). |
| `created_at` / `updated_at` | TEXT (ISO 8601 UTC) NOT NULL | 340/340 | Horodatage de création/dernière modification de la ligne. |
| `keywords` | TEXT (JSON array de strings) | 295/295 | Mots-clés extraits par le LLM au résumé, ex: `["famille", "maison", "poésie"]`. Sert à un embedding Chroma séparé (`story_keywords`) et au fallback FTS. |
| `literary_author` | TEXT | 45/295 | Auteur **littéraire** original (qui a **écrit** l'histoire), enrichi par recherche web (`reference/enrich_web.py`). `NULL` = pas encore enrichi ; `""` = enrichi mais rien trouvé ; sinon le nom (ex: `"Charles Perrault"`). Distinct de `narrator` (qui **lit**). |
| `literary_info` | TEXT (JSON) | 293/295 | Objet `{"publication_year": ..., "notes": ..., "sources": [{"title", "url"}, ...]}` produit par le même enrichissement web. ⚠️ Voir [known-issues.md](./known-issues.md) — contient des sources non pertinentes (voire inappropriées) sur certaines lignes. |
| `narrator` | TEXT | 137/295 | Nom confirmé/corrigé du narrateur (qui **lit** l'histoire à voix haute), ex: `"Romane Bohringer"`. Distinct de `author` (brut, potentiellement mal orthographié, ex: `"Roman Boringher"`). Alimenté soit par `story_cast_verified` (confiance haute), soit par `reference/narrator_identity.py` (clustering acoustique, uniquement si confiance `'haute'`). |
| `age_range` | TEXT | 293/295 | Une valeur parmi un vocabulaire fixe : `tout-petit`, `petit`, `enfant`, `grand-enfant` (voir `reference/classify.py::AGE_RANGES`). Filtrable exactement. |
| `mood_tags` | TEXT (JSON array) | 294/295 | Sous-ensemble d'un vocabulaire fixe de 12 tags : `peur, aventure, humour, tendresse, animaux, magie, amitie, educatif, quotidien, classique, voyage, mystere` (voir `MOOD_TAGS`). |
| `raw_theme_label` | TEXT | 291/295 | Étape intermédiaire de classification thématique "libre" : thème propre à cette histoire en texte libre, proposé par le LLM avant consolidation en classes communes. |
| `theme_class_id` | INTEGER FK → `theme_classes.id` | 294/295 | **Legacy, non utilisé** en pratique — une histoire peut relever de plusieurs thèmes, gérés via la table N-N `story_theme_classes`. Laissé en place sans y toucher. |

### Cycle de vie (`status`)

Le pipeline (`reference/pipeline.py`) fait progresser une histoire à travers ces
états, dans l'ordre (`STAGES`) :

```
discovered → tracks_catalogued → transcribed → grouped → speakers_identified → summarized → ready
```

- **`discovered`** (`scan.py`) : dossier détecté sur disque, ligne créée.
- **`tracks_catalogued`** (`duration.py`) : durée de chaque piste calculée.
- **`transcribed`** (`transcribe.py`) : diarization + transcription de toutes les pistes terminées.
- **`grouped`** (`split_stories.py`) : regroupement/scission en histoires logiques (un dossier peut contenir plusieurs histoires, ou une histoire s'étaler sur plusieurs dossiers).
- **`speakers_identified`** (`identify_speakers.py`) : labels `SPEAKER_XX` mappés à des noms de personnages (`speaker_map`).
- **`summarized`** (`summarize.py`) : résumés + mots-clés générés.
- **`ready`** (`embed.py`) : embeddings Chroma calculés — l'histoire devient visible pour la recherche utilisateur.

États hors chemin normal :
- **`excluded`** : dossier dans `reference/excluded_folders.txt`, jamais traité (12 en base).
- **`missing`** : dossier disparu du disque lors d'un re-scan ; restauré à `ready` automatiquement s'il réapparaît (32 en base — voir [known-issues.md](./known-issues.md), ces histoires restent cherchables par erreur).

**Toutes les requêtes de catalogue/recherche filtrent sur `status = 'ready'`**,
sauf mention contraire explicite dans le code.

---

## `tracks`

Un fichier audio physique. Une histoire (`stories`) a une ou plusieurs pistes,
ordonnées.

| Champ | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Identifiant. Utilisé aussi dans l'URL de streaming (`/stream/{id}`). |
| `story_id` | INTEGER FK → `stories.id` (CASCADE) | Histoire parente. |
| `order_index` | INTEGER NOT NULL | Position dans la playlist de l'histoire (0, 1, 2...). |
| `file_path` | TEXT NOT NULL, UNIQUE | Chemin absolu du fichier audio. Unique **par chemin** (pas par `story_id`) pour survivre à une réattribution lors d'un futur re-scan. |
| `duration_seconds` | REAL | Durée de la piste, calculée par `duration.py`. |
| `cumulative_start_seconds` | REAL | Offset cumulé de cette piste dans la timeline globale de l'histoire (piste 2 démarre à la fin de la piste 1, etc.). Sert à convertir une position "globale" (utilisée par `periods`, `bookmarks`) en `(track_id, offset local)`. |
| `status` | TEXT NOT NULL, défaut `'discovered'` | `discovered` → `duration_known` (transitoire) → `transcribed`. Réel : 165 `discovered`, 1217 `transcribed` (aucune en `duration_known` actuellement, état de passage). |

---

## `transcript_segments`

Sortie brute de la diarization + transcription, phrase par phrase. 47 402
lignes réelles — la table la plus volumineuse.

| Champ | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Identifiant. |
| `track_id` | INTEGER FK → `tracks.id` (CASCADE) | Piste source. |
| `start_seconds` / `end_seconds` | REAL NOT NULL | Bornes temporelles **locales à la piste** (pas globales à l'histoire). |
| `speaker_label` | TEXT NOT NULL | Label brut de diarization, ex: `SPEAKER_00`, `SPEAKER_01`... Stable **au sein d'une piste uniquement** — le même label dans deux pistes différentes de la même histoire ne désigne pas forcément le même personnage (la diarization tourne piste par piste). |
| `text` | TEXT NOT NULL | Texte transcrit du segment. Exemple réel : `"Aujourd'hui, j'ai rencontré le grand méchant loup."` |

---

## `speaker_map`

Résout un `speaker_label` brut (par piste) en nom de personnage lisible.
Table réécrite par `identify_speakers.py` (LLM sur le texte du transcript).

| Champ | Type | Description |
|---|---|---|
| `story_id` | INTEGER FK → `stories.id` (CASCADE), PK (1/3) | Histoire. |
| `track_id` | INTEGER FK → `tracks.id` (CASCADE), PK (2/3) | Piste — **fait partie de la clé** car un label n'est stable qu'au sein d'une piste. |
| `speaker_label` | TEXT, PK (3/3) | Label brut (`SPEAKER_00`...). |
| `character_name` | TEXT NOT NULL | Nom résolu, ex: `"Narrateur"`, `"Palomita"`, `"Chœur d'enfants / Bruitages"`. |

Exemple réel (histoire 13, personnages multiples) :
```
(13, 288, SPEAKER_00, "Chœur d'enfants / Bruitages")
(13, 288, SPEAKER_01, "Palomita")
(13, 288, SPEAKER_02, "Rimitaïta")
```

---

## `periods`

Découpage de chaque histoire en fenêtres de `PERIOD_SECONDS` (180s par
défaut, configurable via env), utilisées pour la recherche "de moment précis"
et pour naviguer dans une histoire longue.

| Champ | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Identifiant. |
| `story_id` | INTEGER FK → `stories.id` (CASCADE) | Histoire. |
| `period_index` | INTEGER NOT NULL, UNIQUE avec `story_id` | Index de la fenêtre (0, 1, 2...). |
| `global_start_seconds` / `global_end_seconds` | REAL NOT NULL | Bornes **globales** (toutes pistes confondues) de la fenêtre. |
| `raw_transcript_text` | TEXT | Texte brut diarizé concaténé de la fenêtre — source la plus fidèle pour la classification (mood/thème), car les résumés édulcorent volontiers les détails marquants. |
| `summary_text` | TEXT | Mini-résumé de la fenêtre. |

---

## `bookmarks`

Position de lecture en cours, une par histoire (l'utilisateur ne peut avoir
qu'un bookmark actif par histoire). **0 ligne actuellement** — feature
existante mais pas (encore) utilisée en pratique par l'agent vocal.

| Champ | Type | Description |
|---|---|---|
| `story_id` | INTEGER PK, FK → `stories.id` (CASCADE) | Histoire (une seule position par histoire). |
| `position_seconds` | REAL NOT NULL | Position globale en secondes. |
| `updated_at` | TEXT NOT NULL | Horodatage. |

Une histoire à moins de `BOOKMARK_FINISH_THRESHOLD_SECONDS` (30s par défaut)
de sa fin est considérée terminée : reprendre repart du début.

---

## `theme_classes` / `story_theme_classes`

Classification thématique "libre" : les thèmes ne sont **pas** une liste
fixée à l'avance, mais découverts depuis le contenu réel du catalogue puis
consolidés (`reference/classify.py::consolidate_theme_classes`, cible 8 à 25
classes). Une histoire peut relever de plusieurs thèmes (many-to-many).

`theme_classes` :
| Champ | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Identifiant. |
| `label` | TEXT NOT NULL | Nom court, ex: `"Ruse et Astuce"`, `"Magie et Enchantements"`. |
| `description` | TEXT NOT NULL | Description plus longue, embeddée séparément dans Chroma pour le routage sémantique d'une requête libre. |
| `created_at` | TEXT NOT NULL | Horodatage. |

`story_theme_classes` (table de jonction) :
| Champ | Type | Description |
|---|---|---|
| `story_id` | INTEGER FK, PK (1/2) | |
| `theme_class_id` | INTEGER FK, PK (2/2) | |

**10 classes réelles actuellement** (id 14 à 23 — voir
[known-issues.md](./known-issues.md) pour la table `theme_classes` **remplacée
en entier** à chaque `consolidate_themes` : les anciennes classes 1-13 ont
disparu de SQLite mais restent orphelines côté Chroma).

---

## `story_cast_verified`

Casting de référence **immuable**, jamais écrasé par le clustering
acoustique. Priorité de fiabilité des sources, de la plus à la moins fiable :
`human` (confirmé par un humain) > `id3_metadata` (tag TPE1 des fichiers
audio) > `artwork_vision` (LLM vision sur la pochette embarquée). Une fois
qu'une entrée existe pour une histoire, `narrator_identity.py` ne la retraite
plus jamais. 216 lignes réelles : 203 `id3_metadata`, 13 `artwork_vision`, 0
`human` actuellement.

| Champ | Type | Description |
|---|---|---|
| `story_id` | INTEGER FK (CASCADE), PK (1/2) | Histoire. |
| `name` | TEXT NOT NULL, PK (2/2) | Nom de la personne (narrateur ou personnage/rôle). |
| `role` | TEXT | Rôle, si connu (souvent `NULL` en pratique). |
| `is_narrator` | INTEGER (bool) NOT NULL, défaut 0 | 1 si cette personne est le narrateur principal. |
| `source` | TEXT NOT NULL, défaut `'human'` | `human` / `id3_metadata` / `artwork_vision`. |
| `created_at` | TEXT NOT NULL | Horodatage. |

---

## `narrator_identities` / `narrator_identity_members`

Résultat **expérimental** de `reference/narrator_identity.py` : identité de
narrateur déduite par LLM à partir des chemins de fichiers d'un cluster
acoustique de voix (voir `eval_voice_embeddings`). **Table intégralement
remplacée** (DELETE + réinsertion) à chaque exécution — le clustering n'est
pas stable d'une run à l'autre. Quand `confidence = 'haute'`, la même
exécution pousse aussi le résultat vers `stories.narrator` (production).
Exclut automatiquement les histoires déjà présentes dans
`story_cast_verified`.

`narrator_identities` (312 lignes réelles : 192 `faible`, 80 `haute`, 40
`moyenne`) :
| Champ | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Identifiant. |
| `inferred_name` | TEXT NOT NULL | Nom déduit, ex: `"Anne"`. |
| `confidence` | TEXT NOT NULL | `haute` / `moyenne` / `faible`. Seules `haute` et `moyenne` sont exposées à `get_voices_for_story`. |
| `is_professional` | INTEGER (bool) NOT NULL | Le LLM estime si c'est un narrateur professionnel. |
| `reasoning` | TEXT NOT NULL | Justification textuelle du LLM (traçabilité). |
| `cluster_threshold` | REAL NOT NULL | Seuil de similarité acoustique utilisé pour ce cluster. |
| `created_at` | TEXT NOT NULL | Horodatage. |

`narrator_identity_members` (1661 lignes) — quels (histoire, piste, label) composent ce cluster :
| Champ | Type | Description |
|---|---|---|
| `identity_id` | INTEGER FK → `narrator_identities.id` (CASCADE) | |
| `story_id` / `track_id` / `speaker_label` | INTEGER / INTEGER / TEXT, PK composite | Identifie la voix physique concernée. |

---

## `eval_voice_embeddings`

Cache **expérimental** (hors production) pour `reference/speaker_voice_eval.py` :
empreintes vocales ECAPA-TDNN, en complément du mapping locuteur actuel (LLM
sur texte, qui reste la référence). Granularité = unité atomique d'une vraie
voix physique : `(track_id, speaker_label)`, jamais toute une histoire (un
même personnage, notamment le narrateur, peut recouvrir plusieurs voix
distinctes selon la piste). 2106 lignes réelles.

| Champ | Type | Description |
|---|---|---|
| `story_id` / `track_id` / `speaker_label` | PK composite | Voix physique identifiée. |
| `character_name` | TEXT NOT NULL | Nom au moment de la capture (annotation, pas utilisé pour construire le clustering). |
| `embedding` | BLOB NOT NULL | Vecteur d'empreinte vocale (768 float, sérialisé). |
| `seconds` | REAL NOT NULL | Durée de parole utilisée pour calculer l'empreinte. |
| `created_at` | TEXT NOT NULL | Horodatage. |

---

## Index de recherche plein texte (FTS5)

Fallback lexical (mots-clés exacts) en complément des embeddings sémantiques
Chroma : une requête très factuelle/rare (ex: `"un arbre à pain"`) peut être
diluée dans un vecteur sémantique, alors qu'une correspondance mot-à-mot la
retrouve à coup sûr. Tokenizer `unicode61 remove_diacritics 2` (insensible
aux accents). Une liste de mots vides français + domaine (`conte`,
`histoire`...) est retirée des requêtes construites depuis du langage
naturel (voir `db.py::_STOPWORDS_FR`).

- **`stories_fts`** : colonnes `title`, `short_summary`, `long_summary`,
  `keywords_text` (indexées), `story_id` (UNINDEXED). Synchronisée à chaque
  écriture de résumé (`sync_story_fts`).
- **`periods_fts`** : colonnes `summary_text`, `raw_transcript_text`
  (indexées), `story_id` + `period_index` (UNINDEXED). Synchronisée à chaque
  écriture de période (`sync_period_fts`).

Ces tables virtuelles génèrent automatiquement des tables internes SQLite
(`*_content`, `*_docsize`, `*_idx`, `*_config`, `*_data`) — implémentation
FTS5, pas à documenter individuellement.
