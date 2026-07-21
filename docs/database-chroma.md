# Base vectorielle — ChromaDB

Deuxième base du système, dédiée à la recherche sémantique (embeddings).
Stockage persistant local (`PersistentClient`), chemin `/data/chroma`
(même volume Docker que SQLite). Modèle d'embedding :
`paraphrase-multilingual-MiniLM-L12-v2` (multilingue, adapté au français).
Gérée entièrement par `chroma_store.py`.

Vérifié contre l'instance réelle le 2026-07-19 : **une seule collection**
nommée `contes`, **3681 documents** au total, espace de similarité cosine.

## Principe : une collection, plusieurs "kinds"

Contrairement à SQLite, il n'y a **pas de tables séparées** : tous les
documents vivent dans la même collection Chroma et sont distingués par un
champ de métadonnée `kind`. C'est ce choix de conception qui rend le champ
`kind` structurellement important — toute requête doit filtrer dessus
(`where={"kind": ...}`) pour ne pas mélanger des granularités différentes.

| `kind` | Nb réel | Rôle |
|---|---|---|
| `story_summary` | 327 | Un par histoire — résumé général (titre + résumé court + résumé long concaténés). |
| `story_keywords` | 327 | Un par histoire — embedding séparé sur la liste de mots-clés seule. |
| `period_summary` | 1502 | Un par période (~3 min) — mini-résumé. |
| `period_transcript` | 1502 | Un par période — texte brut diarizé (rend la diarization elle-même cherchable). |
| `theme_class` | 23 | Un par classe thématique (label + description). |

⚠️ `story_summary`/`story_keywords` = 327 alors que SQLite ne compte que 295
histoires `ready` (327 − 295 = 32, exactement le nombre d'histoires
`missing`). `theme_class` = 23 alors que SQLite `theme_classes` n'a que 10
lignes. Ces deux écarts sont documentés comme anomalies connues dans
[known-issues.md](./known-issues.md) — Chroma n'est jamais purgé quand
`stories.status` change ou quand `theme_classes` est reconstruite.

## Identifiant des documents

Chaque document a un `id` = hash MD5 de parties concaténées, pour être
déterministe et réutilisable en `upsert` (ex: rejouer `embed` sur la même
histoire remplace le document au lieu d'en créer un doublon) :
- `story_summary` : `md5("story:{story_id}")`
- `story_keywords` : `md5("story_keywords:{story_id}")`
- `period_summary` : `md5("period:{story_id}:{period_index}:summary")`
- `period_transcript` : `md5("period:{story_id}:{period_index}:transcript")`
- `theme_class` : `md5("theme_class:{class_id}")`

## Champs de métadonnées par `kind`

### `story_summary`
| Champ | Type | Description |
|---|---|---|
| `kind` | str | `"story_summary"` |
| `story_id` | int | FK logique vers `stories.id` (SQLite). |
| `period_index` | int | Toujours `-1` (convention "pas une période"). |
| `title` | str | Copie du titre au moment de l'embedding (peut dater si le titre a été corrigé depuis par `reconcile_titles`). |
| `author` | str | Copie de `stories.author` au moment de l'embedding — **ne contient ni `narrator` ni `literary_author`**, ces champs n'existaient pas encore quand ce code a été écrit ; ré-enrichis a posteriori côté application via `db.get_narrator_info()`. |
| `global_start_seconds` | float | Toujours `0.0`. |
| `global_end_seconds` | float | Durée totale de l'histoire. |
| `duration_int` | int | Durée totale arrondie (secondes) — sert de filtre numérique `where` (Chroma ne fait pas de comparaison sur float dans les métadonnées de la même façon). |

Contenu du document (texte embeddé) : `"{title}\n{short_summary}\n{long_summary}"`.

### `story_keywords`
Mêmes métadonnées que `story_summary`. Contenu du document : mots-clés
joints par virgule, ex: `"famille, maison, poésie, imaginaire, chanson
rythmique"`. Existe uniquement si `stories.keywords` est non vide. Un
embedding **séparé** plutôt que fusionné à `story_summary` : un détail
littéral et rare (ex: "un arbre à pain") est dilué dans un long résumé en
prose mais ressort dans une courte liste de mots-clés.

### `period_summary` / `period_transcript`
| Champ | Type | Description |
|---|---|---|
| `kind` | str | `"period_summary"` ou `"period_transcript"` |
| `story_id` | int | FK logique. |
| `period_index` | int | Index de la période. |
| `global_start_seconds` / `global_end_seconds` | float | Bornes globales de la période. |

Contenu du document : `summary_text` (pour `period_summary`) ou
`raw_transcript_text` (pour `period_transcript`) — chacun n'est upserté que
si non vide.

### `theme_class`
| Champ | Type | Description |
|---|---|---|
| `kind` | str | `"theme_class"` |
| `class_id` | int | FK logique vers `theme_classes.id`. |
| `label` | str | Copie du libellé. |

Contenu du document : `"{label}\n{description}"`.

## Fonctions de recherche exposées

- **`search_stories(query, n_results, min/max_duration_seconds)`** — interroge
  `story_summary` + `story_keywords` ensemble, dédoublonne par `story_id` en
  gardant le meilleur score, applique un filtre optionnel sur `duration_int`.
- **`search_moments(query, story_id=None, n_results)`** — interroge
  `period_summary` + `period_transcript`, filtrable sur une histoire précise.
- **`search_theme_classes(query, n_results)`** — interroge uniquement
  `theme_class`, sert à router une requête libre (ex: "des histoires de
  pirates") vers la classe thématique la plus proche.

Score retourné = `1 - distance_cosine` (donc plus proche de 1 = plus
pertinent). Utilisé par `contes_tools.py` en complément (jamais en
remplacement) des résultats SQLite (auteur exact, FTS lexical) — voir
`contes_tools.py::_search_stories` pour l'ordre de priorité entre les 4
voies (auteur > sémantique > lexical > thème).

## Relation avec SQLite

ChromaDB ne contient **aucune donnée qui n'existe pas déjà dans SQLite** —
c'est un **index de recherche dérivé**, pas une source de vérité. Toute
correction doit se faire côté SQLite puis être répercutée par un ré-`embed`
(`reference/embed.py` / `reference/pipeline.py --stage embed`). Les seules
données présentes uniquement dans les métadonnées Chroma et pas nommément en
colonne SQLite sont des **copies dénormalisées** (`title`, `author`,
`duration_int`...) prises au moment de l'upsert — à ne jamais considérer
comme à jour sans vérification côté SQLite.
