# Concepts transverses : auteur, narrateur, voix, personnages

Cinq notions différentes désignent des "personnes" liées à une histoire,
réparties sur plusieurs tables. Elles sont fréquemment confondues dans le
code source lui-même (d'où les nombreux commentaires défensifs) — ce
document les rassemble en un seul endroit pour clarifier une bonne fois pour
toutes qui est qui.

## Les 5 notions

| Notion | Question à laquelle ça répond | Champ / table | Fiabilité |
|---|---|---|---|
| **`author`** | Nom brut trouvé dans le dossier | `stories.author` | Basse — ambigu par construction, ne jamais utiliser comme repli automatique pour narrateur ou auteur littéraire. |
| **`literary_author`** | Qui a **écrit** l'histoire à l'origine ? | `stories.literary_author` | Moyenne — enrichi par recherche web, `NULL` si pas encore enrichi, `""` si enrichi sans résultat. |
| **`narrator`** | Qui **lit** l'histoire à voix haute (une seule personne retenue) ? | `stories.narrator` | Haute si vient de `story_cast_verified`, moyenne si vient du clustering acoustique (confiance `haute` uniquement). |
| **`voices`** | Toutes les voix distinctes entendues dans l'enregistrement (narrateur + personnages) | `story_cast_verified` (prioritaire) ou `narrator_identities`/`narrator_identity_members` (repli) | Variable — voir tableau de priorité ci-dessous. |
| **`character_name`** | Qui parle à *cet instant précis* du transcript ? | `speaker_map` (résolution de `speaker_label`) | Haute — dérivé du texte du transcript lui-même par LLM. |

## Pourquoi `author` n'est fiable pour rien

`stories.author` vient uniquement du nom du dossier source. Cas piégeux
observés en pratique :
- Un dossier nommé d'après le narrateur (`"Richard Bohringer"`) → `author` = narrateur, correct par accident.
- Un dossier nommé d'après une collection (`"Contes"`, `"ma boite à histoire"`) → `author` ne désigne ni un narrateur ni un auteur.
- Un dossier nommé d'après l'auteur littéraire (`"Carrol, Lewis"`) → `author` désigne en fait `literary_author`, pas le narrateur.

C'est pourquoi `literary_author` et `narrator` sont des colonnes
**distinctes**, jamais dérivées automatiquement l'une de l'autre à partir de
`author` sans validation (web ou clustering acoustique).

## Priorité de fiabilité pour `voices` (le casting complet)

1. **`story_cast_verified`** (source `human` > `id3_metadata` > `artwork_vision`) — **immuable**, jamais réécrit par le clustering acoustique une fois présent.
2. À défaut, **`narrator_identities` / `narrator_identity_members`** (clustering acoustique ECAPA-TDNN + inférence LLM, confiance `haute`/`moyenne` uniquement) — reconstruit intégralement à chaque exécution, donc potentiellement différent d'une run à l'autre.

`contes_tools.py::story_details` applique exactement cette priorité (voir
`db.get_verified_cast` / `db.get_voices_for_story`).

## Comment chaque champ est peuplé (pipeline)

```
scan.py            → stories.author (brut, dossier)
enrich_web.py       → stories.literary_author, stories.literary_info (recherche web)
identify_speakers.py → speaker_map (LLM sur transcript, par piste)
extract_cast_from_media.py → story_cast_verified (source='id3_metadata' ou 'artwork_vision')
[humain, hors pipeline auto] → story_cast_verified (source='human')
speaker_voice_eval.py → eval_voice_embeddings (empreintes ECAPA-TDNN, expérimental)
narrator_identity.py → narrator_identities, narrator_identity_members,
                        et stories.narrator si confidence='haute'
                        (skip les histoires déjà dans story_cast_verified)
```

## Ce que le wiki devrait afficher pour chaque histoire

Une fiche "Voix & personnages" cohérente devrait distinguer explicitement :
- **Narrateur** (`narrator`, une seule personne, "qui lit")
- **Auteur littéraire** (`literary_author` avec repli sur `author` si absent, "qui a écrit")
- **Casting complet** (`voices`, toutes les voix de l'enregistrement)
- **Personnages par passage** (`speaker_map` + `transcript_segments`, "qui parle à ce moment précis")

Mélanger ces catégories dans une seule liste "personnages" reproduirait
l'erreur observée en pratique côté agent LLM (confusion entre personnages de
fiction et interprètes réels).
