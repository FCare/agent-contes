# Proposition d'organisation du wiki

Deux options, à choisir/combiner. Aucune n'est codée — document de décision
uniquement.

## Contrainte connue
Le wiki sera monté sur un **volume Docker externe** : privilégier un
mécanisme qui écrit sur disque (fichiers statiques ou base embarquée type
SQLite/fichiers), pas une dépendance à un service cloud tiers.

## Ce qui manque pour l'organisation "recueil" telle que proposée

`recueil` (anthologie/collection) n'est **pas une entité de la base** — ni
dans SQLite ni dans Chroma. Le seul signal disponible est le premier segment
de `stories.folder_path`, qui fonctionne bien pour certains cas (`"Agnès
Chaumié"` → 15 histoires, `"Philippe Lejour"` → 18) mais échoue pour 36 %
du catalogue, regroupé sous des dossiers fourre-tout (`"Contes"`,
`"Interprète inconnu"`). Une hiérarchie "recueil → histoire" fiable à 100 %
n'est donc pas directement extractible ; elle demande soit une heuristique
tolérante à l'échec (regrouper seulement quand le dossier parent est
suffisamment spécifique), soit une passe de curation manuelle/LLM
supplémentaire (hors scope actuel du pipeline).

---

## Option A — Wiki hiérarchique adapté à l'idée initiale

Reprend la structure proposée, ajustée aux données réellement disponibles.

```
Accueil
├── Par recueil (best-effort : dossier parent quand non-générique,
│   sinon "Histoires sans recueil identifié" — n'est pas caché, explicite)
│   └── <recueil>
│       └── <histoire> (lien vers la fiche complète)
├── Par histoire (liste alphabétique complète — le niveau le plus fiable)
│   └── <histoire>
│       ├── Résumé (short_summary, long_summary)
│       ├── Informations (durée, âge, ambiances/mood_tags, thèmes)
│       ├── Voix & personnages (voir concepts-personnes.md — narrateur,
│       │   auteur littéraire, casting complet, distincts et non mélangés)
│       ├── Chapitres = periods (~3 min chacune, résumé + lien texte brut)
│       └── Texte intégral (concaténation ordonnée des transcript_segments,
│           par piste puis par ordre temporel — pas stocké tel quel
│           aujourd'hui, à générer à l'assemblage du wiki)
├── Par personnage/voix (page par narrateur/interprète, ex: "Richard
│   Bohringer" → liste des histoires où cette voix apparaît)
├── Par thème (theme_classes → histoires associées)
└── Recherche sémantique (barre de recherche libre)
```

**Recherche sémantique — ne pas recréer un second index.** Les embeddings
existent déjà dans Chroma (`story_summary`, `story_keywords`,
`period_summary`, `period_transcript`). Le générateur du wiki peut soit :
1. **Précalculer** au moment de la génération des liens "histoires
   similaires" / "passages similaires" par page (recherche par similarité
   sur le contenu de la page elle-même) — reste dans un wiki 100% statique.
2. **Ou** exposer une petite API de recherche en lecture seule
   (FastAPI/Flask, quelques dizaines de lignes) qui interroge directement
   `chroma_store.search_stories`/`search_moments`/`search_theme_classes`
   pour une vraie recherche libre côté utilisateur — nécessite un service
   vivant à côté des fichiers statiques (même volume, conteneur séparé ou
   même conteneur que l'agent).

**Génération** : un script `reference/build_wiki.py` (nouvelle étape,
optionnelle, jamais un effet de bord de `pipeline.py --stage all` — comme
`enrich_web`/`classify_*`) qui lit SQLite + Chroma et écrit des pages
Markdown/HTML statiques sur le volume monté. Générateur candidat :
MkDocs (Material) ou Hugo — les deux savent lire un dossier de Markdown et
produire un site statique consultable hors-ligne sur le volume, avec
recherche lexicale intégrée gratuite (à ne pas confondre avec la recherche
sémantique de l'option 2 ci-dessus).

**Avantages** : correspond à l'idée initiale, hiérarchie familière.
**Inconvénients** : le niveau "recueil" restera visiblement imparfait
(bucket "sans recueil" à 36 %) ; nécessite une régénération après chaque
run de pipeline pour rester à jour.

---

## Option B — Alternative : wiki organisé par facettes plutôt que par hiérarchie fixe

Plutôt que de forcer une hiérarchie recueil → histoire → chapitre (dont le
premier niveau est structurellement bruité), organiser autour de ce que les
données supportent **réellement bien** : chaque histoire est une fiche
unique, et la navigation se fait par **facettes croisées** plutôt que par un
arbre à sens unique — plus proche d'un catalogue consultable que d'un wiki
classique.

```
Accueil = recherche sémantique en avant (réutilise directement
          chroma_store.search_stories/search_moments — pas de hiérarchie
          à parcourir pour trouver une histoire)
Facettes combinables (filtres, pas des dossiers imposés) :
  - Thème (theme_classes, 10 valeurs actuelles — fiable, N-N)
  - Tranche d'âge (age_range, vocabulaire fixe 4 valeurs)
  - Ambiance (mood_tags, vocabulaire fixe 12 valeurs)
  - Narrateur/voix (regroupement par narrator/story_cast_verified —
    plus fiable que "recueil" car directement une colonne dédiée)
  - Recueil (facette optionnelle, affichée seulement quand le
    regroupement dossier est spécifique — jamais un niveau obligatoire)
Fiche histoire = identique à l'option A (résumé, infos, voix,
  chapitres/periods, texte intégral)
```

**Différence clé avec l'option A** : "recueil" devient une facette parmi
d'autres au lieu du sommet obligatoire de la hiérarchie — évite d'exposer
une catégorisation qu'on sait imparfaite comme si elle était la colonne
vertébrale du site. Thème/âge/ambiance/narrateur sont chacun des colonnes
dédiées et fiables (vocabulaire fixe ou source directe), donc de meilleurs
points d'entrée.

**Avantages** : n'expose jamais de hiérarchie bancale comme structurante ;
exploite directement les champs les plus fiables de la base ; la recherche
sémantique (déjà existante) devient le point d'entrée principal plutôt
qu'un ajout en bout de liste.
**Inconvénients** : moins "feuilletable" qu'une hiérarchie classique par
recueil pour quelqu'un qui connaît déjà la collection physique (CD/dossier)
et veut naviguer comme sur l'étagère d'origine.

---

## Recommandation

Combiner les deux : structure de fiches identique dans les deux options
(c'est le cœur du contenu), mais **navigation par facettes (option B) en
page d'accueil**, avec la vue "par recueil" (option A) gardée comme une
facette secondaire plutôt que supprimée — elle reste utile pour qui
connaît déjà la collection physique, sans être présentée comme fiable à
100 %. Génération statique (MkDocs/Hugo) + une petite API de recherche
sémantique en lecture seule réutilisant Chroma tel quel, les deux vivant
sur le même volume Docker externe.
