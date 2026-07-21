# Anomalies connues (constatées sur la base réelle, 2026-07-19)

Constatées en interrogeant l'instance réelle du conteneur `contes-agent`
(SQLite + Chroma), pas seulement le code. Documentées ici pour référence —
**non corrigées**, correction à traiter séparément si souhaité.

## 1. `stories.literary_info` pollué par des sources web inappropriées

6 histoires sur 293 (`ready` avec `literary_info` rempli) ont un champ
`literary_info` dont la liste `sources` contient des résultats de recherche
web à caractère pornographique (xHamster, Pornhub, sites allemands
équivalents), remontés tels quels par `reference/enrich_web.py`.

**Histoires concernées** : id 1 (*Ma maison est en carton*), 291
(*Colchiques dans les prés*), 349 (*Le chat et ses compagnons*), 498 (*Ma
bohème*), 513 (*Sarah (Le départ)*), 586 (*Scoubidou, la poupée qui sait
tout*).

**Cause probable** : la requête de recherche web construite à partir du
titre/auteur est trop générique pour certains titres courts ou noms
communs (ex: "Agnès"), et l'API de recherche renvoie des résultats hors
sujet sans filtrage de pertinence côté `enrich_web.py`/`web_search_client.py`.

**Impact** : ce champ n'est pas exposé directement à l'utilisateur final via
`contes_tools.py` actuellement (seul `literary_author` l'est), donc pas de
fuite immédiate côté agent vocal — mais la donnée est en base et pourrait
être exposée par un futur usage (ex: wiki, debug, nouvel outil).

## 2. Documents Chroma orphelins pour les histoires `missing`

`chroma_store` contient 327 documents `story_summary`/`story_keywords`
alors que SQLite ne compte que 295 histoires `status = 'ready'` — écart de
32, qui correspond exactement au nombre d'histoires `status = 'missing'`
(fichier disparu du disque lors d'un re-scan).

**Cause** : aucun mécanisme ne supprime les embeddings Chroma d'une histoire
quand son statut SQLite passe à `missing` — `scan.py::mark_missing` met à
jour `stories.status` mais n'appelle jamais de fonction de suppression côté
`chroma_store`.

**Impact réel** : `chroma_store.search_stories()` interroge Chroma sans
filtrer sur le statut SQLite, et `contes_tools.py::_search_stories` ne
revérifie pas non plus le statut des résultats sémantiques avant de les
renvoyer. Une histoire dont le fichier a disparu du disque **peut donc
apparaître dans les résultats de recherche** et, si l'utilisateur demande à
l'écouter, échouer au moment du streaming (fichier introuvable) plutôt qu'à
la recherche.

## 3. Classes thématiques orphelines côté Chroma

`chroma_store` contient 23 documents `kind="theme_class"` (id 1 à 23) alors
que la table SQLite `theme_classes` n'en contient que 10 (id 14 à 23). Les
classes 1 à 13 (ex: `"Ruse et Astuce"`, `"Peur et Créatures Fantastiques"`)
proviennent d'une exécution antérieure de `consolidate_theme_classes` —
remplacées côté SQLite (`db.replace_theme_classes` fait un `DELETE` complet
avant réinsertion) mais jamais nettoyées côté Chroma
(`chroma_store.upsert_theme_class` ne fait qu'ajouter/remplacer, jamais
supprimer).

**Impact réel** : `search_theme_classes()` peut retourner une classe morte
(id 1-13). `contes_tools.py` appelle ensuite
`db.stories_by_theme_class(class_id)` avec cet id, qui ne matche plus rien
en SQLite → résultat vide silencieux, alors qu'une classe thématique
actuelle réellement proche (ex: id 14 `"Fables et ruses animales"`, proche
sémantiquement de l'ancienne id 1 `"Ruse et Astuce"`) aurait pu être un
meilleur candidat mais n'a pas été retenue à cause du score plus faible de
la classe morte.

---

Les trois anomalies partagent une même cause racine : **Chroma et
`theme_classes`/`stories` peuvent diverger sans qu'aucune synchronisation ou
purge automatique ne les recolle**, sur des opérations qui ne repassent pas
par les fonctions d'`upsert` (changement de statut, remplacement complet de
`theme_classes`). Un correctif générique pourrait ajouter des étapes
explicites de suppression Chroma à ces deux points, plutôt que de traiter
chaque cas isolément.
