import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("DB_PATH", "/data/contes.db"))

# Mots vides ignorés lors de la construction d'une requête FTS5 à partir d'une
# phrase en langage naturel — le fallback lexical ne doit matcher que les mots
# porteurs de sens (ex: "un arbre à pain" -> "arbre" OR "pain"). Inclut aussi des mots
# génériques du DOMAINE (conte, histoire...) : présents dans quasiment tous les résumés
# du catalogue, ils ne discriminent rien et diluaient les vraies requêtes — ex: "contes
# qui font peur" matchait ~20 histoires sans rapport via le seul mot "contes", avec le
# même score figé que les histoires réellement pertinentes (voir KEYWORD_MATCH_SCORE
# côté contes_tools.py), noyant les résultats sémantiques/thématiques plus pertinents.
_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "a", "au", "aux",
    "avec", "pour", "dans", "sur", "en", "est", "que", "qui", "ce", "cette", "ces",
    "son", "sa", "ses", "tu", "as", "il", "elle", "ils", "elles", "on", "nous",
    "vous", "je", "j", "d", "l", "se", "ne", "pas", "plus", "ou", "mais", "donc",
    "or", "ni", "car", "y", "est-ce", "qu",
    "conte", "contes", "histoire", "histoires", "raconte", "raconter", "récit", "récits",
    # Conjugaisons courantes de être/avoir/faire : verbes quasi universels du récit,
    # présents dans la plupart des résumés indépendamment du sujet réel de l'histoire.
    "sont", "était", "étaient", "sera", "seront", "font", "fait", "faire", "faisait",
    "avait", "avaient", "ont", "aura", "auront",
}


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS stories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                folder_path TEXT UNIQUE NOT NULL,
                merged_folders TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'discovered',
                short_summary TEXT,
                long_summary TEXT,
                total_duration_seconds REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                order_index INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                duration_seconds REAL,
                cumulative_start_seconds REAL,
                status TEXT NOT NULL DEFAULT 'discovered'
            );
            CREATE INDEX IF NOT EXISTS idx_tracks_story_order ON tracks(story_id, order_index);
            -- Un fichier physique n'appartient jamais qu'à une seule histoire à la fois ;
            -- unique par file_path (pas par story_id) pour survivre à une réattribution
            -- par split_stories lors d'un futur re-scan du même dossier.
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tracks_file_path ON tracks(file_path);

            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                speaker_label TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_segments_track_start ON transcript_segments(track_id, start_seconds);

            -- La diarization tourne piste par piste (voir reference/transcribe.py) : un même
            -- label SPEAKER_XX dans deux pistes différentes de la même histoire ne désigne
            -- PAS forcément le même personnage — seule la stabilité au sein d'UNE piste est
            -- garantie. La clé inclut donc track_id, jamais seulement story_id.
            CREATE TABLE IF NOT EXISTS speaker_map (
                story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                speaker_label TEXT NOT NULL,
                character_name TEXT NOT NULL,
                PRIMARY KEY (story_id, track_id, speaker_label)
            );

            CREATE TABLE IF NOT EXISTS periods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                period_index INTEGER NOT NULL,
                global_start_seconds REAL NOT NULL,
                global_end_seconds REAL NOT NULL,
                raw_transcript_text TEXT,
                summary_text TEXT,
                UNIQUE(story_id, period_index)
            );

            CREATE TABLE IF NOT EXISTS bookmarks (
                story_id INTEGER PRIMARY KEY REFERENCES stories(id) ON DELETE CASCADE,
                position_seconds REAL NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Fallback lexical (mots-clés exacts) en complément des embeddings : une
            -- requête très factuelle/rare ("un arbre à pain") peut être diluée dans un
            -- vecteur sémantique alors qu'une correspondance mot-à-mot la retrouve à coup sûr.
            CREATE VIRTUAL TABLE IF NOT EXISTS stories_fts USING fts5(
                title, short_summary, long_summary, keywords_text,
                story_id UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS periods_fts USING fts5(
                summary_text, raw_transcript_text,
                story_id UNINDEXED, period_index UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            );

            -- Classes thématiques "libres" : découvertes depuis le contenu réel du
            -- catalogue (pas une liste fixée à l'avance), consolidées à partir d'une
            -- proposition de thème brut par histoire pour éviter d'avoir une classe
            -- quasi unique par histoire. Une recherche par similarité sur label+
            -- description (embedding séparé dans Chroma) route une requête libre
            -- ("des histoires de pirates") vers la ou les classes les plus proches.
            CREATE TABLE IF NOT EXISTS theme_classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Une histoire peut relever de plusieurs thèmes à la fois (ex: Frankenstein
            -- est à la fois "Magie et Enchantements" ET "Peur et Créatures Fantastiques")
            -- — many-to-many plutôt qu'une seule classe par histoire (stories.theme_class_id,
            -- devenue legacy/non utilisée, laissée en place sans y toucher).
            CREATE TABLE IF NOT EXISTS story_theme_classes (
                story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                theme_class_id INTEGER NOT NULL REFERENCES theme_classes(id) ON DELETE CASCADE,
                PRIMARY KEY (story_id, theme_class_id)
            );

            -- Cache expérimental pour reference.speaker_voice_eval : registre de voix par
            -- empreinte ECAPA-TDNN, EN PLUS du mapping locuteur actuel (LLM sur le texte du
            -- transcript, qui reste la référence en production). Granularité = l'unité
            -- atomique d'une vraie voix physique : (track_id, speaker_label), PAS toute une
            -- histoire — la diarization tourne piste par piste, donc un même personnage
            -- ('Narrateur' notamment) peut recouvrir plusieurs voix distinctes selon la
            -- piste. Le clustering (voir cluster_voices) se fait UNIQUEMENT sur ces
            -- embeddings, sans utiliser stories.author ni character_name — ces infos ne
            -- servent qu'à ANNOTER un cluster après coup, jamais à le construire.
            CREATE TABLE IF NOT EXISTS eval_voice_embeddings (
                story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                speaker_label TEXT NOT NULL,
                character_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                seconds REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (story_id, track_id, speaker_label)
            );

            -- Résultat expérimental de reference.narrator_identity : une identité déduite
            -- par LLM à partir des chemins de fichiers d'un cluster acoustique (voir
            -- eval_voice_embeddings / cluster_voices). Table intégralement remplacée à
            -- chaque exécution (DELETE puis réinsertion) — le clustering n'étant pas
            -- stable d'une run à l'autre (seuil, catalogue qui grandit), la conserver
            -- entre deux runs n'aurait pas de sens. Quand confidence='haute', la même
            -- exécution met aussi à jour stories.narrator (voir narrator_identity.run) :
            -- cette table garde la trace complète (raisonnement, membres, seuil) même
            -- pour les identités à confiance faible qu'on ne pousse jamais en production.
            CREATE TABLE IF NOT EXISTS narrator_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inferred_name TEXT NOT NULL,
                confidence TEXT NOT NULL,
                is_professional INTEGER NOT NULL,
                reasoning TEXT NOT NULL,
                cluster_threshold REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS narrator_identity_members (
                identity_id INTEGER NOT NULL REFERENCES narrator_identities(id) ON DELETE CASCADE,
                story_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                speaker_label TEXT NOT NULL,
                PRIMARY KEY (story_id, track_id, speaker_label)
            );
        """)

        async with db.execute("PRAGMA table_info(speaker_map)") as cur:
            speaker_map_cols = {row[1] for row in await cur.fetchall()}
        if "track_id" not in speaker_map_cols:
            # Migration depuis l'ancien schéma (story_id, speaker_label) : ces données
            # supposaient à tort qu'un même label désigne le même personnage à travers
            # toutes les pistes d'une histoire, ce qui est faux en général (diarization
            # par piste, labels non stables d'une piste à l'autre). Sans ambiguïté pour
            # une histoire à une seule piste (le seul track_id possible) : on backfill ces
            # lignes-là. Pour une histoire à plusieurs pistes, le mapping existant peut
            # mélanger des personnages différents sous un même label — aucun backfill
            # fiable possible, on remet uniquement CES histoires à 'grouped' pour que
            # identify_speakers_pending les retraite avec le schéma correct.
            await db.execute("ALTER TABLE speaker_map RENAME TO speaker_map_old")
            await db.execute("""
                CREATE TABLE speaker_map (
                    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                    speaker_label TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    PRIMARY KEY (story_id, track_id, speaker_label)
                )
            """)
            await db.execute("""
                INSERT INTO speaker_map (story_id, track_id, speaker_label, character_name)
                SELECT sm.story_id, t.id, sm.speaker_label, sm.character_name
                FROM speaker_map_old sm
                JOIN tracks t ON t.story_id = sm.story_id
                WHERE sm.story_id IN (
                    SELECT story_id FROM tracks GROUP BY story_id HAVING COUNT(*) = 1
                )
            """)
            await db.execute("""
                UPDATE stories SET status = 'grouped' WHERE status IN
                    ('speakers_identified', 'summarized', 'ready')
                AND id IN (SELECT story_id FROM tracks GROUP BY story_id HAVING COUNT(*) > 1)
            """)
            await db.execute("DROP TABLE speaker_map_old")
            logger.warning(
                "speaker_map migré vers le schéma par piste — histoires multi-pistes "
                "remises à 'grouped' pour retraitement par identify_speakers_pending, "
                "histoires mono-piste préservées telles quelles"
            )

        async with db.execute("PRAGMA table_info(stories)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        if "keywords" not in cols:
            await db.execute("ALTER TABLE stories ADD COLUMN keywords TEXT")
        if "literary_author" not in cols:
            # NULL = pas encore enrichi via recherche web ; "" = enrichi mais rien
            # trouvé ; sinon le nom de l'auteur original (ex: "Charles Perrault").
            await db.execute("ALTER TABLE stories ADD COLUMN literary_author TEXT")
        if "literary_info" not in cols:
            await db.execute("ALTER TABLE stories ADD COLUMN literary_info TEXT")
        if "narrator" not in cols:
            # Nom confirmé/corrigé du narrateur (ex: "Romane Bohringer"), distinct de
            # stories.author qui est la valeur brute tirée du nom de dossier et peut
            # être mal orthographiée (ex: "Roman Boringher").
            await db.execute("ALTER TABLE stories ADD COLUMN narrator TEXT")
        if "age_range" not in cols:
            # Une valeur parmi reference.classify.AGE_RANGES (vocabulaire fixe) —
            # filtrable exactement, contrairement à une recherche sémantique floue.
            await db.execute("ALTER TABLE stories ADD COLUMN age_range TEXT")
        if "mood_tags" not in cols:
            # JSON liste, sous-ensemble de reference.classify.MOOD_TAGS (vocabulaire fixe).
            await db.execute("ALTER TABLE stories ADD COLUMN mood_tags TEXT")
        if "raw_theme_label" not in cols:
            # Étape intermédiaire de la classification "libre" : thème propre à CETTE
            # histoire, en texte libre — sert ensuite à consolider un ensemble réduit
            # de classes communes (theme_classes) plutôt que d'avoir un thème quasi
            # unique par histoire.
            await db.execute("ALTER TABLE stories ADD COLUMN raw_theme_label TEXT")
        if "theme_class_id" not in cols:
            await db.execute(
                "ALTER TABLE stories ADD COLUMN theme_class_id INTEGER REFERENCES theme_classes(id)"
            )

        await db.commit()
    logger.info("DB initialisée")


async def find_story_id_by_title(title_query: str) -> int | None:
    """Résout un titre approximatif vers un story_id réel — filet de sécurité pour le cas
    où l'appelant (LLM) devine un story_id en texte libre au lieu d'appeler search_contes
    d'abord (observé en pratique : 'story_id': 'le petit prince')."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT id FROM stories WHERE title LIKE ? AND status = 'ready' "
            "ORDER BY LENGTH(title) LIMIT 1",
            (f"%{title_query}%",),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return row["id"]

        # Repli tolérant aux variantes orthographiques (ex: 'Renard' deviné par le LLM
        # pour le titre réel 'Le Roman De Renart') — le LIKE ci-dessus est un substring
        # exact, insuffisant pour ce cas. rapidfuzz sur les titres normalisés (accents/
        # casse) trouve la meilleure correspondance ; seuil calibré pour tolérer une
        # faute/variante ponctuelle sans matcher un titre sans rapport.
        async with conn.execute("SELECT id, title FROM stories WHERE status = 'ready'") as cur:
            all_stories = await cur.fetchall()
    if not all_stories:
        return None
    norm_query = normalize_text(title_query)
    choices = {s["id"]: normalize_text(s["title"]) for s in all_stories}
    match = process.extractOne(norm_query, choices, scorer=fuzz.ratio, score_cutoff=70)
    return match[2] if match else None


async def get_story(story_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM stories WHERE id = ?", (story_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def get_narrators(story_ids: list[int]) -> dict[int, str]:
    """Lookup groupé de stories.narrator (nom confirmé/corrigé, voir init_db) — utilisé
    par contes_tools pour enrichir a posteriori des résultats venus de sources qui ne
    portent pas ce champ (Chroma notamment, dont les métadonnées sont figées au moment de
    l'embedding et n'ont jamais narrator). Ne renvoie que les histoires où narrator est
    effectivement renseigné : à l'appelant de retomber sur stories.author sinon."""
    if not story_ids:
        return {}
    async with aiosqlite.connect(DB_PATH) as conn:
        placeholders = ",".join("?" * len(story_ids))
        async with conn.execute(
            f"SELECT id, narrator FROM stories WHERE id IN ({placeholders}) "
            f"AND narrator IS NOT NULL AND narrator != ''",
            story_ids,
        ) as cur:
            rows = await cur.fetchall()
    return {r[0]: r[1] for r in rows}


async def get_tracks_for_story(story_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM tracks WHERE story_id = ? ORDER BY order_index", (story_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_track(track_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def get_periods_for_story(story_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT period_index, global_start_seconds, global_end_seconds, summary_text "
            "FROM periods WHERE story_id = ? ORDER BY period_index",
            (story_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_raw_periods_for_story(story_id: int) -> list[dict]:
    """Texte brut diarizé de chaque période, dans l'ordre — la source la plus fidèle pour
    la classification (mood/thème), contrairement aux résumés qui édulcorent volontiers
    les détails marquants (violence, peur) au profit d'un ton narratif plus neutre."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT period_index, raw_transcript_text FROM periods "
            "WHERE story_id = ? ORDER BY period_index",
            (story_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_bookmark(story_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM bookmarks WHERE story_id = ?", (story_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def save_bookmark(story_id: int, position_seconds: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO bookmarks (story_id, position_seconds, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(story_id) DO UPDATE SET "
            "position_seconds = excluded.position_seconds, updated_at = excluded.updated_at",
            (story_id, position_seconds, now),
        )
        await conn.commit()


async def delete_bookmark(story_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("DELETE FROM bookmarks WHERE story_id = ?", (story_id,))
        await conn.commit()


def _fts_match_expr(query: str) -> str | None:
    words = re.findall(r"\w+", query.lower())
    terms = [w for w in words if len(w) >= 2 and w not in _STOPWORDS_FR]
    if not terms:
        return None
    return " OR ".join(f'"{w.replace(chr(34), chr(34) * 2)}"' for w in terms)


async def sync_story_fts(conn: aiosqlite.Connection, story_id: int, title: str,
                          short_summary: str, long_summary: str, keywords: list[str]) -> None:
    await conn.execute("DELETE FROM stories_fts WHERE story_id = ?", (story_id,))
    await conn.execute(
        "INSERT INTO stories_fts (story_id, title, short_summary, long_summary, keywords_text) "
        "VALUES (?, ?, ?, ?, ?)",
        (story_id, title, short_summary or "", long_summary or "", ", ".join(keywords)),
    )


async def sync_period_fts(conn: aiosqlite.Connection, story_id: int, period_index: int,
                           summary_text: str, raw_transcript_text: str) -> None:
    await conn.execute(
        "DELETE FROM periods_fts WHERE story_id = ? AND period_index = ?", (story_id, period_index)
    )
    await conn.execute(
        "INSERT INTO periods_fts (story_id, period_index, summary_text, raw_transcript_text) "
        "VALUES (?, ?, ?, ?)",
        (story_id, period_index, summary_text or "", raw_transcript_text or ""),
    )


async def search_stories_fts(query: str, limit: int = 5) -> list[dict]:
    match_expr = _fts_match_expr(query)
    if not match_expr:
        return []
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute(
                """
                SELECT s.id AS story_id, s.title, s.author, s.short_summary, s.long_summary,
                       s.keywords, s.total_duration_seconds, bm25(stories_fts) AS rank
                FROM stories_fts
                JOIN stories s ON s.id = stories_fts.story_id
                WHERE stories_fts MATCH ? AND s.status = 'ready'
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, limit),
            ) as cur:
                rows = await cur.fetchall()
        except aiosqlite.OperationalError as e:
            logger.error(f"search_stories_fts: requête FTS invalide ({match_expr!r}): {e}")
            return []
    return [dict(r) for r in rows]


async def search_moments_fts(query: str, story_id: int | None = None, limit: int = 5) -> list[dict]:
    match_expr = _fts_match_expr(query)
    if not match_expr:
        return []
    sql = """
        SELECT p.story_id, p.period_index, p.global_start_seconds, p.global_end_seconds,
               p.summary_text, p.raw_transcript_text, bm25(periods_fts) AS rank
        FROM periods_fts
        JOIN periods p ON p.story_id = periods_fts.story_id AND p.period_index = periods_fts.period_index
        WHERE periods_fts MATCH ?
    """
    params: list = [match_expr]
    if story_id is not None:
        sql += " AND p.story_id = ?"
        params.append(story_id)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute(sql, params) as cur:
                rows = await cur.fetchall()
        except aiosqlite.OperationalError as e:
            logger.error(f"search_moments_fts: requête FTS invalide ({match_expr!r}): {e}")
            return []
    return [dict(r) for r in rows]


def normalize_text(s: str) -> str:
    """Lowercase, accent-stripped form used for accent-insensitive comparisons
    (author names notably vary in accent usage/typos)."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


# Mots qui n'aident pas à distinguer un narrateur/auteur (soit des mots de liaison
# de noms de dossiers comme "Lu par X", soit des mots génériques que la requête
# contient presque toujours, comme "conte(s)" — sans quoi un dossier nommé juste
# "Contes" matcherait quasiment toute recherche).
_AUTHOR_NOISE_WORDS = {"lu", "par", "conte", "contes", "interprete", "raconte", "racontee", "racontes"}


def _significant_tokens(s: str, extra_stopwords: set = frozenset()) -> set:
    words = re.findall(r"\w+", normalize_text(s.replace("_", " ")))
    return {w for w in words if len(w) >= 2 and w not in _STOPWORDS_FR and w not in extra_stopwords}


def _duration_age_mood_clauses(min_duration_seconds, max_duration_seconds,
                                age_range: str | None, mood: str | None) -> tuple[list[str], list]:
    clauses = ["status = 'ready'"]
    params: list = []
    if min_duration_seconds is not None:
        clauses.append("total_duration_seconds >= ?")
        params.append(min_duration_seconds)
    if max_duration_seconds is not None:
        clauses.append("total_duration_seconds <= ?")
        params.append(max_duration_seconds)
    if age_range:
        clauses.append("age_range = ?")
        params.append(age_range)
    if mood:
        # mood_tags est un JSON array stocké en texte ; un simple LIKE sur la
        # représentation JSON suffit pour un vocabulaire fixe de mots courts.
        clauses.append("mood_tags LIKE ?")
        params.append(f'%"{mood}"%')
    return clauses, params


async def count_ready_stories(min_duration_seconds: float | None = None,
                               max_duration_seconds: float | None = None,
                               age_range: str | None = None, mood: str | None = None) -> int:
    clauses, params = _duration_age_mood_clauses(min_duration_seconds, max_duration_seconds, age_range, mood)
    async with aiosqlite.connect(DB_PATH) as conn:
        async with conn.execute(
            f"SELECT COUNT(*) FROM stories WHERE {' AND '.join(clauses)}", params
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


async def sample_stories(limit: int | None = None, offset: int = 0,
                          min_duration_seconds: float | None = None,
                          max_duration_seconds: float | None = None,
                          age_range: str | None = None, mood: str | None = None) -> list[dict]:
    # Pas de limite par défaut : une liste doit pouvoir couvrir tout le catalogue.
    # Ordre alphabétique + offset (pas RANDOM()) : un appel avec un offset donné doit
    # toujours retomber sur la même tranche, pour permettre un parcours par plage
    # explicite (ex: histoires 12 à 52) sans trou ni doublon.
    clauses, params = _duration_age_mood_clauses(min_duration_seconds, max_duration_seconds, age_range, mood)
    sql_limit = limit if limit is not None else -1  # SQLite : -1 = pas de limite
    params += [sql_limit, offset]
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            f"SELECT id, title, author, short_summary, total_duration_seconds FROM stories "
            f"WHERE {' AND '.join(clauses)} ORDER BY title LIMIT ? OFFSET ?",
            params,
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def stories_missing_traits(story_id: int | None = None, limit: int | None = None) -> list[dict]:
    """Histoires 'ready' sans age_range/mood_tags encore assignés (classification fixe)."""
    sql = ("SELECT id, title, author, short_summary, keywords FROM stories "
           "WHERE status = 'ready' AND age_range IS NULL")
    params: tuple = ()
    if story_id is not None:
        sql += " AND id = ?"
        params = (story_id,)
    sql += " ORDER BY id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def set_story_traits(story_id: int, age_range: str, mood_tags: list[str]) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "UPDATE stories SET age_range = ?, mood_tags = ? WHERE id = ?",
            (age_range, json.dumps(mood_tags, ensure_ascii=False), story_id),
        )
        await conn.commit()


async def stories_missing_raw_theme(story_id: int | None = None, limit: int | None = None) -> list[dict]:
    sql = ("SELECT id, title, author, short_summary, keywords FROM stories "
           "WHERE status = 'ready' AND raw_theme_label IS NULL")
    params: tuple = ()
    if story_id is not None:
        sql += " AND id = ?"
        params = (story_id,)
    sql += " ORDER BY id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def set_raw_theme_label(story_id: int, label: str) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("UPDATE stories SET raw_theme_label = ? WHERE id = ?", (label, story_id))
        await conn.commit()


async def all_raw_theme_labels() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT id, title, raw_theme_label FROM stories "
            "WHERE status = 'ready' AND raw_theme_label IS NOT NULL ORDER BY id"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def replace_theme_classes(classes: list[dict]) -> dict[str, int]:
    """Remplace entièrement la table theme_classes (la consolidation part de zéro à
    chaque exécution) et renvoie un mapping label -> id nouvellement créé. Les
    assignations story->classes (story_theme_classes) sont réinitialisées explicitement
    pour ne jamais pointer vers une classe supprimée."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("DELETE FROM story_theme_classes")
        await conn.execute("DELETE FROM theme_classes")
        label_to_id: dict[str, int] = {}
        for c in classes:
            cur = await conn.execute(
                "INSERT INTO theme_classes (label, description, created_at) VALUES (?, ?, ?)",
                (c["label"], c["description"], now),
            )
            label_to_id[c["label"]] = cur.lastrowid
        await conn.commit()
    return label_to_id


async def get_theme_classes() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT id, label, description FROM theme_classes ORDER BY label") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def stories_missing_theme_assignment(story_id: int | None = None, limit: int | None = None) -> list[dict]:
    sql = ("SELECT id, title, author, short_summary, keywords, raw_theme_label FROM stories s "
           "WHERE status = 'ready' AND raw_theme_label IS NOT NULL "
           "AND NOT EXISTS (SELECT 1 FROM story_theme_classes stc WHERE stc.story_id = s.id)")
    params: tuple = ()
    if story_id is not None:
        sql += " AND id = ?"
        params = (story_id,)
    sql += " ORDER BY id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def set_story_theme_classes(story_id: int, theme_class_ids: list[int]) -> None:
    """Remplace l'ensemble des classes de cette histoire (1 à N) par la liste donnée."""
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("DELETE FROM story_theme_classes WHERE story_id = ?", (story_id,))
        await conn.executemany(
            "INSERT OR IGNORE INTO story_theme_classes (story_id, theme_class_id) VALUES (?, ?)",
            [(story_id, cid) for cid in theme_class_ids],
        )
        await conn.commit()


async def stories_by_theme_class(class_id: int, limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT s.id, s.title, s.author, s.short_summary, s.total_duration_seconds FROM stories s "
            "JOIN story_theme_classes stc ON stc.story_id = s.id "
            "WHERE s.status = 'ready' AND stc.theme_class_id = ? ORDER BY s.title LIMIT ?",
            (class_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def list_stories_by_author(author_query: str, limit: int = 10) -> list[dict]:
    """Direct author filter (folder-derived stories.author), not routed through
    embeddings/FTS — a narrator's name is rarely mentioned in a story's own
    transcript/summary, so semantic/lexical content search can't find it.
    Matches by significant-word overlap (either direction) rather than raw
    substring, so "les contes de Richard Bohringer" matches author folder
    "Lu par Richard Bohringer" despite neither string containing the other."""
    query_tokens = _significant_tokens(author_query, _AUTHOR_NOISE_WORDS)
    if not query_tokens:
        return []
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT id, title, author, narrator, short_summary, total_duration_seconds "
            "FROM stories WHERE status = 'ready'"
        ) as cur:
            rows = await cur.fetchall()
    matches = []
    for r in rows:
        # Le nom brut du dossier (author) peut être mal orthographié (ex: "Roman
        # Boringher") — narrator, quand connu, est la forme corrigée/confirmée ; on
        # matche sur les deux, l'un ou l'autre suffit.
        candidates = [r["author"]] + ([r["narrator"]] if r["narrator"] else [])
        for candidate in candidates:
            tokens = _significant_tokens(candidate, _AUTHOR_NOISE_WORDS)
            if tokens and (tokens <= query_tokens or query_tokens <= tokens):
                matches.append(dict(r))
                break
    return matches[:limit]
