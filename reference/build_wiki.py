"""Génère un wiki statique (MkDocs) depuis le catalogue SQLite : une fiche par
histoire (résumé, infos, voix & personnages, chapitres, texte intégral), des
pages de navigation par facette (thème/âge/ambiance/voix/personnages, page
d'accueil — voir docs/wiki-proposal.md option B), un index alphabétique combiné
histoires+recueils, et une navigation secondaire par recueil reconstruite
depuis folder_path (option A, best-effort — voir _recueil_key). Étape
EXTRA_STAGE explicite (reference/pipeline.py), aussi appelée automatiquement
en fin de synchro quotidienne (main.py).

Régénération complète à chaque exécution (DELETE + réécriture de
WIKI_SRC_DIR, comme db.replace_theme_classes) : la génération est une pure
mise en forme de ce qui est déjà en base, sans coût LLM, donc pas besoin
d'un état incrémental.
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path

import aiosqlite

import db
from . import scan as scan_stage

logger = logging.getLogger(__name__)

WIKI_SRC_DIR = Path(os.environ.get("WIKI_SRC_DIR", "/wiki/src"))
WIKI_SITE_DIR = Path(os.environ.get("WIKI_SITE_DIR", "/wiki/site"))
MKDOCS_CONFIG = Path(__file__).resolve().parent.parent / "mkdocs.yml"
WIKI_ASSETS_SRC = Path(__file__).resolve().parent.parent / "wiki_theme"

# Domaine public du wiki (voir main.py::stream_track, route déjà utilisée pour la
# lecture MQTT via contes_tools.get_playlist) — sert à fournir l'URL ABSOLUE de
# chaque piste sur la fiche histoire, pas seulement un chemin relatif, pour qu'elle
# reste correcte/copiable telle quelle indépendamment de la page depuis laquelle on
# la lit. Même convention de nom de domaine par défaut que llm.py::LLM_BASE_URL.
WIKI_PUBLIC_BASE_URL = os.environ.get("WIKI_PUBLIC_BASE_URL", "https://contes.caronboulme.fr").rstrip("/")

# ---------------------------------------------------------------------------
# Regroupement par recueil (best-effort — pas d'entité "recueil" en base,
# voir docs/wiki-proposal.md et docs/database-sqlite.md)
# ---------------------------------------------------------------------------

# Même regex que reference/scan.py::mark_missing : split_stories.py suffixe
# folder_path avec "#N" quand plusieurs histoires proviennent du même dossier
# physique. Sans ce strip, "La rue broca#1".."#12" ou "Le petit prince#1".."#3"
# ne se regroupent jamais alors qu'ils forment un seul recueil scindé.
_SPLIT_SUFFIX_RE = re.compile(r"#\d+$")

# Dossiers fourre-tout observés dans le catalogue réel (voir docs/wiki-proposal.md) :
# ne désignent aucun recueil réel, juste un bac générique ("Contes" = 56 histoires,
# "Interprète inconnu" = 49, "Various Artists" = 3). Calibré sur le catalogue actuel,
# pas un vocabulaire figé — à ajuster si de nouveaux bacs génériques apparaissent.
_GENERIC_BUCKETS = {"Contes", "Interprète inconnu", "Various Artists"}

_UNSORTED_KEY = "__unsorted__"
_UNSORTED_LABEL = "Sans recueil identifié"


def _recueil_key(folder_path: str) -> str:
    """Premier segment de folder_path, suffixe de split retiré, ou le
    sentinel _UNSORTED_KEY si ce segment est un dossier fourre-tout connu."""
    top = folder_path.split("/", 1)[0]
    top = _SPLIT_SUFFIX_RE.sub("", top).strip()
    if not top or top in _GENERIC_BUCKETS:
        return _UNSORTED_KEY
    return top


def _group_by_recueil(stories: list[dict]) -> dict[str, list[dict]]:
    """Groupe par _recueil_key, puis replie dans _UNSORTED_KEY tout groupe à
    une seule histoire : un recueil à un seul membre n'apporte rien de plus
    qu'une fiche histoire seule, et produirait des centaines de pages de
    recueil à un seul élément pour les nombreux dossiers spécifiques mais
    isolés du catalogue (voir la distribution réelle dans docs/wiki-proposal.md)."""
    groups: dict[str, list[dict]] = {}
    for s in stories:
        groups.setdefault(_recueil_key(s["folder_path"]), []).append(s)

    final: dict[str, list[dict]] = {_UNSORTED_KEY: groups.pop(_UNSORTED_KEY, [])}
    for key, members in groups.items():
        if len(members) < 2:
            final[_UNSORTED_KEY].extend(members)
        else:
            final[key] = members
    return final


def _recueil_member_order_key(folder_path: str) -> int:
    """Ordre narratif au sein d'un recueil scindé par split_stories.py : le
    suffixe "#N" de folder_path reflète l'ordre des groupes tel que détecté par
    le LLM à partir de la séquence originale des pistes (voir
    split_stories.py::_apply_split, "#i" pour le i-ème groupe, le premier gardant
    folder_path sans suffixe) — bien plus fiable que l'ordre alphabétique des
    titres pour un recueil comme "La rue broca" (#1 à #12). 0 si pas de suffixe
    (dossier jamais scindé, ex: simples albums d'un même narrateur) : dans ce
    cas toutes les histoires du recueil sont à égalité et retombent sur le tri
    alphabétique (départage secondaire, voir son usage dans run())."""
    m = _SPLIT_SUFFIX_RE.search(folder_path)
    return int(m.group()[1:]) if m else 0


# ---------------------------------------------------------------------------
# Slugs (identifiants de page déterministes)
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, suffix: int | str | None = None) -> str:
    base = _SLUG_RE.sub("-", db.normalize_text(text)).strip("-") or "x"
    return f"{base}-{suffix}" if suffix is not None else base


def _story_slug(title: str, story_id: int) -> str:
    # Suffixe story_id : plusieurs histoires du catalogue peuvent partager un
    # titre proche ou identique, contrairement aux thèmes/recueils (peu
    # nombreux, distincts par construction).
    return _slugify(title, story_id)


def _recueil_display_name(key: str, members: list[dict]) -> str:
    """Nom affiché du recueil : le sous-dossier partagé par TOUS ses membres
    (ex: "Philippe Chatel/Enfants-Emilie Jolie[#N]" -> "Enfants-Emilie Jolie"),
    bien plus descriptif que le seul dossier parent (souvent le narrateur, voir
    _recueil_key) — sans ce sous-dossier commun (absent chez au moins un membre,
    ou différent d'un membre à l'autre), repli sur `key` tel quel. Réutilise
    scan._strip_disc_suffix pour retirer un éventuel "[Disc N]"/"(CD2)" laissé
    par le regroupement multi-disque (voir scan.py::_group_stories, qui NE
    nettoie PAS folder_path lui-même, seulement le titre)."""
    if key == _UNSORTED_KEY:
        return _UNSORTED_LABEL
    seconds = set()
    for m in members:
        parts = m["folder_path"].split("/", 1)
        if len(parts) < 2:
            return key
        second = _SPLIT_SUFFIX_RE.sub("", parts[1]).strip()
        second = scan_stage._strip_disc_suffix(second).strip()
        seconds.add(second)
    if len(seconds) == 1:
        candidate = next(iter(seconds))
        if candidate:
            return candidate
    return key


# Résout un slug d'URL de streaming (voir _track_stream_url ci-dessous et son usage
# dans _render_story_page) vers (story_id, numéro de piste 1-indexé) — utilisé par
# main.py::stream_by_slug. Format généré : "{slug du titre}-{story_id}" ou
# "{slug du titre}-{story_id}-piste-{n}" (voir _story_slug). Extraction en DEUX
# regex séquentielles plutôt qu'une seule combinée : un groupe optionnel final
# après un ".+" glouton ne backtrack PAS comme on pourrait s'y attendre — le
# moteur s'arrête à la première correspondance trouvée en réduisant le ".+" au
# minimum nécessaire pour matcher juste "-{story_id}" et ne tente jamais
# d'inclure "-piste-{n}" dans le suffixe, y compris quand ce suffixe est
# réellement présent (bug constaté : 'alice-au-pays-des-merveilles-4-piste-2'
# résolvait à tort vers story_id=2 au lieu de 4, piste=1 au lieu de 2).
_PISTE_SUFFIX_RE = re.compile(r"-piste-(\d+)$")
_STORY_ID_SUFFIX_RE = re.compile(r"-(\d+)$")


def parse_stream_slug(slug: str) -> tuple[int, int] | None:
    piste = 1
    m_piste = _PISTE_SUFFIX_RE.search(slug)
    if m_piste:
        piste = int(m_piste.group(1))
        slug = slug[:m_piste.start()]
    m_id = _STORY_ID_SUFFIX_RE.search(slug)
    if not m_id:
        return None
    return int(m_id.group(1)), piste


def _track_stream_url(story_title: str, story_id: int, order_index: int, multi_track: bool) -> str:
    """URL de streaming absolue, expressive (titre de l'histoire, pas un id brut ni
    une extension de fichier — voir main.py::stream_by_slug qui la résout à
    l'inverse via parse_stream_slug) plutôt que /stream/{track_id}. La route legacy
    /stream/{track_id:int} reste inchangée, utilisée telle quelle côté MQTT
    (contes_tools.get_playlist)."""
    slug = _story_slug(story_title, story_id)
    if multi_track:
        slug = f"{slug}-piste-{order_index + 1}"
    return f"{WIKI_PUBLIC_BASE_URL}/stream/{slug}"


# Paliers de durée : partition (pas de chevauchement) plutôt que des seuils
# cumulatifs — sinon "moins de 15 minutes" recontiendrait aussi les histoires
# déjà listées sous "moins de 5 minutes", redondant pour naviguer. Ordre
# chronologique (pas alphabétique) : voir la boucle dédiée dans run().
_DURATION_BUCKETS = [
    (300, "moins-de-5-min", "Moins de 5 minutes"),
    (900, "5-a-15-min", "5 à 15 minutes"),
    (3600, "15-min-a-1h", "15 minutes à 1 heure"),
    (None, "plus-d-1h", "Plus d'une heure"),
]


def _duration_bucket_slug(seconds: float | None) -> str | None:
    if not seconds:
        return None
    for threshold, slug, _label in _DURATION_BUCKETS:
        if threshold is None or seconds < threshold:
            return slug
    return None


# ---------------------------------------------------------------------------
# Regroupement de noms par forme normalisée (voix et personnages) : la même
# personne/personnage ressort parfois avec une casse ou des accents différents
# d'une histoire à l'autre (ex: "Grand-mère" / "grand-mère", "François MOREL" /
# "François Morel" — variation introduite par les LLM d'identification, un par
# histoire, sans mémoire de leurs décisions sur les autres histoires). Sans ce
# regroupement, la même personne/personnage apparaît comme deux entrées
# distinctes dans les pages "Toutes les voix"/"Personnages".
# ---------------------------------------------------------------------------

def _add_names_to_index(index: dict[str, dict], names: set[str], story: dict) -> None:
    """Ajoute `story` à `index` pour chaque nom de `names`, regroupés par forme
    normalisée (casse/accents) — une seule entrée par histoire même si plusieurs
    variantes du même nom apparaissent dans cette histoire."""
    seen_keys = set()
    for name in names:
        key = db.normalize_text(name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        entry = index.setdefault(key, {"variants": {}, "stories": []})
        entry["variants"][name] = entry["variants"].get(name, 0) + 1
        entry["stories"].append(story)


def _merge_name_variants(index: dict[str, dict]) -> dict[str, list[dict]]:
    """Résout chaque groupe de variantes vers un unique libellé canonique (la
    variante la plus fréquente, départagée alphabétiquement) -> ses histoires."""
    merged: dict[str, list[dict]] = {}
    for data in index.values():
        canonical = sorted(data["variants"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        merged[canonical] = data["stories"]
    return merged


# ---------------------------------------------------------------------------
# Lecture DB spécifique à la génération (jointures propres à cette étape,
# même convention que les autres stages — voir reference/summarize.py)
# ---------------------------------------------------------------------------

async def _get_speaker_map(conn: aiosqlite.Connection, story_id: int) -> dict:
    async with conn.execute(
        "SELECT track_id, speaker_label, character_name FROM speaker_map WHERE story_id = ?",
        (story_id,),
    ) as cur:
        return {(r["track_id"], r["speaker_label"]): r["character_name"] for r in await cur.fetchall()}


async def _full_transcript(conn: aiosqlite.Connection, story_id: int, speaker_map: dict) -> str:
    """Texte intégral, dans l'ordre global (piste puis position temporelle —
    même jointure que reference/summarize.py::_bucket_into_periods), regroupé
    en paragraphes par tour de parole consécutif du même intervenant."""
    async with conn.execute(
        """
        SELECT ts.text, ts.speaker_label, t.id AS track_id
        FROM transcript_segments ts
        JOIN tracks t ON t.id = ts.track_id
        WHERE t.story_id = ?
        ORDER BY t.order_index, ts.start_seconds
        """,
        (story_id,),
    ) as cur:
        rows = await cur.fetchall()

    paragraphs: list[str] = []
    current_speaker: str | None = None
    current_lines: list[str] = []
    for r in rows:
        speaker = speaker_map.get((r["track_id"], r["speaker_label"]), r["speaker_label"])
        if speaker != current_speaker:
            if current_lines:
                paragraphs.append(f"**{current_speaker}** — {' '.join(current_lines)}")
            current_speaker = speaker
            current_lines = []
        current_lines.append(r["text"])
    if current_lines:
        paragraphs.append(f"**{current_speaker}** — {' '.join(current_lines)}")
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Rendu Markdown
# ---------------------------------------------------------------------------

def _format_duration(seconds: float | None) -> str:
    if not seconds:
        return "durée inconnue"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def _format_mmss(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def _playlist_widget_lines(m3u_link: str, download_link: str) -> list[str]:
    """Balisage du lecteur de playlist (voir wiki_theme/playlist.js) — même
    widget pour une histoire multi-pistes que pour un recueil complet, <audio>
    ne sachant pas lire un .m3u directement. `m3u_link` : chemin tel que résolu
    pour les URLs en style dossier ("../" en dur — un attribut HTML brut comme
    data-playlist n'est jamais réécrit par MkDocs, contrairement à un lien
    Markdown normal, voir `download_link`)."""
    return [
        f'<div class="wiki-playlist" data-playlist="{m3u_link}">',
        '  <div class="wiki-playlist-now"></div>',
        '  <audio controls preload="none" style="width:100%"></audio>',
        '  <div class="wiki-playlist-controls">',
        '    <button type="button" class="wiki-playlist-prev">◀ Précédent</button>',
        '    <button type="button" class="wiki-playlist-next">Suivant ▶</button>',
        "  </div>",
        "</div>",
        "",
        f"([Télécharger la playlist .m3u]({download_link}) pour l'ouvrir dans un lecteur externe.)",
        "",
    ]


def _render_story_page(story: dict, themes: list[dict], recueil_info: tuple[str, str] | None,
                        verified_cast: list[dict], voices: list[dict],
                        periods: list[dict], full_text: str, tracks: list[dict]) -> tuple[str, str | None]:
    """Retourne (contenu Markdown, contenu .m3u ou None si une seule piste —
    voir _playlist_widget_lines, pas de playlist utile pour un fichier seul)."""
    lines = [f"# {story['title']}", ""]
    if story.get("short_summary"):
        lines += [story["short_summary"], ""]

    # Écouter : URL absolue de streaming par piste (titre de l'histoire, pas un id
    # brut — voir _track_stream_url/main.py::stream_by_slug) — fournie en clair
    # (pas seulement le lecteur intégré) pour rester copiable/utilisable telle
    # quelle hors du wiki. Plusieurs pistes -> UN SEUL lecteur playlist (comme un
    # recueil) plutôt qu'un <audio> par piste empilé (mauvaise expérience
    # constatée sur une histoire à 18 pistes, voir _playlist_widget_lines).
    m3u_content = None
    if tracks:
        multi = len(tracks) > 1
        story_slug = _story_slug(story["title"], story["id"])
        lines += ["## Écouter", ""]
        if multi:
            lines += _playlist_widget_lines(f"../{story_slug}.m3u", f"{story_slug}.m3u")
            entries = []
            for t in tracks:
                url = _track_stream_url(story["title"], story["id"], t["order_index"], True)
                piste_label = f"Piste {t['order_index'] + 1}"
                lines.append(f"**{piste_label}** ({_format_duration(t.get('duration_seconds'))}) : `{url}`")
                lines.append("")
                entries.append((f"{story['title']} ({piste_label})", t.get("duration_seconds"), url))
            m3u_content = _render_m3u(entries)
        else:
            t = tracks[0]
            url = _track_stream_url(story["title"], story["id"], t["order_index"], False)
            lines.append(f"**Écouter** ({_format_duration(t.get('duration_seconds'))}) : `{url}`")
            lines.append("")
            lines.append(f'<audio controls preload="none" style="width:100%" src="{url}"></audio>')
            lines.append("")

    lines += ["## Informations", ""]
    lines.append(f"- **Durée** : {_format_duration(story.get('total_duration_seconds'))}")
    lines.append(f"- **Tranche d'âge** : {story.get('age_range') or 'non classée'}")
    mood_tags = json.loads(story["mood_tags"]) if story.get("mood_tags") else []
    lines.append(f"- **Ambiances** : {', '.join(mood_tags) if mood_tags else 'aucune'}")
    if themes:
        theme_links = ", ".join(f"[{t['label']}](../themes/{_slugify(t['label'])}.md)" for t in themes)
        lines.append(f"- **Thèmes** : {theme_links}")
    else:
        lines.append("- **Thèmes** : non classés")
    if recueil_info is not None:
        r_label, r_slug = recueil_info
        lines.append(f"- **Recueil** : [{r_label}](../recueils/{r_slug}.md)")
    else:
        lines.append(f"- **Recueil** : {_UNSORTED_LABEL}")
    lines.append("")

    if story.get("long_summary"):
        lines += ["## Résumé détaillé", "", story["long_summary"], ""]

    # Voix & personnages : distinction narrateur / auteur littéraire / casting
    # complet, jamais mélangés (voir docs/concepts-personnes.md) — même
    # priorité de fiabilité que contes_tools.py::story_details (casting
    # vérifié d'abord, clustering acoustique en repli).
    lines += ["## Voix & personnages", ""]
    narrator = story.get("narrator") or "non identifié"
    literary_author = story.get("literary_author") or story.get("author") or "non identifié"
    lines.append(f"- **Narrateur** (qui lit l'histoire) : {narrator}")
    lines.append(f"- **Auteur littéraire** (qui l'a écrite) : {literary_author}")
    lines.append("")
    if verified_cast:
        lines.append("### Casting complet (vérifié)")
        lines.append("")
        for c in verified_cast:
            role = f" — {c['role']}" if c.get("role") else ""
            tag = " *(narrateur)*" if c.get("is_narrator") else ""
            lines.append(f"- {c['name']}{role}{tag}")
        lines.append("")
    elif voices:
        lines.append("### Voix détectées (clustering acoustique, confiance indicative)")
        lines.append("")
        for v in voices:
            lines.append(f"- {v['name']} (confiance {v['confidence']})")
        lines.append("")
    else:
        lines.append("_Aucune voix identifiée pour cette histoire._")
        lines.append("")

    if periods:
        lines += ["## Chapitres", ""]
        for p in periods:
            start, end = _format_mmss(p["global_start_seconds"]), _format_mmss(p["global_end_seconds"])
            lines.append(f"### Chapitre {p['period_index'] + 1} ({start}–{end}) {{: #periode-{p['period_index']} }}")
            lines.append("")
            if p.get("summary_text"):
                lines.append(p["summary_text"])
            lines.append("")

    if full_text:
        lines += [
            "## Texte intégral",
            "",
            '??? note "Afficher le texte intégral"',
            "",
        ]
        for paragraph in full_text.split("\n\n"):
            lines.append(f"    {paragraph}")
            lines.append("")

    return "\n".join(lines), m3u_content


def _render_list_page(title: str, intro: str, items: list[tuple[str, str, str]]) -> str:
    """items = [(label, relative_link, extra_info)]."""
    lines = [f"# {title}", ""]
    if intro:
        lines += [intro, ""]
    for label, link, extra in items:
        suffix = f" — {extra}" if extra else ""
        lines.append(f"- [{label}]({link}){suffix}")
    lines.append("")
    return "\n".join(lines)


def _render_m3u(entries: list[tuple[str, float | None, str]]) -> str:
    """entries = [(label, duration_seconds, url)] — playlist M3U standard,
    lisible par un lecteur média (VLC, lecteur mobile natif...) : évite d'avoir
    à concaténer/transcoder l'audio côté serveur pour "un seul lien qui joue
    tout dans l'ordre", chaque piste restant son URL de streaming existante."""
    lines = ["#EXTM3U"]
    for label, duration_seconds, url in entries:
        duration_int = int(duration_seconds) if duration_seconds else -1
        lines.append(f"#EXTINF:{duration_int},{label}")
        lines.append(url)
    lines.append("")
    return "\n".join(lines)


def _render_recueil_page(label: str, members: list[dict], m3u_link: str | None) -> str:
    lines = [f"# {label}", "", f"{len(members)} histoire(s), dans l'ordre du recueil.", ""]
    if m3u_link:
        lines += ["## Écouter tout le recueil", ""]
        lines += _playlist_widget_lines(f"../{m3u_link}", m3u_link)
    lines += ["## Histoires", ""]
    for m in members:
        lines.append(f"- [{m['title']}](../histoires/{_story_slug(m['title'], m['id'])}.md)")
    lines.append("")
    return "\n".join(lines)


def _render_all_index_page(stories: list[dict], recueils: list[tuple[str, str]]) -> str:
    """Liste combinée, triée alphabétiquement, de toutes les histoires et de tous
    les recueils — chacun renvoyant vers sa page respective. Les recueils sont
    marqués d'un astérisque pour les distinguer des histoires individuelles."""
    entries = []
    for s in stories:
        entries.append((s["title"], f"{_story_slug(s['title'], s['id'])}.md", False))
    for label, link in recueils:
        entries.append((label, link, True))
    entries.sort(key=lambda e: db.normalize_text(e[0]))

    lines = [
        "# Toutes les histoires", "",
        "Toutes les histoires du catalogue et tous les recueils, triés alphabétiquement "
        "ensemble — un recueil est marqué d'un astérisque (*) pour le distinguer d'une "
        "histoire individuelle.", "",
    ]
    for label, link, is_recueil in entries:
        suffix = " \\*" if is_recueil else ""
        lines.append(f"- [{label}{suffix}]({link})")
    lines.append("")
    return "\n".join(lines)


def _render_index_page(n_stories: int, n_recueils: int, n_themes: int, n_voix: int,
                        n_personnages: int) -> str:
    return "\n".join([
        "# Les contes",
        "",
        f"{n_stories} histoires au catalogue.",
        "",
        "## Rechercher",
        "",
        "Recherche libre (titre, thème, ambiance, ou un détail précis d'une histoire) :",
        "",
        '<div class="wiki-search">',
        '  <input id="wiki-search-input" type="search" '
        'placeholder="ex: une histoire de pirates, la recette de la potion…">',
        '  <div id="wiki-search-results"></div>',
        "</div>",
        "",
        "## Explorer par...",
        "",
        f"- [Toutes les histoires](histoires/index.md) ({n_stories + n_recueils})",
        f"- [Thème](themes/index.md) ({n_themes})",
        "- [Tranche d'âge](ages/index.md)",
        "- [Ambiance](ambiances/index.md)",
        "- [Durée](durees/index.md)",
        f"- [Toutes les voix](voix/index.md) ({n_voix})",
        f"- [Tous les personnages](personnages/index.md) ({n_personnages})",
        "",
    ])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def run() -> dict:
    if WIKI_SRC_DIR.exists():
        shutil.rmtree(WIKI_SRC_DIR)
    WIKI_SRC_DIR.mkdir(parents=True, exist_ok=True)
    if WIKI_ASSETS_SRC.exists():
        shutil.copytree(WIKI_ASSETS_SRC, WIKI_SRC_DIR / "assets", dirs_exist_ok=True)

    stories = await db.all_ready_stories()
    story_ids = [s["id"] for s in stories]
    cast_bulk = await db.get_verified_cast_bulk(story_ids)
    themes_bulk = await db.get_theme_classes_bulk(story_ids)
    recueil_groups = _group_by_recueil(stories)
    story_to_recueil = {
        s["id"]: key for key, members in recueil_groups.items() for s in members
    }
    # Nom affiché + slug résolus UNE fois par recueil (voir _recueil_display_name) :
    # partagés par la fiche histoire (lien "Recueil :") et par la page du recueil
    # lui-même, pour qu'ils restent toujours cohérents entre les deux.
    recueil_names: dict[str, tuple[str, str]] = {}
    for key, members in recueil_groups.items():
        display_name = _recueil_display_name(key, members)
        slug = "sans-recueil-identifie" if key == _UNSORTED_KEY else _slugify(display_name)
        recueil_names[key] = (display_name, slug)

    theme_classes = await db.get_theme_classes()
    # Toutes les voix entendues (narrateur ou personnage secondaire — voir
    # docs/concepts-personnes.md), par opposition à personnages_index qui liste
    # les PERSONNAGES DE FICTION tels qu'ils apparaissent dans le transcript
    # (speaker_map.character_name) : deux axes distincts, jamais mélangés.
    # dict[str, dict] pendant la collecte : {"variants": {nom_brut: nb_occurrences},
    # "stories": [...]}, indexé par forme normalisée — résolu en dict[str, list[dict]]
    # (libellé canonique -> histoires) par _merge_name_variants avant génération des pages.
    voix_index: dict[str, dict] = {}
    personnages_index: dict[str, dict] = {}
    age_index: dict[str, list[dict]] = {}
    mood_index: dict[str, list[dict]] = {}
    duration_index: dict[str, list[dict]] = {}

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        for story in stories:
            sid = story["id"]
            verified_cast = cast_bulk.get(sid, [])
            voices = [] if verified_cast else await db.get_voices_for_story(sid)
            periods = await db.get_periods_for_story(sid)
            speaker_map = await _get_speaker_map(conn, sid)
            full_text = await _full_transcript(conn, sid, speaker_map)
            tracks = await db.get_tracks_for_story(sid)
            recueil_key = story_to_recueil.get(sid)
            recueil_info = recueil_names.get(recueil_key) if recueil_key is not None else None

            slug = _story_slug(story["title"], sid)
            content, m3u_content = _render_story_page(
                story, themes_bulk.get(sid, []), recueil_info,
                verified_cast, voices, periods, full_text, tracks,
            )
            await _write(WIKI_SRC_DIR / "histoires" / f"{slug}.md", content)
            if m3u_content:
                await _write(WIKI_SRC_DIR / "histoires" / f"{slug}.m3u", m3u_content)

            # Toutes les voix (narrateur ou pas) : casting vérifié + repli clustering
            # acoustique, comme sur la fiche histoire elle-même, plus stories.narrator
            # en filet de sécurité si ni l'un ni l'autre ne l'a capté.
            voice_names = {c["name"] for c in verified_cast} | {v["name"] for v in voices}
            if story.get("narrator"):
                voice_names.add(story["narrator"])
            _add_names_to_index(voix_index, voice_names, story)

            # Personnages de fiction tels qu'ils apparaissent dans le transcript
            # (speaker_map résolu uniquement — pas les labels SPEAKER_XX bruts non
            # identifiés, voir _full_transcript).
            _add_names_to_index(personnages_index, set(speaker_map.values()), story)

            if story.get("age_range"):
                age_index.setdefault(story["age_range"], []).append(story)
            for tag in (json.loads(story["mood_tags"]) if story.get("mood_tags") else []):
                mood_index.setdefault(tag, []).append(story)
            dslug = _duration_bucket_slug(story.get("total_duration_seconds"))
            if dslug:
                duration_index.setdefault(dslug, []).append(story)

    # Recueils : ordre narratif (suffixe "#N", voir _recueil_member_order_key) plutôt
    # qu'alphabétique — retombe naturellement sur l'alphabétique pour les recueils
    # jamais scindés (tous à égalité sur la clé primaire, départagés par le titre).
    recueil_items = []
    for key, members in sorted(recueil_groups.items(), key=lambda kv: recueil_names[kv[0]][0]):
        if not members:
            continue
        label, rslug = recueil_names[key]
        ordered_members = sorted(
            members,
            key=lambda m: (_recueil_member_order_key(m["folder_path"]), db.normalize_text(m["title"])),
        )

        m3u_link = None
        if key != _UNSORTED_KEY:
            # Pas de playlist pour "Sans recueil identifié" : ces histoires n'ont
            # justement aucun rapport narratif entre elles, les enchaîner n'aurait
            # pas de sens (voir _UNSORTED_LABEL).
            entries = []
            for m in ordered_members:
                m_tracks = await db.get_tracks_for_story(m["id"])
                multi = len(m_tracks) > 1
                for t in m_tracks:
                    url = _track_stream_url(m["title"], m["id"], t["order_index"], multi)
                    piste_suffix = f" (piste {t['order_index'] + 1})" if multi else ""
                    entries.append((f"{m['title']}{piste_suffix}", t.get("duration_seconds"), url))
            if entries:
                await _write(WIKI_SRC_DIR / "recueils" / f"{rslug}.m3u", _render_m3u(entries))
                m3u_link = f"{rslug}.m3u"

        await _write(
            WIKI_SRC_DIR / "recueils" / f"{rslug}.md",
            _render_recueil_page(label, ordered_members, m3u_link),
        )
        recueil_items.append((label, f"{rslug}.md", f"{len(members)} histoire(s)"))
    await _write(
        WIKI_SRC_DIR / "recueils" / "index.md",
        _render_list_page(
            "Recueils",
            "Regroupement best-effort depuis l'arborescence des dossiers d'origine — "
            "n'est pas garanti exhaustif ni toujours pertinent, voir "
            f"« {_UNSORTED_LABEL} » pour les histoires non regroupées.",
            recueil_items,
        ),
    )

    # Toutes les histoires + tous les recueils, triés alphabétiquement ensemble
    # (voir histoires/index.md) — liens vers les recueils relatifs à ../recueils/.
    recueil_links_for_all = [
        (label, f"../recueils/{link}") for label, link, _extra in recueil_items
    ]
    await _write(
        WIKI_SRC_DIR / "histoires" / "index.md",
        _render_all_index_page(stories, recueil_links_for_all),
    )

    # Thèmes
    theme_items = []
    for tc in theme_classes:
        members = [s for s in stories if tc["id"] in {t["id"] for t in themes_bulk.get(s["id"], [])}]
        if not members:
            continue
        tslug = _slugify(tc["label"])
        story_items = [
            (m["title"], f"../histoires/{_story_slug(m['title'], m['id'])}.md", "")
            for m in sorted(members, key=lambda m: m["title"])
        ]
        await _write(
            WIKI_SRC_DIR / "themes" / f"{tslug}.md",
            _render_list_page(tc["label"], tc["description"], story_items),
        )
        theme_items.append((tc["label"], f"{tslug}.md", f"{len(members)} histoire(s)"))
    await _write(
        WIKI_SRC_DIR / "themes" / "index.md",
        _render_list_page("Thèmes", "", theme_items),
    )

    # Toutes les voix (narrateur ou personnage secondaire)
    voix_index_resolved = _merge_name_variants(voix_index)
    voix_items = []
    for name, members in sorted(voix_index_resolved.items(), key=lambda kv: db.normalize_text(kv[0])):
        vslug = _slugify(name)
        story_items = [
            (m["title"], f"../histoires/{_story_slug(m['title'], m['id'])}.md", "")
            for m in sorted(members, key=lambda m: m["title"])
        ]
        await _write(
            WIKI_SRC_DIR / "voix" / f"{vslug}.md",
            _render_list_page(name, f"{len(members)} histoire(s) où {name} se fait entendre.", story_items),
        )
        voix_items.append((name, f"{vslug}.md", f"{len(members)} histoire(s)"))
    await _write(
        WIKI_SRC_DIR / "voix" / "index.md",
        _render_list_page(
            "Toutes les voix",
            "Toute personne entendue dans un enregistrement — narrateur principal ou "
            "voix secondaire — voir docs/concepts-personnes.md pour la distinction avec "
            "les personnages de fiction (page « Personnages »).",
            voix_items,
        ),
    )

    # Personnages de fiction (distincts des voix/interprètes ci-dessus)
    personnages_index_resolved = _merge_name_variants(personnages_index)
    personnage_items = []
    for name, members in sorted(personnages_index_resolved.items(), key=lambda kv: db.normalize_text(kv[0])):
        pslug = _slugify(name)
        story_items = [
            (m["title"], f"../histoires/{_story_slug(m['title'], m['id'])}.md", "")
            for m in sorted(members, key=lambda m: m["title"])
        ]
        await _write(
            WIKI_SRC_DIR / "personnages" / f"{pslug}.md",
            _render_list_page(name, f"{len(members)} histoire(s) où apparaît {name}.", story_items),
        )
        personnage_items.append((name, f"{pslug}.md", f"{len(members)} histoire(s)"))
    await _write(
        WIKI_SRC_DIR / "personnages" / "index.md",
        _render_list_page(
            "Personnages",
            "Personnages tels qu'ils apparaissent dans les transcriptions (speaker_map) — "
            "voir docs/concepts-personnes.md pour la distinction avec les voix/interprètes "
            "réels (page « Toutes les voix »).",
            personnage_items,
        ),
    )

    # Âges (vocabulaire fixe)
    age_items = []
    for age, members in sorted(age_index.items()):
        story_items = [
            (m["title"], f"../histoires/{_story_slug(m['title'], m['id'])}.md", "")
            for m in sorted(members, key=lambda m: m["title"])
        ]
        await _write(
            WIKI_SRC_DIR / "ages" / f"{age}.md",
            _render_list_page(age, f"{len(members)} histoire(s).", story_items),
        )
        age_items.append((age, f"{age}.md", f"{len(members)} histoire(s)"))
    await _write(WIKI_SRC_DIR / "ages" / "index.md", _render_list_page("Tranches d'âge", "", age_items))

    # Ambiances (vocabulaire fixe)
    mood_items = []
    for mood, members in sorted(mood_index.items()):
        story_items = [
            (m["title"], f"../histoires/{_story_slug(m['title'], m['id'])}.md", "")
            for m in sorted(members, key=lambda m: m["title"])
        ]
        await _write(
            WIKI_SRC_DIR / "ambiances" / f"{mood}.md",
            _render_list_page(mood, f"{len(members)} histoire(s).", story_items),
        )
        mood_items.append((mood, f"{mood}.md", f"{len(members)} histoire(s)"))
    await _write(WIKI_SRC_DIR / "ambiances" / "index.md", _render_list_page("Ambiances", "", mood_items))

    # Durées (paliers fixes, ordre chronologique — pas alphabétique, voir _DURATION_BUCKETS)
    duration_items = []
    for _threshold, dslug, dlabel in _DURATION_BUCKETS:
        members = duration_index.get(dslug, [])
        if not members:
            continue
        story_items = [
            (m["title"], f"../histoires/{_story_slug(m['title'], m['id'])}.md", "")
            for m in sorted(members, key=lambda m: m["title"])
        ]
        await _write(
            WIKI_SRC_DIR / "durees" / f"{dslug}.md",
            _render_list_page(dlabel, f"{len(members)} histoire(s).", story_items),
        )
        duration_items.append((dlabel, f"{dslug}.md", f"{len(members)} histoire(s)"))
    await _write(WIKI_SRC_DIR / "durees" / "index.md", _render_list_page("Durées", "", duration_items))

    # Accueil
    await _write(
        WIKI_SRC_DIR / "index.md",
        _render_index_page(len(stories), len(recueil_items), len(theme_items),
                            len(voix_items), len(personnage_items)),
    )

    from mkdocs.commands.build import build as mkdocs_build
    from mkdocs.config import load_config

    if WIKI_SITE_DIR.exists():
        shutil.rmtree(WIKI_SITE_DIR)
    cfg = load_config(str(MKDOCS_CONFIG), docs_dir=str(WIKI_SRC_DIR), site_dir=str(WIKI_SITE_DIR))
    mkdocs_build(cfg)

    result = {
        "stories": len(stories),
        "recueils": len(recueil_items),
        "themes": len(theme_items),
        "voix": len(voix_items),
        "personnages": len(personnage_items),
    }
    logger.info(f"build_wiki: {result}")
    return result
