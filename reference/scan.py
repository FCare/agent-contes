import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

import db

logger = logging.getLogger(__name__)

CONTES_ROOT = Path(os.environ.get("CONTES_ROOT", "/contes"))
EXCLUDED_FILE = Path(__file__).parent / "excluded_folders.txt"
AUDIO_EXTENSIONS = (".mp3", ".ogg")

# Matches a trailing "Disc 1" / "(CD2)" / "Disque 3" style suffix used to
# split one story across several physical folders.
_DISC_SUFFIX_RE = re.compile(
    r"[\s_]*[\(\[]?\s*(?:disc|disque|cd)\s*\.?\s*(\d+)\s*[\)\]]?\s*$", re.IGNORECASE
)
_TRACK_NUM_RE = re.compile(r"^0*(\d+)")


def _load_excluded() -> set[str]:
    if not EXCLUDED_FILE.exists():
        return set()
    return {
        line.strip()
        for line in EXCLUDED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _normalize(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return ascii_name.strip().lower()


def _disc_number(name: str) -> int | None:
    m = _DISC_SUFFIX_RE.search(name)
    return int(m.group(1)) if m else None


def _strip_disc_suffix(name: str) -> str:
    return _DISC_SUFFIX_RE.sub("", name).strip()


def _canonical_key(name: str) -> str:
    return _normalize(_strip_disc_suffix(name))


def _track_sort_key(filename: str) -> tuple:
    m = _TRACK_NUM_RE.match(filename)
    return (0, int(m.group(1))) if m else (1, filename.lower())


def track_title(file_path: str) -> str:
    """Human-readable title for a single track: filename without extension,
    directory, or leading track number."""
    stem = Path(file_path).stem
    return _TRACK_NUM_RE.sub("", stem).strip()


def _find_story_folders(root: Path) -> list[Path]:
    """Any directory containing at least one audio file directly is a story folder."""
    folders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(f.lower().endswith(AUDIO_EXTENSIONS) for f in filenames):
            folders.append(Path(dirpath))
    return folders


def _group_stories(folders: list[Path]) -> list[dict]:
    """Group sibling folders that differ only by a Disc/CD suffix into one story."""
    groups: dict[tuple, list[Path]] = {}
    for folder in folders:
        key = (folder.parent, _canonical_key(folder.name))
        groups.setdefault(key, []).append(folder)

    stories = []
    for group_folders in groups.values():
        ordered = sorted(group_folders, key=lambda f: _disc_number(f.name) or 1)
        tracks = []
        for folder in ordered:
            audio_files = sorted(
                (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS),
                key=lambda f: _track_sort_key(f.name),
            )
            tracks.extend(audio_files)
        stories.append({
            "author": ordered[0].relative_to(CONTES_ROOT).parts[0],
            "title": _strip_disc_suffix(ordered[0].name),
            "folder_path": ordered[0].relative_to(CONTES_ROOT).as_posix(),
            "merged_folders": [f.relative_to(CONTES_ROOT).as_posix() for f in ordered],
            "tracks": [t.relative_to(CONTES_ROOT).as_posix() for t in tracks],
        })
    return stories


async def scan(only_new: bool = False) -> dict:
    excluded = _load_excluded()
    stories = _group_stories(_find_story_folders(CONTES_ROOT))
    now = datetime.now(timezone.utc).isoformat()
    n_new = n_updated = n_excluded = n_tracks = 0

    async with aiosqlite.connect(db.DB_PATH) as conn:
        for story in stories:
            status = "excluded" if story["author"] in excluded else "discovered"
            if status == "excluded":
                n_excluded += 1

            async with conn.execute(
                "SELECT id FROM stories WHERE folder_path = ?", (story["folder_path"],)
            ) as cur:
                existing = await cur.fetchone()

            if existing and only_new:
                continue

            await conn.execute(
                """
                INSERT INTO stories (title, author, folder_path, merged_folders, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(folder_path) DO UPDATE SET
                    title = excluded.title,
                    merged_folders = excluded.merged_folders,
                    updated_at = excluded.updated_at,
                    status = CASE WHEN excluded.status = 'excluded' THEN 'excluded' ELSE stories.status END
                """,
                (story["title"], story["author"], story["folder_path"],
                 json.dumps(story["merged_folders"], ensure_ascii=False),
                 status, now, now),
            )
            n_new += 0 if existing else 1
            n_updated += 1 if existing else 0

            async with conn.execute(
                "SELECT id FROM stories WHERE folder_path = ?", (story["folder_path"],)
            ) as cur:
                (story_id,) = await cur.fetchone()

            for order_index, rel_path in enumerate(story["tracks"]):
                # ON CONFLICT(file_path), pas (story_id, file_path) : si split_stories a
                # depuis réattribué cette piste à une autre histoire, on ne veut pas en
                # recréer une copie fantôme sous l'histoire d'origine — juste laisser
                # story_id tel qu'il est et ne toucher que l'ordre.
                await conn.execute(
                    """
                    INSERT INTO tracks (story_id, order_index, file_path, status)
                    VALUES (?, ?, ?, 'discovered')
                    ON CONFLICT(file_path) DO UPDATE SET
                        order_index = excluded.order_index
                    """,
                    (story_id, order_index, rel_path),
                )
                n_tracks += 1
        await conn.commit()

    result = {"stories": len(stories), "new": n_new, "updated": n_updated,
              "excluded": n_excluded, "tracks": n_tracks}
    logger.info(f"scan: {result}")
    return result
