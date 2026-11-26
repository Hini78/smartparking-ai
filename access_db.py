"""SQLite для демо (часть 4): белый список и история решений.

База лежит в data/processed/access.sqlite и в git не попадает
(папка data/ уже в .gitignore).

В белом списке хранится уже нормализованная строка — та же,
которую возвращает CRNN (латиница, без пробелов).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from demo_core import DECISION_LABELS, PLATE_RE, normalize_plate

DB_PATH = Path("data/processed/access.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS whitelist (
    plate    TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    source   TEXT NOT NULL,      -- 'фото' или 'видео'
    plate    TEXT,               -- что прочитала CRNN (может быть пустым)
    ocr_conf REAL,
    det_conf REAL,
    votes    INTEGER,             -- для видео: сколько кадров проголосовало
    decision TEXT NOT NULL,       -- ALLOW / DENY / MANUAL
    reason   TEXT,
    time_ms  REAL
);
"""


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Открывает базу и создаёт таблицы, если их ещё нет."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    con.commit()
    return con


# -------------------------------------------------------------------------- #
# Белый список
# -------------------------------------------------------------------------- #
def get_whitelist(con: sqlite3.Connection) -> list[str]:
    rows = con.execute("SELECT plate FROM whitelist ORDER BY plate").fetchall()
    return [r["plate"] for r in rows]


def add_plate(con: sqlite3.Connection, plate: str) -> tuple[bool, str]:
    """Добавляет нормализованный номер. Возвращает (успех, сообщение)."""
    norm = normalize_plate(plate)
    if not norm:
        return False, "пустая строка"
    if not PLATE_RE.fullmatch(norm):
        return False, (f"«{norm}» не похож на российский номер "
                       "(буква, 3 цифры, 2 буквы, регион из 2-3 цифр)")
    try:
        con.execute("INSERT INTO whitelist (plate, added_at) VALUES (?, ?)",
                    (norm, datetime.now().isoformat(timespec="seconds")))
        con.commit()
    except sqlite3.IntegrityError:
        return False, f"номер {norm} уже есть в списке"
    return True, f"добавлен номер {norm}"


def remove_plate(con: sqlite3.Connection, plate: str) -> tuple[bool, str]:
    norm = normalize_plate(plate)
    cur = con.execute("DELETE FROM whitelist WHERE plate = ?", (norm,))
    con.commit()
    if cur.rowcount:
        return True, f"номер {norm} удалён"
    return False, f"номера {norm} нет в списке"


# -------------------------------------------------------------------------- #
# История
# -------------------------------------------------------------------------- #
def add_history(con: sqlite3.Connection, *, source: str, plate: str,
                ocr_conf: float, det_conf, votes, decision: str,
                reason: str, time_ms: float) -> None:
    con.execute(
        "INSERT INTO history (ts, source, plate, ocr_conf, det_conf, votes,"
        " decision, reason, time_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), source, plate or None,
         ocr_conf, det_conf, votes, decision, reason, time_ms))
    con.commit()


def get_history(con: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    rows = con.execute(
        "SELECT ts, source, plate, ocr_conf, det_conf, votes, decision,"
        " reason, time_ms FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return rows


def get_stats(con: sqlite3.Connection, day: str) -> dict[str, dict]:
    """Счётчики решений за день `day` (префикс ISO-даты, например '2026-11-25').

    Возвращает {decision: {"count": int, "avg_ms": float}}.
    """
    rows = con.execute(
        "SELECT decision, COUNT(*) AS c, AVG(time_ms) AS t FROM history"
        " WHERE ts LIKE ? GROUP BY decision", (day + "%",)).fetchall()
    return {r["decision"]: {"count": r["c"], "avg_ms": r["t"]} for r in rows}


def decision_label(decision: str) -> str:
    return DECISION_LABELS.get(decision, decision)
