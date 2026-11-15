"""
SQLite persistence for astronauts, readings, tasks, XP and badges.

Uses only the Python standard library (``sqlite3``).  A fresh connection is opened per
operation (WAL mode) which keeps it safe for FastAPI worker threads and Streamlit reruns.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS astronauts (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, astronaut_id TEXT NOT NULL, ts TEXT NOT NULL,
    values_json TEXT NOT NULL, assessment_json TEXT);
CREATE INDEX IF NOT EXISTS idx_readings_astro ON readings(astronaut_id, id);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, astronaut_id TEXT NOT NULL, parameter_key TEXT NOT NULL,
    title TEXT NOT NULL, description TEXT NOT NULL, severity_level TEXT NOT NULL, risk_pct REAL NOT NULL,
    duration_min INTEGER NOT NULL, points INTEGER NOT NULL, tag TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
    source TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT);
CREATE INDEX IF NOT EXISTS idx_tasks_astro ON tasks(astronaut_id, status);
CREATE TABLE IF NOT EXISTS xp_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, astronaut_id TEXT NOT NULL, task_id INTEGER, points INTEGER NOT NULL, ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS badges (
    astronaut_id TEXT NOT NULL, badge_id TEXT NOT NULL, earned_at TEXT NOT NULL, PRIMARY KEY (astronaut_id, badge_id));
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------- astronauts
    def ensure_astronaut(self, astronaut_id: str, name: Optional[str] = None) -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO astronauts(id, name, created_at) VALUES (?,?,?)",
                      (astronaut_id, name or astronaut_id, utcnow()))

    def list_astronauts(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM astronauts ORDER BY created_at")]

    # ---------------------------------------------------------------- readings
    def add_reading(self, astronaut_id: str, values: Dict[str, float], assessment: Optional[Dict[str, Any]]) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO readings(astronaut_id, ts, values_json, assessment_json) VALUES (?,?,?,?)",
                            (astronaut_id, utcnow(), json.dumps(values),
                             json.dumps(assessment, default=float) if assessment else None))
            return int(cur.lastrowid)

    def recent_values(self, astronaut_id: str, n: int = 13) -> List[Dict[str, float]]:
        """Last `n` stored value-sets, oldest first."""
        with self._conn() as c:
            rows = c.execute("SELECT values_json FROM readings WHERE astronaut_id=? ORDER BY id DESC LIMIT ?",
                             (astronaut_id, n)).fetchall()
        return [json.loads(r["values_json"]) for r in reversed(rows)]

    def latest_assessment(self, astronaut_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT assessment_json FROM readings WHERE astronaut_id=? AND assessment_json IS NOT NULL "
                            "ORDER BY id DESC LIMIT 1", (astronaut_id,)).fetchone()
        return json.loads(row["assessment_json"]) if row else None

    def risk_history(self, astronaut_id: str, n: int = 60) -> List[Dict[str, Any]]:
        """Overall risk over time (for the trend chart)."""
        with self._conn() as c:
            rows = c.execute("SELECT ts, assessment_json FROM readings WHERE astronaut_id=? AND assessment_json IS NOT NULL "
                             "ORDER BY id DESC LIMIT ?", (astronaut_id, n)).fetchall()
        out = []
        for r in reversed(rows):
            a = json.loads(r["assessment_json"])
            out.append({"ts": r["ts"], "overall_risk_pct": a["overall"]["risk_pct"],
                        "level": a["overall"]["level"]})
        return out

    # ------------------------------------------------------------------- tasks
    def add_task(self, astronaut_id: str, t: Dict[str, Any]) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO tasks(astronaut_id, parameter_key, title, description, severity_level, risk_pct, duration_min, "
                "points, tag, status, source, created_at) VALUES (?,?,?,?,?,?,?,?,?, 'pending', ?, ?)",
                (astronaut_id, t["parameter_key"], t["title"], t["description"], t["severity_level"], t["risk_pct"],
                 t["duration_min"], t["points"], t["tag"], t["source"], utcnow()))
            return int(cur.lastrowid)

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list_tasks(self, astronaut_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM tasks WHERE astronaut_id=?", [astronaut_id]
        if status:
            q += " AND status=?"
            args.append(status)
        q += " ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id DESC"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    def update_task_text(self, task_id: int, title: str, description: str, tag: Optional[str] = None) -> None:
        with self._conn() as c:
            if tag is not None:
                c.execute("UPDATE tasks SET title=?, description=?, tag=? WHERE id=?", (title, description, tag, task_id))
            else:
                c.execute("UPDATE tasks SET title=?, description=? WHERE id=?", (title, description, task_id))

    def complete_task(self, task_id: int) -> bool:
        """Atomically flip pending -> done. Returns False if it was already completed."""
        with self._conn() as c:
            cur = c.execute("UPDATE tasks SET status='done', completed_at=? WHERE id=? AND status='pending'",
                            (utcnow(), task_id))
            return cur.rowcount == 1

    # --------------------------------------------------------------- xp / stats
    def add_xp(self, astronaut_id: str, task_id: Optional[int], points: int) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO xp_events(astronaut_id, task_id, points, ts) VALUES (?,?,?,?)",
                      (astronaut_id, task_id, points, utcnow()))

    def total_xp(self, astronaut_id: str) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COALESCE(SUM(points),0) FROM xp_events WHERE astronaut_id=?", (astronaut_id,)).fetchone()[0])

    def completion_stats(self, astronaut_id: str) -> Dict[str, Any]:
        with self._conn() as c:
            done = c.execute("SELECT tag, severity_level, completed_at FROM tasks WHERE astronaut_id=? AND status='done'",
                             (astronaut_id,)).fetchall()
        by_tag: Dict[str, int] = {}
        black = 0
        dates = set()
        for r in done:
            by_tag[r["tag"]] = by_tag.get(r["tag"], 0) + 1
            black += r["severity_level"] == "black"
            if r["completed_at"]:
                dates.add(r["completed_at"][:10])
        return {"tasks_completed": len(done), "by_tag": by_tag, "black_done": black, "dates": sorted(dates)}

    # ------------------------------------------------------------------ badges
    def earned_badges(self, astronaut_id: str) -> Dict[str, str]:
        with self._conn() as c:
            return {r["badge_id"]: r["earned_at"] for r in
                    c.execute("SELECT badge_id, earned_at FROM badges WHERE astronaut_id=?", (astronaut_id,))}

    def award_badge(self, astronaut_id: str, badge_id: str) -> bool:
        with self._conn() as c:
            cur = c.execute("INSERT OR IGNORE INTO badges(astronaut_id, badge_id, earned_at) VALUES (?,?,?)",
                            (astronaut_id, badge_id, utcnow()))
            return cur.rowcount == 1
