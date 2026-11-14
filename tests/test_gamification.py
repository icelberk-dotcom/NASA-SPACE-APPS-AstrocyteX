from datetime import date

import pytest

from backend.gamification import (BADGES, Gamification, compute_streak, level_bounds, level_for_xp, rank_for_level,
                                  task_points)
from backend.storage import Storage


@pytest.fixture()
def gam(tmp_path):
    db = Storage(tmp_path / "g.db")
    db.ensure_astronaut("a1")
    return Gamification(db), db


def _task(db, severity="yellow", tag="hydration", minutes=15):
    return db.add_task("a1", {"parameter_key": "fluid_intake_ml", "title": f"t{tag}{severity}{minutes}{db.total_xp('a1')}",
                              "description": "d", "severity_level": severity, "risk_pct": 55.0, "duration_min": minutes,
                              "points": task_points(severity, minutes), "tag": tag, "source": "test"})


def test_points_scale_with_severity_and_duration():
    assert task_points("yellow", 5) < task_points("red", 5) < task_points("black", 5)
    assert task_points("yellow", 60) == 10 + 12 and task_points("yellow", 1) == 10


def test_levels_and_ranks():
    assert level_for_xp(0) == 1 and level_for_xp(59) == 1 and level_for_xp(60) == 2 and level_for_xp(240) == 3
    lo, hi = level_bounds(3)
    assert (lo, hi) == (240, 540)
    assert rank_for_level(1) == "Cadet" and rank_for_level(99) == "Fleet Admiral"


def test_completing_a_task_awards_points_once(gam):
    g, db = gam
    tid = _task(db)
    res = g.complete_task("a1", tid)
    assert res.points_awarded == task_points("yellow", 15) and res.state.xp == res.points_awarded
    assert "first_step" in [b.id for b in res.new_badges]
    again = g.complete_task("a1", tid)  # idempotent
    assert again.already_completed and again.points_awarded == 0 and g.state("a1").xp == res.points_awarded


def test_level_up_and_progress_bar(gam):
    g, db = gam
    results = [g.complete_task("a1", _task(db, "red", "exercise", 20)) for _ in range(3)]
    assert results[-1].state.level >= 2 and 0.0 <= results[-1].state.progress < 1.0
    assert sum(r.leveled_up for r in results) >= 1  # the level-up flag fires exactly when the level changes


def test_badges_for_tag_counts_and_black_alert(gam):
    g, db = gam
    for _ in range(3):
        g.complete_task("a1", _task(db, "yellow", "hydration", 5))
    earned = {b.id for b in g.state("a1").badges if b.earned}
    assert {"first_step", "hydration_hero"} <= earned and "code_black_survivor" not in earned
    res = g.complete_task("a1", _task(db, "black", "medical", 5))
    assert "code_black_survivor" in [b.id for b in res.new_badges] and "houston_called" in [b.id for b in res.new_badges]


def test_cannot_complete_foreign_task(gam):
    g, db = gam
    db.ensure_astronaut("a2")
    tid = _task(db)
    with pytest.raises(KeyError):
        g.complete_task("a2", tid)


def test_streak_logic():
    today = date(2026, 11, 14)
    assert compute_streak([], today) == 0
    assert compute_streak(["2026-11-14", "2026-11-13", "2026-11-12"], today) == 3
    assert compute_streak(["2026-11-13", "2026-11-12"], today) == 2  # yesterday still counts
    assert compute_streak(["2026-11-10"], today) == 0
    assert compute_streak(["2026-11-14", "2026-11-12"], today) == 1  # gap breaks the streak


def test_badge_ids_unique():
    assert len({b.id for b in BADGES}) == len(BADGES)
