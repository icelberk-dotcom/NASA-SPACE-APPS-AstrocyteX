"""
Gamification: XP, levels, ranks, streaks and badges.

Rules
-----
* Completing a mitigation task awards the task's points exactly once (idempotent).
* Points scale with the severity of the problem that created the task
  (yellow 10, red 25, black 50) plus a small bonus for longer tasks.
* Level ``L`` starts at ``60 * (L-1)^2`` XP, so early levels come quickly and later ones
  need steadily more effort; the UI progress bar shows the fraction to the next level.
* A *streak* is the number of consecutive UTC days (ending today or yesterday) on which at
  least one task was completed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List

from backend.schemas import BadgeOut, CompletionResult, GamificationState, TaskOut
from backend.storage import Storage

XP_PER_LEVEL_UNIT = 60
RANKS = ["Cadet", "Specialist", "Flight Engineer", "Mission Specialist", "Pilot", "Commander", "Fleet Admiral"]
SEVERITY_POINTS = {"yellow": 10, "red": 25, "black": 50}


def task_points(severity_level: str, duration_min: int) -> int:
    """Points for a task: severity base + 2 points per 5 minutes (max +12)."""
    return SEVERITY_POINTS.get(severity_level, 10) + min(duration_min // 5, 6) * 2


def level_for_xp(xp: int) -> int:
    return int(math.sqrt(max(xp, 0) / XP_PER_LEVEL_UNIT)) + 1


def level_bounds(level: int) -> tuple[int, int]:
    """(xp needed to *reach* this level, xp needed for the next one)."""
    return XP_PER_LEVEL_UNIT * (level - 1) ** 2, XP_PER_LEVEL_UNIT * level ** 2


def rank_for_level(level: int) -> str:
    return RANKS[min((level - 1) // 2, len(RANKS) - 1)]


def compute_streak(date_strings: List[str], today: date | None = None) -> int:
    today = today or datetime.now(timezone.utc).date()
    days = {date.fromisoformat(d) for d in date_strings}
    cursor = today if today in days else today - timedelta(days=1)
    streak = 0
    while cursor in days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


@dataclass(frozen=True)
class BadgeDef:
    id: str
    name: str
    description: str
    icon: str
    check: Callable[[Dict[str, Any]], bool]


def _tag(tag: str, n: int) -> Callable[[Dict[str, Any]], bool]:
    return lambda s: s["by_tag"].get(tag, 0) >= n


BADGES: List[BadgeDef] = [
    BadgeDef("first_step", "First Step", "Complete your first mitigation task.", "STEP", lambda s: s["tasks_completed"] >= 1),
    BadgeDef("hydration_hero", "Hydration Hero", "Complete 3 hydration tasks.", "H2O", _tag("hydration", 3)),
    BadgeDef("iron_lungs", "Iron Lungs", "Complete 3 guided-breathing tasks.", "LUNG", _tag("breathing", 3)),
    BadgeDef("bone_guardian", "Bone Guardian", "Complete 3 resistance/exercise tasks (ARED, treadmill).", "ARED", _tag("exercise", 3)),
    BadgeDef("sleep_master", "Sleep Master", "Complete 2 sleep-hygiene tasks.", "SLP", _tag("sleep", 2)),
    BadgeDef("chow_champion", "Chow Champion", "Complete 3 nutrition tasks.", "NUTR", _tag("nutrition", 3)),
    BadgeDef("zero_g_zen", "Zero-G Zen", "Complete 2 wellbeing tasks.", "CALM", _tag("wellbeing", 2)),
    BadgeDef("ecg_ace", "ECG Ace", "Complete an ECG re-check task.", "ECG", _tag("ecg", 1)),
    BadgeDef("houston_called", "Houston, We Called", "Complete a task that involves contacting the flight surgeon.", "COMM", _tag("medical", 1)),
    BadgeDef("storm_shelter", "Storm Shelter", "Complete a radiation-protection task (shelter, dosimetry, dose budget).", "SHLD", _tag("radiation", 1)),
    BadgeDef("habitat_keeper", "Habitat Keeper", "Complete 3 habitat / environment tasks (CO2, O2, temperature, noise).", "ENV", _tag("environment", 3)),
    BadgeDef("code_black_survivor", "Code Black Survivor", "Complete a task created by a black (critical) alert.", "CRIT", lambda s: s["black_done"] >= 1),
    BadgeDef("orbit_streak_3", "3-Day Orbit Streak", "Complete tasks 3 days in a row.", "D03", lambda s: s["streak"] >= 3),
    BadgeDef("week_in_orbit", "Week in Orbit", "Complete tasks 7 days in a row.", "D07", lambda s: s["streak"] >= 7),
    BadgeDef("mission_ready", "Mission Ready", "Complete 10 tasks.", "M10", lambda s: s["tasks_completed"] >= 10),
    BadgeDef("mission_veteran", "Mission Veteran", "Complete 25 tasks.", "M25", lambda s: s["tasks_completed"] >= 25),
    BadgeDef("rising_star", "Rising Star", "Reach level 3.", "LV03", lambda s: s["level"] >= 3),
    BadgeDef("stellar_commander", "Stellar Commander", "Reach level 6.", "LV06", lambda s: s["level"] >= 6),
]
BADGE_BY_ID = {b.id: b for b in BADGES}


class Gamification:
    def __init__(self, storage: Storage):
        self.db = storage

    def _badge_out(self, b: BadgeDef, earned: Dict[str, str]) -> BadgeOut:
        return BadgeOut(id=b.id, name=b.name, description=b.description, icon=b.icon,
                        earned=b.id in earned, earned_at=earned.get(b.id))

    def _stats(self, astronaut_id: str) -> Dict[str, Any]:
        stats = self.db.completion_stats(astronaut_id)
        stats["streak"] = compute_streak(stats["dates"])
        stats["level"] = level_for_xp(self.db.total_xp(astronaut_id))
        return stats

    def state(self, astronaut_id: str) -> GamificationState:
        xp = self.db.total_xp(astronaut_id)
        level = level_for_xp(xp)
        lo, hi = level_bounds(level)
        stats = self._stats(astronaut_id)
        earned = self.db.earned_badges(astronaut_id)
        return GamificationState(
            astronaut_id=astronaut_id, xp=xp, level=level, rank=rank_for_level(level), level_xp=lo, next_level_xp=hi,
            progress=round((xp - lo) / max(hi - lo, 1), 4), streak_days=stats["streak"],
            tasks_completed=stats["tasks_completed"], badges=[self._badge_out(b, earned) for b in BADGES])

    def complete_task(self, astronaut_id: str, task_id: int) -> CompletionResult:
        task = self.db.get_task(task_id)
        if task is None or task["astronaut_id"] != astronaut_id:
            raise KeyError(f"task {task_id} not found for astronaut {astronaut_id}")
        before_level = level_for_xp(self.db.total_xp(astronaut_id))
        if not self.db.complete_task(task_id):  # already done -> no points, idempotent
            return CompletionResult(task=TaskOut(**self.db.get_task(task_id)), points_awarded=0, already_completed=True,
                                    state=self.state(astronaut_id))
        points = int(task["points"])
        self.db.add_xp(astronaut_id, task_id, points)
        stats = self._stats(astronaut_id)
        new_badges: List[BadgeOut] = []
        for b in BADGES:
            if b.check(stats) and self.db.award_badge(astronaut_id, b.id):
                new_badges.append(self._badge_out(b, self.db.earned_badges(astronaut_id)))
        return CompletionResult(task=TaskOut(**self.db.get_task(task_id)), points_awarded=points,
                                leveled_up=stats["level"] > before_level, new_badges=new_badges,
                                state=self.state(astronaut_id))