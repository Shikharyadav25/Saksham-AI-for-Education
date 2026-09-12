import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class MemoryState:
    skill_id: str
    stability: float       # Half-life stability in days
    last_practiced_at: datetime
    retrievability: float  # Current predicted retention probability R(t) in [0, 1]
    repetition_count: int


class MemoryModel:
    """
    Implements a two-component memory model (Bjork / Ebbinghaus / Leitner style)
    tracking retrievability decay R(t) = exp(-delta_t / S).
    """

    def __init__(self, default_initial_stability: float = 2.0):
        self.default_stability = default_initial_stability
        self.memory_records: Dict[str, MemoryState] = {}

    def get_or_create(self, skill_id: str, now: Optional[datetime] = None) -> MemoryState:
        current_time = now or datetime.now(timezone.utc)
        if skill_id not in self.memory_records:
            self.memory_records[skill_id] = MemoryState(
                skill_id=skill_id,
                stability=self.default_stability,
                last_practiced_at=current_time,
                retrievability=1.0,
                repetition_count=0,
            )
        return self.memory_records[skill_id]

    def update(
        self,
        skill_id: str,
        is_correct: bool,
        timestamp: Optional[datetime] = None,
    ) -> MemoryState:
        now = timestamp or datetime.now(timezone.utc)
        record = self.get_or_create(skill_id, now)

        delta_days = max(0.0, (now - record.last_practiced_at).total_seconds() / 86400.0)
        # Update current retrievability prior to answer
        current_r = math.exp(-delta_days / max(0.1, record.stability))

        if is_correct:
            # Memory strengthens more when retrieved at lower retrievability (desirable difficulty)
            difficulty_bonus = 1.0 + (1.0 - current_r) * 1.5
            new_stability = record.stability * (1.8 * difficulty_bonus)
            record.repetition_count += 1
            record.retrievability = 1.0
        else:
            # Memory lapses drop stability
            new_stability = max(0.5, record.stability * 0.5)
            record.retrievability = 0.4

        record.stability = min(120.0, new_stability)
        record.last_practiced_at = now
        return record

    def compute_retrievability(self, skill_id: str, now: Optional[datetime] = None) -> float:
        if skill_id not in self.memory_records:
            return 1.0
        record = self.memory_records[skill_id]
        current_time = now or datetime.now(timezone.utc)
        delta_days = max(0.0, (current_time - record.last_practiced_at).total_seconds() / 86400.0)
        r = math.exp(-delta_days / max(0.1, record.stability))
        record.retrievability = float(min(1.0, max(0.0, r)))
        return record.retrievability

    def get_review_urgency(self, skill_id: str, target_retention: float = 0.85) -> float:
        """Returns urgency score between 0.0 (no review needed) and 1.0 (critically decayed)."""
        r = self.compute_retrievability(skill_id)
        if r >= target_retention:
            return 0.0
        return float((target_retention - r) / target_retention)
