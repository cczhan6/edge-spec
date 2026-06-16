from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    REQUEST_ARRIVE = "REQUEST_ARRIVE"
    TARGET_ONLY_ARRIVE_EDGE = "TARGET_ONLY_ARRIVE_EDGE"
    SERVER_ONLY_ARRIVE_EDGE = "SERVER_ONLY_ARRIVE_EDGE"
    SERVER_ONLY_DRAFT_DONE = "SERVER_ONLY_DRAFT_DONE"
    SERVER_ONLY_VERIFY_DONE = "SERVER_ONLY_VERIFY_DONE"
    DRAFT_DONE = "DRAFT_DONE"
    PROACTIVE_DRAFT_DONE = "PROACTIVE_DRAFT_DONE"
    PACKET_ARRIVE_EDGE = "PACKET_ARRIVE_EDGE"
    VERIFY_DONE = "VERIFY_DONE"
    BATCH_FLUSH = "BATCH_FLUSH"
    BATCH_TIMEOUT = "BATCH_TIMEOUT"
    BATCH_VERIFY_DONE = "BATCH_VERIFY_DONE"
    RESULT_ARRIVE_DEVICE = "RESULT_ARRIVE_DEVICE"
    REQUEST_FINISH = "REQUEST_FINISH"


@dataclass(order=True)
class Event:
    time_ms: float
    event_id: int
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False, default=None)
