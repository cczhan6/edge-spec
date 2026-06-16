from __future__ import annotations

import heapq
import unittest

from src.events import Event, EventType


class EventOrderTest(unittest.TestCase):
    def test_event_id_breaks_time_ties(self) -> None:
        queue = [
            Event(5.0, 2, EventType.REQUEST_ARRIVE),
            Event(5.0, 1, EventType.DRAFT_DONE),
            Event(4.0, 9, EventType.REQUEST_FINISH),
        ]
        heapq.heapify(queue)
        self.assertEqual([heapq.heappop(queue).event_id for _ in range(3)], [9, 1, 2])


if __name__ == "__main__":
    unittest.main()
