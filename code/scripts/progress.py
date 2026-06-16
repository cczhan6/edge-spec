from __future__ import annotations

import shutil
import sys
import time
from typing import TextIO


class ProgressReporter:
    def __init__(
        self,
        total: int,
        label: str,
        stream: TextIO | None = None,
        unit: str = "it",
        label_width: int | None = None,
    ) -> None:
        self.total = max(1, total)
        self.label = label
        self.stream = stream or sys.stderr
        self.unit = unit
        self.label_width = max(label_width or 0, len(label))
        self.completed = 0
        self.width = 40
        self._started_at = time.perf_counter()
        self._last_line_len = 0
        self._line_active = False

    def start(self, item: str = "") -> float:
        self._write(self.completed, item)
        return time.perf_counter()

    def update(self, completed: int, item: str = "") -> None:
        self.completed = min(max(0, completed), self.total)
        self._write(self.completed, item)

    def done(self, item: str, start_time: float) -> None:
        self.completed = min(self.completed + 1, self.total)
        elapsed_s = time.perf_counter() - start_time
        self._write(self.completed, f"done: {item} {elapsed_s:.1f}s")

    def _write(self, completed: int, item: str) -> None:
        percent = int(100 * completed / self.total)
        elapsed_s = time.perf_counter() - self._started_at
        if completed:
            rate = completed / elapsed_s if elapsed_s else 0.0
            remaining_s = (self.total - completed) / rate if rate else 0.0
            timing = f"{_format_duration(elapsed_s)}<{_format_duration(remaining_s)}, {rate:.2f}{self.unit}/s"
        else:
            timing = f"{_format_duration(elapsed_s)}<?, ?{self.unit}/s"
        label = self.label.ljust(self.label_width)
        suffix = f" {item}" if item else ""
        line_without_bar = f"{label}: {percent:3d}%|| {completed}/{self.total} [{timing}]{suffix}"
        width = _bar_width(self.width, len(line_without_bar))
        filled = int(width * completed / self.total)
        bar = "█" * filled + " " * (width - filled)
        line = f"{label}: {percent:3d}%|{bar}| {completed}/{self.total} [{timing}]{suffix}"
        padding = " " * max(0, self._last_line_len - len(line))
        prefix = "\r" if self._line_active else ""
        self.stream.write(f"{prefix}{line}{padding}")
        self.stream.flush()
        self._last_line_len = len(line)
        self._line_active = True

    def finish_line(self) -> None:
        if self._line_active:
            self.stream.write("\n")
            self.stream.flush()
            self._line_active = False
            self._last_line_len = 0


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _bar_width(default_width: int, line_without_bar_len: int) -> int:
    columns = shutil.get_terminal_size(fallback=(100, 24)).columns
    available = columns - line_without_bar_len - 1
    return max(1, min(default_width, available))
