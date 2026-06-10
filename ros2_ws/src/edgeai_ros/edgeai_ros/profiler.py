from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


class CSVProfiler:
    """Simple robust CSV logger that supports dynamic row dictionaries."""

    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: List[str] = list(fieldnames)
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames, extrasaction="ignore")
        self.writer.writeheader()

    def log(self, row: Dict):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self.writer.writerow(clean)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
