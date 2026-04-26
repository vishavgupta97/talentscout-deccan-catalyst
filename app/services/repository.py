from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings
from app.models import Candidate


def _sqlite_path(database_url: str) -> Path:
    """Extract a filesystem path from the configured sqlite URL."""

    parsed = urlparse(database_url)
    if parsed.scheme != "sqlite":
        raise ValueError("This demo build only supports sqlite database URLs.")
    path = parsed.path or ""
    return Path(path)


class CandidateRepository:
    """Thin repository around the local sourced candidate pool.

    For the hackathon build we keep storage deliberately boring: a small sqlite
    file seeded from JSON. That gives us deterministic demo data, fast startup,
    and a clean upgrade path to Postgres later.
    """

    def __init__(self, database_url: str | None = None, dataset_path: str | None = None):
        self.database_url = database_url or settings.database_url
        self.dataset_path = Path(dataset_path or settings.dataset_path)
        self.db_path = _sqlite_path(self.database_url)

    def initialize(self) -> None:
        """Create the local table and seed it on first run."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            count = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            if count == 0:
                # Seed once from the checked-in dataset so demos stay stable.
                rows = json.loads(self.dataset_path.read_text(encoding="utf-8"))
                conn.executemany(
                    "INSERT INTO candidates (candidate_id, payload) VALUES (?, ?)",
                    [(row["candidate_id"], json.dumps(row)) for row in rows],
                )
            conn.commit()

    def list_candidates(self) -> list[Candidate]:
        """Return all candidate profiles as validated Pydantic models."""

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT payload FROM candidates").fetchall()
        return [Candidate.model_validate_json(payload) for (payload,) in rows]
