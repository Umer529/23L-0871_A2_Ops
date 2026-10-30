"""Task 5 - database trail. One row per prediction: what was decided, when,
by which model, and how confidently. SQLAlchemy Core keeps the same code working
on PostgreSQL (Compose / cloud) and SQLite (tests, quick local runs)."""
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table, create_engine,
                        select, text)

log = logging.getLogger(__name__)
metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("filename", String(255), nullable=False),
    Column("image_sha256", String(64), nullable=False),
    Column("label", String(20), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("p_ai", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_name", String(100), nullable=False),
    Column("model_version", String(50), nullable=False),
    Column("inference_mode", String(10), nullable=False),
    Column("latency_ms", Float, nullable=False),
)


class PredictionStore:
    def __init__(self, url, retries=10, delay=2.0):
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        for attempt in range(1, retries + 1):
            try:
                metadata.create_all(self.engine)
                break
            except Exception as e:
                # the readable cause (wrong password, unknown host...) ends up in `docker compose logs web`
                log.error("database not reachable (attempt %d/%d): %s", attempt, retries,
                          str(e).splitlines()[0])
                if attempt == retries:
                    raise
                time.sleep(delay)
        log.info("database ready: %s", self.engine.url.render_as_string(hide_password=True))

    def add(self, **row):
        row.setdefault("created_at", datetime.now(timezone.utc))
        with self.engine.begin() as conn:
            return conn.execute(predictions.insert().values(**row)).inserted_primary_key[0]

    def recent(self, limit=100):
        q = select(predictions).order_by(predictions.c.id.desc()).limit(limit)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(q)]

    def count(self):
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one()

    def ping(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
