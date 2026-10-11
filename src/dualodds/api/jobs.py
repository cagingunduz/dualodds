"""Arka plan işleri: aynı anda tek bir veri işi çalışır (maliyet ve dosya güvenliği)."""

from __future__ import annotations

import threading
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

MAX_JOBS = 50
MAX_LOG_LINES = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"  # queued | running | done | failed
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None

    def to_dict(self, include_log: bool = True) -> dict:
        d = asdict(self)
        if not include_log:
            d.pop("log")
        return d


class JobBusy(RuntimeError):
    pass


class JobManager:
    def __init__(self):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._running: str | None = None

    def submit(self, kind: str, fn: Callable[[Callable[[str], None]], dict], params: dict | None = None) -> Job:
        """`fn(log)` arka planda çalışır. Başka iş çalışıyorsa JobBusy fırlatır."""
        with self._lock:
            if self._running is not None:
                raise JobBusy(f"Şu an çalışan bir iş var: {self._running}")
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params or {})
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.popitem(last=False)
            self._running = job.id

        def log(line: str) -> None:
            stamped = f"{datetime.now(timezone.utc):%H:%M:%S} {line}"
            print(f"[{job.kind}:{job.id}] {line}", flush=True)  # Railway log ekranı
            job.log.append(stamped)
            del job.log[:-MAX_LOG_LINES]

        def run() -> None:
            job.status, job.started_at = "running", _now()
            try:
                job.result = fn(log)
                job.status = "done"
            except Exception as e:
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
                log(traceback.format_exc())
            finally:
                job.finished_at = _now()
                with self._lock:
                    self._running = None

        threading.Thread(target=run, name=f"job-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(reversed(self._jobs.values()))
