"""DualOdds HTTP API + zamanlayıcı (Railway'de tek servis olarak çalışır).

Ortam değişkenleri: bkz. docs/RAILWAY.md. `/health` dışındaki her uç nokta
`Authorization: Bearer <ADMIN_TOKEN>` ister. ADMIN_TOKEN tanımlı değilse
korumalı uç noktalar kapalıdır (yanlışlıkla herkese açık maliyetli iş olmasın).

Etkileşimli dokümantasyon: /docs
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from dualodds import __version__
from dualodds.api.jobs import JobBusy, JobManager
from dualodds.config import Settings
from dualodds.models import DixonColesModel
from dualodds.pipeline import load_matches, refresh_matches, video_scan, weekly_job
from dualodds.video.profile import team_profile
from dualodds.video.store import VideoStore

_bearer = HTTPBearer(auto_error=False)


class VideoScanRequest(BaseModel):
    since: date | None = None  # Varsayılan: SCAN_DAYS gün önce
    until: date | None = None
    team: str | None = None
    limit: int | None = None  # Varsayılan: SCAN_LIMIT
    fps: float | None = None


class AnalyzeRequest(BaseModel):
    home: str
    away: str
    odds: tuple[float, float, float] | None = None
    notes: str | None = None
    use_video: bool = True


def create_app(settings: Settings | None = None, finder=None, watcher=None, source=None, analyst_client=None) -> FastAPI:
    """`finder`, `watcher`, `source`, `analyst_client` testlerde sahte nesne vermek içindir."""
    settings = settings or Settings.from_env()
    jobs = JobManager()
    store = VideoStore(settings.video_dir)
    model_cache: dict = {}

    def submit_weekly(params: dict | None = None):
        return jobs.submit("weekly", lambda log: weekly_job(settings, log=log, finder=finder, watcher=watcher, source=source), params)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        scheduler = None
        if settings.scheduler_enabled:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            def scheduled():
                try:
                    submit_weekly({"trigger": "schedule"})
                except JobBusy as e:
                    print(f"Zamanlanmış iş atlandı: {e}", flush=True)

            scheduler = BackgroundScheduler(timezone=settings.timezone)
            scheduler.add_job(scheduled, CronTrigger.from_crontab(settings.schedule_cron, timezone=settings.timezone))
            scheduler.start()
            print(f"Zamanlayıcı açık: '{settings.schedule_cron}' ({settings.timezone})", flush=True)
        yield
        if scheduler is not None:
            scheduler.shutdown(wait=False)

    app = FastAPI(title="DualOdds", version=__version__, lifespan=lifespan)

    def require_admin(creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]) -> None:
        if not settings.admin_token:
            raise HTTPException(503, "ADMIN_TOKEN tanımlı değil; korumalı uç noktalar kapalı")
        if creds is None or not hmac.compare_digest(creds.credentials, settings.admin_token):
            raise HTTPException(401, "Geçersiz veya eksik token", headers={"WWW-Authenticate": "Bearer"})

    Admin = Depends(require_admin)

    def matches_or_404():
        try:
            return load_matches(settings)
        except RuntimeError as e:
            raise HTTPException(404, str(e)) from None

    def fitted_model():
        df = matches_or_404()
        key = (len(df), str(df["date"].max()), int(df["home_goals"].notna().sum()))
        if model_cache.get("key") != key:
            model_cache.update(key=key, model=DixonColesModel().fit(df), matches=df)
        return model_cache["model"], model_cache["matches"]

    def submit(kind: str, fn, params: dict):
        try:
            job = jobs.submit(kind, fn, params)
        except JobBusy as e:
            raise HTTPException(409, str(e)) from None
        return {"job_id": job.id, "status_url": f"/jobs/{job.id}"}

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/status", dependencies=[Admin])
    def status():
        try:
            n_matches = len(load_matches(settings))
        except RuntimeError:
            n_matches = 0
        return {
            "league": settings.league,
            "data_dir": str(settings.data_dir),
            "matches": n_matches,
            "video_analyses": len(store.all()),
            "scheduler": {"enabled": settings.scheduler_enabled, "cron": settings.schedule_cron, "tz": settings.timezone},
            "keys": {k: bool(os.environ.get(k)) for k in ("GEMINI_API_KEY", "YOUTUBE_API_KEY", "ANTHROPIC_API_KEY")},
        }

    @app.post("/jobs/refresh-data", dependencies=[Admin])
    def job_refresh():
        return submit("refresh-data", lambda log: {"matches": len(refresh_matches(settings, source=source, log=log))}, {})

    @app.post("/jobs/video-scan", dependencies=[Admin])
    def job_video_scan(req: VideoScanRequest):
        since = req.since or date.today() - timedelta(days=settings.scan_days)
        limit = settings.scan_limit if req.limit is None else req.limit

        def run(log):
            return video_scan(
                load_matches(settings), store, since=since, until=req.until, team=req.team,
                limit=limit, fps=req.fps, finder=finder, watcher=watcher, log=log,
            )

        return submit("video-scan", run, {**req.model_dump(mode="json"), "since": str(since), "limit": limit})

    @app.post("/jobs/weekly", dependencies=[Admin])
    def job_weekly():
        try:
            job = submit_weekly({"trigger": "manual"})
        except JobBusy as e:
            raise HTTPException(409, str(e)) from None
        return {"job_id": job.id, "status_url": f"/jobs/{job.id}"}

    @app.get("/jobs", dependencies=[Admin])
    def list_jobs():
        return [j.to_dict(include_log=False) for j in jobs.list()]

    @app.get("/jobs/{job_id}", dependencies=[Admin])
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "İş bulunamadı")
        return job.to_dict()

    @app.get("/matches", dependencies=[Admin])
    def matches(team: str | None = None, limit: int = Query(20, le=500)):
        df = matches_or_404()
        if team:
            df = df[(df["home"] == team) | (df["away"] == team)]
        out = df.tail(limit).copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out.astype(object).where(out.notna(), None).to_dict(orient="records")

    @app.get("/teams", dependencies=[Admin])
    def teams():
        model, _ = fitted_model()
        r = model.ratings().round(3)
        return [{"team": t, **row} for t, row in r.to_dict(orient="index").items()]

    @app.get("/predict", dependencies=[Admin])
    def predict(home: str, away: str):
        model, _ = fitted_model()
        try:
            return {"model_as_of": str(model.fitted_as_of.date()), **model.predict(home, away).to_dict()}
        except KeyError as e:
            raise HTTPException(404, str(e)) from None

    @app.get("/video/matches", dependencies=[Admin])
    def video_matches(team: str | None = None):
        records = store.all()
        if team:
            records = [r for r in records if team in (r.home, r.away)]
        return [
            {
                "date": r.match_date,
                "home": r.home,
                "away": r.away,
                "verified": r.analysis.video_matches_fixture,
                "observed_score": r.analysis.observed_score,
                "events": len(r.analysis.events),
                "video_url": r.video_url,
                "channel": r.channel,
            }
            for r in records
        ]

    @app.get("/video/match", dependencies=[Admin])
    def video_match(date: str, home: str, away: str):
        path = store.path_for(date, home, away)
        if not path.exists():
            raise HTTPException(404, "Bu maç için video analizi yok")
        from dualodds.video.schema import StoredVideoAnalysis

        return StoredVideoAnalysis.model_validate_json(path.read_text(encoding="utf-8"))

    @app.get("/video/profile/{team}", dependencies=[Admin])
    def video_profile(team: str, last_n: int = Query(5, ge=1, le=38)):
        return team_profile(team, store.for_team(team, last_n=last_n))

    @app.post("/analyze", dependencies=[Admin])
    def analyze(req: AnalyzeRequest):
        from dualodds.agents.analyst import AnalystError, analyze_match
        from dualodds.agents.tools import AnalysisContext

        model, df = fitted_model()
        ctx = AnalysisContext(matches=df, model=model, video_store=store if req.use_video else None)
        try:
            report = analyze_match(ctx, req.home, req.away, odds=req.odds, notes=req.notes, client=analyst_client)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except AnalystError as e:
            raise HTTPException(502, str(e)) from None
        return {"report": report.model_dump(), "markdown": report.to_markdown()}

    return app


def __getattr__(name: str):
    # `uvicorn dualodds.api.app:app` için tembel oluşturma (import sırasında ortam okunmasın)
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
