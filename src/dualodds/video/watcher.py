"""Maç özeti videosunu izleyen ajan (Google Gemini).

Gemini, herkese açık YouTube videolarını URL ile doğrudan işleyebilir; videoyu
indirmemize gerek kalmaz. Çıktı, `MatchVideoAnalysis` şemasına uyan
yapılandırılmış JSON'dur.

Ortam değişkenleri:
    GEMINI_API_KEY          Gemini API anahtarı
    DUALODDS_GEMINI_MODEL   Model adı (varsayılan: DEFAULT_MODEL)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from dualodds.data import entities
from dualodds.video.schema import MatchVideoAnalysis, StoredVideoAnalysis

DEFAULT_MODEL = "gemini-3.8-flash"

PROMPT = """Sen elit bir futbol video analistisin (opposition scout). Bu video bir Premier League maç özeti.

Beklenen maç: {home} (ev sahibi) – {away} (deplasman), tarih {match_date}.{score_hint}

Görevin:
1. Önce videonun gerçekten bu maç olup olmadığını doğrula (takım formaları, skor tabelası, spiker). Değilse video_matches_fixture=false yap ve diğer alanları minimumda tut.
2. Videodaki HER gol ve önemli pozisyonu tek tek kaydet: videodaki zaman, tabeladaki dakika, hangi takım, pozisyonun nasıl doğduğu (origin), hangi koridordan geldiği ve savunmanın neyi yanlış yaptığı.
3. Her takım için pres yüksekliği, oyun kurma tarzı, ağırlıklı hücum koridorları, güçlü ve zayıf yönleri yalnızca videoda GÖRDÜĞÜN kanıtlara dayanarak yaz.

Kurallar:
- Görmediğin şeyi uydurma. Emin olmadığın oyuncu adını yazma; diziliş net değilse null bırak; pres/oyun kurma anlaşılmıyorsa "unclear" seç.
- Özet videolar maçın küçük ve gol anlarına yanlı bir kesitini gösterir. Bunu limitations alanında somut olarak belirt.
- Serbest metin alanlarını Türkçe yaz."""


class VideoAnalysisError(RuntimeError):
    pass


RETRY_DELAYS = (10, 30, 90)  # saniye; Gemini "high demand" (503) genelde kısa sürer


def is_transient(exc: Exception) -> bool:
    """Tekrar denenmeye değer geçici hata mı (aşırı yük, kota, bağlantı kopması)?"""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    name = type(exc).__name__
    return name in ("RemoteProtocolError", "ReadTimeout", "ConnectError", "ConnectTimeout", "ReadError")


def with_retries(fn, delays=RETRY_DELAYS, sleep=time.sleep, log=None):
    for attempt, delay in enumerate((*delays, None)):
        try:
            return fn()
        except Exception as e:
            if delay is None or not is_transient(e):
                raise
            if log:
                log(f"Geçici hata ({type(e).__name__}), {delay} sn sonra tekrar ({attempt + 1}/{len(delays)})")
            sleep(delay)


def gemini_video_json(
    video_url: str, prompt: str, schema, fps: float | None = None, model: str | None = None, client=None, sleep=time.sleep
):
    """Gemini'ye herkese açık bir YouTube videosunu izletir, `schema` (pydantic) tipinde döndürür.

    Geçici hatalarda (503 aşırı yük vb.) artan beklemelerle tekrar dener.
    """
    from google import genai
    from google.genai import types

    model = model or os.environ.get("DUALODDS_GEMINI_MODEL", DEFAULT_MODEL)
    client = client or genai.Client()
    video_part = types.Part(
        file_data=types.FileData(file_uri=video_url),
        # Hızlı aksiyonlar için örnekleme hızı artırılabilir (token maliyeti orantılı artar)
        video_metadata=types.VideoMetadata(fps=fps) if fps else None,
    )
    response = with_retries(
        lambda: client.models.generate_content(
            model=model,
            contents=types.Content(role="user", parts=[video_part, types.Part(text=prompt)]),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            ),
        ),
        sleep=sleep,
    )
    parsed = response.parsed
    if isinstance(parsed, schema):
        return parsed
    if not response.text:
        raise VideoAnalysisError(f"Gemini boş yanıt döndürdü: {video_url}")
    return schema.model_validate_json(response.text)


def watch_highlights(
    video_url: str,
    home: str,
    away: str,
    match_date: str,
    score: str | None = None,
    video_title: str = "",
    channel: str = "",
    fps: float | None = None,
    model: str | None = None,
    client=None,
) -> StoredVideoAnalysis:
    """Videoyu Gemini'ye izletir ve saklanmaya hazır kayıt döndürür."""
    model = model or os.environ.get("DUALODDS_GEMINI_MODEL", DEFAULT_MODEL)
    prompt = PROMPT.format(
        home=entities.team(home).name,
        away=entities.team(away).name,
        match_date=match_date,
        score_hint=f" Resmî sonuç: {score}." if score else "",
    )
    analysis = gemini_video_json(video_url, prompt, MatchVideoAnalysis, fps=fps, model=model, client=client)

    return StoredVideoAnalysis(
        match_date=str(match_date)[:10],
        home=home,
        away=away,
        video_url=video_url,
        video_title=video_title,
        channel=channel,
        analyzer_model=model,
        analyzed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        analysis=analysis,
    )
