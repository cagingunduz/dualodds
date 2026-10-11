# DualOdds

Çok kaynaklı futbol verisini tek bir omurgada birleştiren, olasılık modelleri
ve yapay zeka ajanlarıyla maç analizi üreten abonelik platformu.

Mimari, veri kaynakları, model yığını, ajan tasarımı, hukuki notlar ve yol
haritası: **[docs/MIMARI.md](docs/MIMARI.md)**

## Kurulum

```bash
pip install -e ".[dev]"
pytest
```

## Hızlı başlangıç

```bash
# Sentetik ligde uçtan uca backtest (internet gerekmez)
dualodds demo

# Süper Lig 2024/25 sonuç + oranlarını indir (football-data.co.uk)
dualodds fetch --league T1 --season 2425 --out data/raw/T1_2425.csv

# Walk-forward backtest: model vs. Pinnacle kapanışı, CLV, ROI
dualodds backtest --csv data/raw/T1_2425.csv --start 2025-01-01

# Model olasılıkları
dualodds predict --csv data/raw/T1_2425.csv --home Galatasaray --away Fenerbahce

# Claude analist ajanıyla rapor (ANTHROPIC_API_KEY gerekir)
dualodds analyze --csv data/raw/T1_2425.csv --home Galatasaray --away Fenerbahce \
  --odds 2.10 3.50 3.30 --notes "Deplasman takımında iki stoper sakat"
```

## Haftalık analiz (ana akış)

Ücretli veri sağlayıcısı yok. Önümüzdeki maçlar ve oranlar The Odds API'den gelir.
Gemini, iki takımın bu sezonki videolarını izler: Sky Sports PL maç özetleri ile
Tifo, Football Made Simple, Football Meta, The Coaches' Voice ve Adam Clery'nin
taktik videoları. Her maç için analizi Gemini yazar. Oranlar süzgeç görevi görür:
yalnızca piyasadan ayrışan ve en az 2 farklı videoya dayanan görüşler sinyal
olarak kalır.

```bash
export GEMINI_API_KEY=... YOUTUBE_API_KEY=... ODDS_API_KEY=...
dualodds week --days 7 --max-videos 40
# Raporlar: data/reports/week_<tarih>/INDEX.md ve maç başına .md/.json
```

Her video bir kez izlenir ve sonucu saklanır. Sonraki haftalarda yalnızca yeni videolar izlenir.

## Video ajanı (Premier League maç özetleri)

Gemini, herkese açık YouTube özetlerini indirmeden izler. Her gol ve
pozisyonun nasıl doğduğunu, hangi koridordan geldiğini ve savunmanın neyi
yanlış yaptığını yapılandırılmış olarak çıkarır. Claude analisti bu
gözlemleri takım profili olarak kullanır.

```bash
pip install -e ".[dev]"
export GEMINI_API_KEY=...        # https://aistudio.google.com
export YOUTUBE_API_KEY=...       # Google Cloud → YouTube Data API v3
export ANTHROPIC_API_KEY=...

dualodds fetch --league E0 --season 2627 --out data/raw/E0_2627.csv

# Özet videosu ara / tek video izlet
dualodds video find  --home Arsenal --away Chelsea --date 2026-09-20
dualodds video watch --url "https://www.youtube.com/watch?v=..." --home Arsenal --away Chelsea --date 2026-09-20 --score 2-1

# Bir tarih aralığındaki tüm maçların özetlerini bul + izlet + kaydet
dualodds video scan --csv data/raw/E0_2627.csv --since 2026-09-01 --limit 10

# Takımın birleşik video profili
dualodds video profile --team Arsenal

# Taktik analiz kanalları (Tifo, Football Made Simple, Football Meta, Coaches' Voice, Adam Clery)
dualodds video channels
dualodds video tactical-scan --since 2026-09-01 --limit 5
dualodds video tactics --team Arsenal
dualodds video matchup --home Arsenal --away Chelsea

# Video profilini de kullanan analist raporu
dualodds analyze --csv data/raw/E0_2627.csv --home Arsenal --away Chelsea --video-dir data/lake/video
```

## Sunucu (Railway)

API + zamanlayıcı + kalıcı volume tek servis olarak çalışır. Kurulum adımları:
**[docs/RAILWAY.md](docs/RAILWAY.md)**

Yerelde çalıştırmak için:

```bash
ADMIN_TOKEN=dev DUALODDS_DATA_DIR=data uvicorn dualodds.api.app:app --reload
# http://localhost:8000/docs
```

## İlkeler

1. **Sayılar modelden, yorum ajandan.** LLM olasılık uydurmaz.
2. **Sızıntı yok.** Her model yalnızca maçtan önce bilinen veriyle eğitilir.
3. **Rakip piyasa.** Başarı, kapanış oranlarına karşı log-loss/kalibrasyon ve CLV ile ölçülür.
4. **Lisanslı veri.** Her kaynak lisans metadatasıyla kaydedilir.
