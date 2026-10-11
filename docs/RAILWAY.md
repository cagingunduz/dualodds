# Railway kurulumu

DualOdds Railway'de **tek bir servis** olarak çalışır:

- **HTTP API** (FastAPI): tahmin, video analizleri, takım profilleri, analist raporu
- **Zamanlayıcı** (servisin içinde): haftada iki kez veriyi yeniler ve yeni maçların özetlerini izletir
- **Volume** (`/data`): maç verisi ve video analizleri kalıcı olarak burada durur

Tek servis olmasının sebebi şu: Railway'de bir volume yalnızca bir servise bağlanabiliyor ve volume'lu servis tek kopya (replica) çalışmak zorunda.

## 1. Projeyi oluştur

1. https://railway.com adresine GitHub hesabınla gir.
2. **New Project → Deploy from GitHub repo**'ya tıkla ve `cagingunduz/dualodds`'u seç. Repo listede yoksa **Configure GitHub App** ile Railway'e bu repoya erişim ver.
3. Railway kökteki `Dockerfile`'ı otomatik bulur. Ek bir build ayarına gerek yok.
4. Servisin **Settings → Source** bölümünde dalın `claude/awesome-hamilton-c98exl` olduğundan emin ol. Bu dala her push otomatik olarak yeniden yayına alınır.

## 2. Volume ekle

Proje ekranında servise sağ tıkla (veya ⌘K → "Volume") → **Add Volume** → mount path: `/data`

Volume olmazsa her yeniden yayında indirilen veriler ve video analizleri silinir.
Video analizleri Gemini'ye para ödenerek üretildiği için kaybolmamalı.

## 3. Değişkenleri gir (Variables sekmesi)

| Değişken | Değer | Not |
|---|---|---|
| `GEMINI_API_KEY` | Gemini anahtarın | Video izleme |
| `YOUTUBE_API_KEY` | YouTube Data API anahtarın | Özet videosu arama |
| `ANTHROPIC_API_KEY` | Claude anahtarın | Analist raporu (`/analyze`) |
| `SPORTMONKS_API_TOKEN` | Sportmonks token'ı | Maç istatistikleri, kadro, sakatlık, xG (entegrasyon sırada) |
| `ODDS_API_KEY` | The Odds API anahtarı | Oran hareketleri (entegrasyon sırada) |
| `ADMIN_TOKEN` | Uzun, rastgele bir parola | API'yi korur. Örnek üretim: `openssl rand -hex 32` |
| `SCHEDULER_ENABLED` | `true` | Otomatik haftalık tarama |
| `SCAN_LIMIT` | `20` (isteğe bağlı) | Tek taramada en fazla kaç yeni maç özeti (maliyet sınırı) |
| `TACTICAL_LIMIT` | `10` (isteğe bağlı) | Tek taramada en fazla kaç yeni taktik video |
| `SCHEDULE_CRON` | `0 6 * * 2,5` (isteğe bağlı) | Salı ve Cuma 06:00 (İstanbul) |
| `DUALODDS_GEMINI_MODEL` | (isteğe bağlı) | Varsayılan `gemini-3.8-flash` |

`DUALODDS_DATA_DIR=/data` Dockerfile'da zaten tanımlı.

**ADMIN_TOKEN tanımlı değilse** `/health` dışındaki bütün uç noktalar kapalıdır.
Bu bilinçli bir tercih; maliyetli işlerin yanlışlıkla herkese açılmasını önler.

## 4. Alan adı ve sağlık kontrolü

- **Settings → Networking → Generate Domain**: `https://<isim>.up.railway.app` adresi oluşur.
- **Settings → Deploy → Healthcheck Path**: `/health`
- Replica sayısı 1 kalmalı (volume nedeniyle).

## 5. İlk çalıştırma

Tarayıcıda `https://<alan-adın>/docs` adresini aç. Sağ üstteki **Authorize**'a
tıklayıp `ADMIN_TOKEN` değerini gir. Ardından sırayla:

1. `POST /jobs/refresh-data`: Premier League'in bu ve geçen sezonunu indirir.
2. `GET /status`: maç sayısını ve anahtarların görünüp görünmediğini gösterir (yalnızca var/yok, değerleri göstermez).
3. `POST /jobs/video-scan` ile ilk denemeyi küçük tut:
   ```json
   {"since": "2026-09-13", "limit": 3}
   ```
4. `GET /jobs/{job_id}`: işin ilerlemesi, logu ve sonucu.
5. `GET /video/matches`, `GET /video/profile/Arsenal`
6. `POST /analyze`:
   ```json
   {"home": "Arsenal", "away": "Chelsea", "odds": [2.1, 3.4, 3.6]}
   ```

Aynı işlemler komut satırından:

```bash
URL=https://<alan-adın>
T="Authorization: Bearer $ADMIN_TOKEN"
curl -X POST $URL/jobs/refresh-data -H "$T"
curl -X POST $URL/jobs/video-scan -H "$T" -H 'content-type: application/json' -d '{"since":"2026-09-13","limit":3}'
curl $URL/jobs -H "$T"
```

İş logları Railway'in **Deploy Logs** ekranında da `[video-scan:<id>] ...` satırlarıyla görünür.

## Uç noktalar

| Yöntem | Yol | Açıklama |
|---|---|---|
| GET | `/health` | Açık sağlık kontrolü |
| GET | `/status` | Veri, zamanlayıcı ve anahtar durumu |
| POST | `/jobs/refresh-data` | Maç verisini yenile |
| POST | `/jobs/video-scan` | Özetleri bul ve izlet (`since`, `until`, `team`, `limit`, `fps`) |
| POST | `/jobs/tactical-scan` | Taktik kanallarının yeni videolarını analiz et (`since`, `limit`) |
| GET | `/video/channels` | İzlenen kanallar ve çözülen kanal ID'leri |
| GET | `/video/tactical/{team}` | Takımın taktik video notları |
| GET | `/video/matchup?home=&away=` | İki takımın taktik eşleşmesi |
| POST | `/jobs/weekly` | Zamanlanmış işi elle tetikle (yenile + son `SCAN_DAYS` günün özetleri ve taktik videoları) |
| GET | `/jobs`, `/jobs/{id}` | İş listesi ve detayı |
| GET | `/matches?team=` | Maçlar ve oranlar |
| GET | `/teams` | Model güç tablosu |
| GET | `/predict?home=&away=` | Model olasılıkları |
| GET | `/video/matches?team=` | Analiz edilmiş videolar |
| GET | `/video/match?date=&home=&away=` | Tek maçın tam video analizi |
| GET | `/video/profile/{team}` | Takımın video izci profili |
| POST | `/analyze` | Claude analist raporu (video profili dahil) |

Aynı anda tek bir veri işi çalışır. İkinci bir iş istersen `409` döner.

## Maliyet kontrolü

- YouTube'da arama yapılmaz; kanalların yükleme listeleri gezilir (50 video için 1 birim). Kanal ID'si handle ile bulunamazsa bir kez arama yapılır (100 birim) ve sonuç önbelleğe yazılır.
- Her yeni maç özeti ve her yeni taktik video için bir Gemini analizi yapılır.
- Daha önce analiz edilen maç tekrar analiz edilmez.
- `SCAN_LIMIT` tek bir taramadaki üst sınırdır.
- `fps` parametresini artırmak hızlı aksiyonları daha iyi yakalar, ama Gemini maliyetini aynı oranda artırır.
