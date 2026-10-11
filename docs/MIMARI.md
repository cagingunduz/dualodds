# DualOdds — Mimari ve Yol Haritası

> Hedef: Futbol için "Aladdin" — dağınık onlarca veri kaynağını tek bir veri
> omurgasında birleştiren, bunları istatistiksel ve yapay zeka modelleriyle
> olasılığa çeviren, ajanlar aracılığıyla analiz üreten ve bunu aylık
> abonelikle satan bir platform.

## 1. Aladdin benzetmesini doğru okumak

BlackRock Aladdin'i güçlü kılan şey "tahmin" değil, üç şeydir:

1. **Tek veri omurgası:** Herkes aynı, temizlenmiş, zaman damgalı veriye bakar.
2. **Risk motoru:** Her pozisyonun olasılık dağılımı, korelasyonu ve senaryosu hesaplanır.
3. **İş akışına gömülü olmak:** Karar veren kişi ekranı açtığında ihtiyacı olan her şey oradadır.

DualOdds'un karşılığı:

| Aladdin | DualOdds |
|---|---|
| Piyasa + portföy verisi | Maç, olay (event), takip (tracking), oran, kadro, haber, hava durumu verisi |
| Risk modelleri | Skor dağılımı, olasılık, kalibrasyon, piyasa karşılaştırması |
| Portföy görünümü | Abonenin seçimlerinin toplam riski, korelasyonu, Kelly tabanlı kasa yönetimi |
| Analist ekipleri | Yapay zeka ajanları (veri toplama, taktik, nicel, piyasa, editör) |

**Ürünün farkı "daha çok tahmin" değil; ölçülebilir kalite + şeffaflık + risk
yönetimi** olmalı.

## 2. Gerçekçi beklenti: rakibimiz piyasa

Bahis piyasaları, özellikle keskin bahisçilerin (Pinnacle, Betfair borsası)
**kapanış oranları**, kamuya açık bilginin neredeyse tamamını zaten fiyatlar.
"Ne kadar çok veri o kadar iyi" ancak şu koşullarda doğrudur:

- Veri **maçtan önce, zamanında** bilinen bilgiyi temsil ediyorsa (sızıntı yok),
- Piyasanın henüz fiyatlamadığı bir sinyal içeriyorsa,
- Ve bu katkı **walk-forward backtest ile kanıtlanmışsa**.

### Faz 0 bulgusu (bu repodaki kodla, gerçek veri)

Süper Lig 2023/24 + 2024/25, Ekim 2024'ten itibaren walk-forward, Dixon-Coles
temel modeli:

| Metrik | Model | Pinnacle kapanışı (marjı temizlenmiş) |
|---|---|---|
| Log-loss | 0.949 | **0.911** |
| RPS | 0.195 | **0.182** |
| Model "değerli" gördüğü 223 bahis: ROI | −27% | |
| Ortalama CLV | −6.4% | |

Yorum: Yalnızca sonuçlardan beslenen temel model piyasanın gerisinde kalıyor —
bu beklenen bir durum. Platformun bütün değeri, bu farkı kapatan ve **ölçülebilir
şekilde** aşan ek veri ve modellerden gelecek. Bu tablo her yeni model/veri için
yeniden üretilecek tek doğru ölçüttür.

**Başarı ölçütleri (sırasıyla):**
1. Kalibrasyon (model %60 diyorsa, bu maçların ~%60'ı gerçekleşiyor mu?)
2. Log-loss / RPS'te kapanış piyasasına yaklaşmak veya geçmek
3. Pozitif ortalama **CLV** (kapanış oranından daha iyi oran almak) — kârlılığın en güvenilir öncü göstergesi
4. Ancak bunlardan sonra ROI (çok gürültülü, binlerce bahis gerekir)

## 3. Katmanlı mimari

```
 ┌──────────────────────────── KAYNAKLAR ────────────────────────────┐
 │ Sonuç/Oran · Event · Tracking · Kadro/Sakatlık · Haber · Hava ·   │
 │ Hakem · Takvim/Seyahat · Transfer · Maç anlatımı/metin · Video    │
 └───────────────────────────────┬────────────────────────────────────┘
                                 ▼
 [1] Toplama (Ingestion)  — kaynak başına bağlayıcı, lisans metadatası, ham arşiv
                                 ▼
 [2] Veri gölü: bronze (ham) → silver (temiz, kanonik) → gold (özellikler)
                                 ▼
 [3] Varlık eşleştirme   — takım/oyuncu/hakem/stadyum için tek kimlik
                           ("Fenerbahce" = "Fenerbahçe" = "Fenerbahce SK")
                                 ▼
 [4] Özellik deposu      — zaman noktası doğru (point-in-time) özellikler
                                 ▼
 [5] Model katmanı       — L0 istatistik · L1 xG/oyuncu · L2 ML · L3 taktik
                                 ▼
 [6] Topluluk + kalibrasyon + piyasa harmanlama
                                 ▼
 [7] Risk/portföy        — EV, Kelly, korelasyon, maruziyet
                                 ▼
 [8] Ajan katmanı        — veri, taktik, nicel, piyasa, editör ajanları
                                 ▼
 [9] Ürün                — Web, mobil, Telegram/push bildirim, API, performans sayfası
```

**En çok hafife alınan iki katman:**
- **[3] Varlık eşleştirme:** Kaynak sayısı arttıkça isim/kimlik uyuşmazlıkları
  katlanarak büyür. Erken yatırım yapılmazsa her yeni kaynak haftalarca temizlik demektir.
- **[4] Zaman noktası doğruluğu:** Her özellik "maçtan X saat önce biliniyor muydu?"
  sorusuna cevap vermeli. Sızıntılı bir özellik backtest'i harika, canlı sonucu felaket yapar.

## 4. Veri kaynakları

| Tür | Örnek kaynaklar | Ne kazandırır | Not |
|---|---|---|---|
| Sonuç + oran | football-data.co.uk (bu repoda), The Odds API, Betfair Exchange API | Temel model, backtest, CLV | Ücretsiz/ucuz; ticari kullanımı teyit edin |
| Fikstür, kadro, sakatlık, canlı | API-Football, Sportmonks | Kim oynuyor, kim yok | Uygun fiyatlı aylık planlar |
| Event (olay) verisi | Hudl StatsBomb (açık veri seti ücretsiz/sınırlı), Opta/Stats Perform, Wyscout | xG, pas ağları, pres, taktik metrikler | En büyük kalite sıçraması; lisansı pahalı |
| Tracking (konum) | SkillCorner (yayından üretilmiş), Second Spectrum | Savunma hattı yüksekliği, kompaktlık, koşu yükü | Çok pahalı; Faz 3 |
| Haber / basın toplantısı | Kulüp siteleri, RSS, yerel spor medyası | Rotasyon, motivasyon, iç sorunlar | LLM ile yapılandırılmış sinyale çevrilir |
| Hava durumu | Open-Meteo | Yağmur/rüzgar → gol beklentisi | Ücretsiz |
| Hakem | Event verisi, geçmiş maçlar | Kart/penaltı eğilimi | Kart pazarları için değerli |
| Takvim/seyahat | Fikstür + stadyum koordinatları | Yorgunluk, Avrupa kupası etkisi | Kendimiz hesaplarız |
| Video / maç özetleri | Lisanslı platformlar (ör. Wyscout/Hudl video) | Taktik kurgu analizi | **Telif! Aşağıya bakın** |

### Maç özetlerinden taktik analizi: video ajanı (uygulandı)

Claude API video girdisi kabul etmiyor. YouTube videosundan kare çıkarmak için
de videoyu indirmek gerekir, bu da YouTube kullanım şartlarına aykırıdır. Bu
yüzden "göz" olarak **Google Gemini** kullanılır. Gemini herkese açık YouTube
videolarını **URL ile, indirmeden** işleyebilir. Claude ise analist ve
orkestratör olarak kalır.

```
football-data (maç listesi, skor)
        │
        ▼
video/finder.py   YouTube Data API ile özeti bul; resmî/yayıncı kanalı,
                  başlıkta iki takım, yayın tarihi ile puanla
        │
        ▼
video/watcher.py  Gemini videoyu izler → MatchVideoAnalysis (yapılandırılmış JSON):
                  - video doğru maç mı? görülen skor
                  - her gol/pozisyon: an, dakika, takım, nasıl doğdu (kontra,
                    orta, cutback, önde baskı, duran top...), koridor,
                    savunmanın hatası, pozisyon kalitesi
                  - takım başına pres, oyun kurma, koridorlar, güçlü/zayıf yönler
        │
        ▼
video/store.py    Maç başına bir JSON (aynı maç iki kez ödenmez)
        │
        ▼
video/profile.py  Takımın son N maçında tekrar eden örüntüler
                  ("son 3 maçta 2 kez sağ bek arkasından kontra yedi")
        │
        ▼
Claude analist    get_video_scouting aracıyla profili okur, model sayılarıyla birleştirir
```

Önemli sınırlar:
- **Özetler yanlıdır:** 90 dakikanın 5–10 dakikasını, ağırlıkla gol anlarını
  gösterir. Pozisyonların nasıl doğduğu ve savunma hataları güvenilir sinyaldir.
  Pres yüksekliği veya oyun kurma tarzı ise düşük güvenlidir. Profil bu uyarıyı
  her zaman taşır.
- **Doğrulama:** Gemini videonun gerçekten o maç olup olmadığını kontrol eder.
  Uyuşmayan videolar profillere girmez.
- **Erişim:** Yalnızca herkese açık videolar işlenebilir. Premier League
  özetleri bazı bölgelerde coğrafi kısıtlı olabilir; bu yüzden NBC Sports ve
  kulüp kanalları da aday kanallar arasında.
- **Hukuk:** Video indirilmez, saklanmaz ve yeniden yayınlanmaz. Yalnızca
  türetilmiş gözlemler (metin/kod) saklanır. Yine de ticari ürün için YouTube
  ve Gemini API kullanım şartları avukatla gözden geçirilmeli. Gemini'nin
  YouTube URL özelliği şu an önizleme aşamasında ve koşulları değişebilir.
- **Ölçüm:** Video sinyalleri (ör. "kontra yeme oranı") özellik olarak modele
  girip walk-forward backtest'te katkı göstermedikçe yalnızca rapor yorumunda
  kullanılır.
- **Uzun vadede:** Event verisi (Opta/StatsBomb) aynı bilgiyi tam maç için
  verir. Video ajanı, event verisinin göremediği bağlamı (neden, nasıl)
  tamamlayıcı olarak kalır.

### Takımın oyun kurgusu: "taktik parmak izi"

Event verisinden her takım için maç maç güncellenen bir vektör:

- **Pres:** PPDA (rakibin savunma pası başına müdahale), yüksek top kazanımı
- **Oyun kurma:** Kalecinin kısa/uzun dağıtımı, geriden oyun kurma oranı, direkt oyun oranı
- **Yön:** Kanat/merkez kullanımı, sol-sağ dengesi, progresif pas/taşıma
- **Saha eğimi (field tilt):** Son üçte birdeki pas payı
- **Topa sahip olma tipi:** Hızlı hücum oranı, uzun pozisyon oranı
- **Duran top:** Duran toptan xG payı, korner rutinleri
- **Savunma:** Blok yüksekliği (tracking ile), kontra atak yeme oranı

Bu vektörler üzerinden **stil eşleşmesi modeli**: "Yüksek pres yapan takımlar,
geriden oyun kuran ama baskı altında top kaybeden takımlara karşı tarihsel olarak
ne yapıyor?" — piyasanın fiyatlamakta zorlandığı tür sinyal budur.

## 5. Model yığını

| Seviye | Model | Girdi | Durum |
|---|---|---|---|
| L0 | Dixon-Coles (zaman ağırlıklı) | Skorlar | ✅ Bu repoda |
| L0 | Elo / pi-rating | Skorlar | Sırada |
| L1 | xG tabanlı Poisson / bivariate Poisson | Şut/xG verisi | Faz 1 |
| L1 | Oyuncu bazlı kadro gücü | Muhtemel 11'ler, oyuncu katkı puanları | Faz 2 — en kritik model; sakatlık/rotasyon etkisini sayıya çevirir |
| L2 | Gradient boosting (LightGBM) | Tüm özellik deposu | Faz 1-2; hedef: sonuç olasılığı veya **piyasa artığı** |
| L3 | Taktik stil eşleşmesi | Taktik parmak izi vektörleri | Faz 2 |
| L4 | Maç içi (canlı) model | Canlı olay akışı | Faz 3 |

**Topluluk ve kalibrasyon:** Modeller yığınlama (stacking) ile birleştirilir,
izotonik/sıcaklık ölçekleme ile kalibre edilir. Piyasa olasılığı güçlü bir
**ön bilgi (prior)** olarak kullanılır: model piyasadan ancak kanıtlanmış sinyal
olduğunda sapar.

**Pazarlar:** 1X2, alt/üst, KG var/yok, Asya handikapı, doğru skor, korner/kart.
Skor matrisi tabanlı modeller tüm gol pazarlarını tutarlı biçimde tek kaynaktan
üretir (bu repoda `models/markets.py`).

## 6. Ajan mimarisi

Altın kural: **LLM olasılık uydurmaz.** Sayılar modelden gelir; ajanlar veri
toplar, nitel bilgiyi yapılandırır, yorumlar ve denetler. Nitel sinyaller
(ör. "yıldız forvet sakat") yapılandırılmış özelliğe çevrilip modele geri girer
ve **backtest edilir** — kanıtlanmayan sinyal ürüne girmez.

```
                     ┌───────────────────────┐
                     │  Orkestratör ajan     │  maç listesi, görev dağıtımı
                     └──────────┬────────────┘
     ┌──────────────┬───────────┼─────────────┬──────────────┐
     ▼              ▼           ▼             ▼              ▼
 Veri/Haber     Taktik       Nicel         Piyasa/Risk     Editör/
 ajanı          analist      ajan          ajanı           Doğrulayıcı
 (sakatlık,     (parmak      (modelleri    (oran hareketi, (rapordaki her
 rotasyon,      izi, stil    çalıştırır,   değer, Kelly,   sayının kaynağa
 basın topl.)   eşleşmesi)   senaryolar)   korelasyon)     uyduğunu kontrol)
```

- **Model:** Claude Opus 5.5 (`claude-opus-5-5`), uyarlamalı düşünme; ağır okuma
  işleri (yüzlerce haber metni) için alt ajanlarda daha ucuz model kullanılabilir.
- **Toplu ön-maç raporları:** Her hafta yüzlerce maç için rapor üretimi
  gerçek zamanlı değildir → **Batch API** (%50 daha ucuz) uygundur.
- **Prompt caching:** Sabit sistem talimatı ve araç tanımları önbelleğe alınır.
- **Bu repoda:** `agents/analyst.py` — araç kullanan tek analist ajan; model
  olasılıkları, form, geçmiş karşılaşmalar ve piyasa karşılaştırmasını araçlarla
  toplar, raporu `submit_report` aracıyla yapılandırılmış olarak teslim eder.
  Refusal durumunda sunucu tarafı yedek model (`fallbacks: "default"`) açıktır.

## 7. Değerlendirme ve şeffaflık

- **Walk-forward backtest** (`evaluation/backtest.py`): model her adımda yalnızca
  geçmiş veriyle eğitilir.
- Metrikler: log-loss, Brier, RPS, kalibrasyon tablosu, CLV, ROI.
- **Canlı kağıt üzerinde takip:** Her yayınlanan tahmin zaman damgasıyla
  değiştirilemez şekilde kaydedilir (tahmin anındaki oran dahil).
- **Herkese açık performans sayfası:** Tipster sektörünün en büyük sorunu
  güvensizlik. Doğrulanabilir, silinemeyen geçmiş en güçlü pazarlama aracıdır.

## 8. Ürün ve abonelik

| Paket | İçerik |
|---|---|
| Ücretsiz | Haftalık birkaç maç önizlemesi, performans sayfası |
| Pro (aylık) | Tüm desteklenen ligler, tam olasılıklar, ajan raporları, bildirimler |
| Elite (aylık) | Değer/risk paneli, kasa yönetimi, "Analiste sor" (sohbet ajanı), erken erişim |
| API (B2B) | Ham olasılık ve özellik API'si — medya, veri şirketleri, fantasy platformları |

Ödeme: Türkiye için iyzico/PayTR, uluslararası için Stripe. Abonelik yönetimi,
deneme süresi, kupon. **B2B API** uzun vadede tüketici aboneliğinden daha
istikrarlı gelir sağlayabilir ve hukuki riski daha düşüktür.

## 9. Hukuki ve uyum (lansmandan önce avukat görüşü şart)

- **Türkiye'de bahis:** Yasal bahis tek kanaldan yürür; yasa dışı bahsin
  reklamı/teşviki ve bu sitelere yönlendirme suçtur (7258 sayılı Kanun).
  Yasa dışı bahis sitesi reklamı, affiliate geliri veya yönlendirme **kesinlikle olmamalı**.
- Ürünü **"istatistiksel analiz ve veri"** olarak konumlandırın; kazanç vaadi
  ("garanti kupon", "%90 tutturma") kullanmayın. Tüketici mevzuatı açısından da risklidir.
- 18+ yaş doğrulaması, sorumlu oyun uyarıları, abonelik iptalinin kolay olması.
- **Veri lisansları:** Her kaynağın ticari kullanım ve yeniden dağıtım şartları
  (bu repoda `SourceInfo.license / commercial_use` alanlarıyla kod içinde izlenir).
  Kazıma (scraping) çoğu sitenin şartlarına aykırıdır.
- **Video/görsel telif** (FSEK, yayıncı hakları) — yukarıya bakın.
- **KVKK/GDPR:** Abone verisi, ödeme verisi, açık rıza, veri işleme sözleşmeleri.
- Yurtdışı pazarlarda (ör. İngiltere/AB) tipster ve bahis reklamı kuralları ayrıca incelenmeli.

## 10. Teknoloji yığını önerisi

| Katman | Başlangıç | Ölçeklendiğinde |
|---|---|---|
| Dil | Python (modeller, veri), TypeScript (web) | aynı |
| Depolama | Parquet + DuckDB, PostgreSQL | Lakehouse (Iceberg/Delta), TimescaleDB |
| İş akışı | Prefect veya Dagster | aynı + kuyruk (Redis/Kafka) canlı veri için |
| Model takibi | MLflow | aynı |
| API | FastAPI | aynı + önbellek (Redis) |
| Web | Next.js | aynı |
| Ajanlar | Claude API (araç kullanımı, Batch API) | Managed Agents / çoklu ajan |
| Gözlem | Sentry, Grafana | aynı |

## 11. Yol haritası

**Faz 0 — Temel (tamamlandı):** Kanonik veri şeması, kaynak arayüzü,
football-data.co.uk bağlayıcısı, Dixon-Coles, pazar türetme, oran matematiği
(Shin, Kelly, CLV), walk-forward backtest, Claude analist ajanı, testler.

**Faz 0.5 — Video ajanı (tamamlandı, Premier League):** Özet bulma, Gemini ile
izleme, gözlem deposu, takım profili, Claude analistine entegrasyon.

**Faz 1 — MVP (0–3 ay):**
- 5 büyük lig + Süper Lig; fikstür/kadro API'si; varlık eşleştirme tablosu
- Elo + xG tabanlı model + LightGBM; topluluk ve kalibrasyon
- Zaman noktası doğru özellik deposu
- Haftalık toplu ajan raporları (Batch API), editör/doğrulayıcı ajan
- Web arayüzü, ödeme, değiştirilemez tahmin kaydı, performans sayfası
- Kapalı beta; **hedef: kapanış piyasasına log-loss'ta yaklaşmak, kalibre olmak**

**Faz 2 — Derinlik (3–9 ay):**
- Event verisi lisansı → taktik parmak izi ve stil eşleşmesi modeli
- Oyuncu bazlı kadro gücü modeli; haber/sakatlık ajanı sinyallerinin backtest'i
- Ek pazarlar (Asya handikapı, korner, kart), portföy/risk paneli
- "Analiste sor" sohbet ajanı; B2B API

**Faz 3 — Ar-Ge (9+ ay):**
- Tracking verisi; lisanslı video üzerinde bilgisayarlı görü
- Canlı (maç içi) model ve bildirimler
- Ek ligler ve sporlar

## 12. Bu repodaki kod haritası

```
src/dualodds/
  data/schema.py                 Kanonik maç tablosu
  data/sources/base.py           Kaynak arayüzü + lisans metadatası + kayıt defteri
  data/sources/football_data_uk.py  Sonuç + oran bağlayıcısı
  data/synthetic.py              Gücü bilinen sentetik lig (test/demo)
  models/dixon_coles.py          L0 skor modeli
  models/markets.py              Skor matrisi → 1X2, alt/üst, KG, skorlar
  evaluation/odds.py             Marj temizleme (orantısal, Shin), EV, Kelly, CLV
  evaluation/metrics.py          Log-loss, Brier, RPS, kalibrasyon
  evaluation/backtest.py         Walk-forward backtest + bahis simülasyonu
  agents/tools.py                Ajan araçları (saf Python)
  agents/analyst.py              Claude analist ajanı
  agents/report.py               Rapor şeması (sayılar koddan, yorum LLM'den)
  data/entities.py               Takım adı eşleştirme (Premier League)
  video/finder.py                YouTube'da maç özeti bulma ve puanlama
  video/watcher.py               Gemini video izleme ajanı
  video/schema.py                Video gözlem şeması
  video/store.py                 Video analiz deposu
  video/profile.py               Takım bazlı video izci profili
  cli.py                         demo / fetch / backtest / predict / analyze
```
