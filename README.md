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

## İlkeler

1. **Sayılar modelden, yorum ajandan.** LLM olasılık uydurmaz.
2. **Sızıntı yok.** Her model yalnızca maçtan önce bilinen veriyle eğitilir.
3. **Rakip piyasa.** Başarı, kapanış oranlarına karşı log-loss/kalibrasyon ve CLV ile ölçülür.
4. **Lisanslı veri.** Her kaynak lisans metadatasıyla kaydedilir.
