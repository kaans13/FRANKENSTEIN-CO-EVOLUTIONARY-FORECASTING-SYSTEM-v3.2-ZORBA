# FRANKENSTEIN-CO-EVOLUTIONARY-FORECASTING-SYSTEM-v3.2-ZORBA

# Zorba v3.2: Frankenstein Co-Evolutionary Forecasting System

Zorba v3.2, finansal piyasalar için geliştirilmiş "Rejim-Farkındalıklı" bir zaman serisi tahminleme sistemidir. Finansal durum uzayı mahallelerini (FSN) kullanarak tarihsel analojiler üzerinden tahmin üretir.

## 🚀 Öne Çıkan Özellikler

- **Financial State Neighborhood (FSN):** Mahalanobis mesafesi ve rejim tespiti ile zenginleştirilmiş komşuluk algoritması.
- **Regime Detection:** Piyasa momentumu ve volatilitesine göre 4 farklı durum tespiti (Trend↑, Trend↓, Mean-Rev, Volatile).
- **Structural Break Penalty:** CUSUM algoritması ile anomali dönemlerinden (kırılmalardan) gelen verilerin otomatik filtrelenmesi.
- **Liquidity-Adjusted Temporal Decay:** Volatilitenin yüksek olduğu dönemlerde hafızayı kısaltan, stabil dönemlerde uzatan dinamik zaman ağırlıklandırması.
- **Hybrid Ensemble:** 10+ farklı modelin (LGBM, XGB, SVR, MLP vb.) ağırlıklı kombinasyonu.

## 📊 Metrikler
Sistem özellikle şu metriklere odaklanır:
- **Directional Accuracy (DA):** Fiyatın yönünü doğru tahmin etme oranı.
- **Return-Weighted DA (RWDA):** Yüksek getirili (volatil) anlardaki yön doğruluğuna daha fazla ağırlık veren metrik.

## 🛠 Kurulum ve Kullanım

```python
# Gerekli kütüphaneler
# numpy, pandas, sklearn, lightgbm, xgboost, torch

# Çalıştırma
python main.py


🏗 Mimari Şeması
Veri Hazırlama: Yüzde bazlı split (%70 Train, %15 Val, %15 Test).

FSN Fit: Eğitim verisinden kovaryans ve rejim eşiklerinin çıkarılması.

Pattern Discovery: Hedef değişkenin olasılıksal dağılımının çıkarılması.

Ensemble Tahmin: Modellerin eğitilmesi ve FSN analojisi ile ağırlıklandırılması.




---

### Metodolojik Eleştiri ve Notlar (Analitik Bakış)
Sistemi geliştirmek veya test etmek istersen şu noktaları sorgulamanı öneririm:

1.  **Regime Drift:** Rejim tespiti için kullanılan yüzdelik dilimler (percentiles) eğitim setinden alınıyor. Eğer piyasanın yapısı kalıcı olarak değişirse (örneğin 2026'daki volatilite tabanı 2024'ten çok farklıysa), bu sabit eşikler sistemi körleştirebilir. Dinamik bir *rolling percentile* mekanizması gerekebilir.
2.  **Mahalanobis Complexity:** $O(N \cdot D)$ işlem maliyeti büyük veri setlerinde FSN'i yavaşlatabilir. Gerçek zamanlı sistemlerde `KDTree` veya `FAISS` entegrasyonu performansı artıracaktır.
3.  **ACF-Momentum İlişkisi:** Otokorelasyon (ACF) bazlı momentum skoru, piyasanın "hafızasını" ölçmek için iyi bir proksidir. Ancak negatif ACF her zaman Mean-Reversion anlamına gelmez; bazen sadece gürültüdür (noise). Gürültü ile mean-reversion'ı ayıracak bir sinyal kalitesi filtresi eklenebilir.
4.  **Overfitting Riski:** Ensemble içinde çok fazla model olması (Ridge'den XGB'ye) validasyon setinde çok iyi sonuç verip test setinde "over-fitting" yapabilir. Ağırlıklandırmanın (ensemble weights) ne kadar dinamik olduğunu ve model çeşitliliğinin korelasyonunu kontrol etmelisin.
