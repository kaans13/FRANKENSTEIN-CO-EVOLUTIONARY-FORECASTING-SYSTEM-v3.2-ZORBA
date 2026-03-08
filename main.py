"""
🧬 FRANKENSTEIN CO-EVOLUTIONARY FORECASTING SYSTEM v3.2 — ZORBA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mimari: Zorba 3.2 — Yüzde-Bazlı Split + Finansal DSN + Rejim Tespiti

v3.1 → v3.2 DEĞİŞİKLİKLER:
  FIX-5: Yıl bazlı veri bölme → Yüzde bazlı (iloc)
          Train=%70 | Val=%15 | Test=%15
          2026 gibi kısa yıl sorununu tamamen çözer.

  FIX-6: DynamicStateNeighborhood → FinancialStateNeighborhood (FSN)
          Tamamen yeniden tasarlandı — finansal piyasa mekaniği ile:
          • Momentum-Reversal Rejim Tespiti (otomatik HMM-benzeri)
          • Liquidity-Adjusted Kernel: vol-ağırlıklı mesafe
          • Structural Break Detector: kırılma anlarında mahalle susturulur
          • Autocorrelation-Informed Bandwidth: lag-otokorelasyona göre dinamik h
          • Financal Analogy: "Benzer piyasa ortamlarını bul" prensibi

════════════════════════════════════════════════════════════════
KORUNAN TEMEL ÖZELLİKLER (v3.1'den):
  FIX-1: Hiperbolik zaman sönümlemesi — korundu
  FIX-2: Niş-Varyans Paradoksu — korundu
  FIX-3: T_min = 0.20 — korundu
  FIX-4: Anomali eşiği mekaniği → FSN'e entegre edildi

════════════════════════════════════════════════════════════════
FİNANSAL DURUM UZAYI (FSN) TASARIM FELSEFESİ:
  Bir portföy yöneticisi tarihsel analogları nasıl arar?
  1. "Bu dönem hangi rejimde?" (boğa/ayı/yatay/volatil)
  2. "Benzer rejimde hangi dönemler vardı?"
  3. "O dönemlerde bu sinyal ne anlama geliyordu?"
  4. "Yapısal kırılma var mı? (COVID, FTX, halving vb.)"

  FSN bu mantığı matematiksel olarak uygular:
  Adım A: Rejim Tespiti (4 durum: Trend-Up/Down, MeanRev, Volatile)
  Adım B: Aynı rejim içindeki geçmiş anlardan aday havuzu
  Adım C: Mahalanobis mesafesi (korelasyonu hesaba katar)
  Adım D: Structural break penaltısı
  Adım E: Liquidity-adjusted temporal weighting
  Adım F: Autocorrelation-informed prediction

════════════════════════════════════════════════════════════════
VERİ SIZINTISI KORUMALARI (tam liste — v3.2):
  ✦ PatternMemory.fit()     → SADECE y_train
  ✦ FSN.build()             → SADECE train dönemi (t < split)
  ✦ Rejim tespiti           → shift(1) bazlı, t anı yok
  ✦ Structural break        → eğitim dağılımından threshold
  ✦ Mahalanobis covariance  → sadece train'de hesaplanır
  ✦ Autocorr bandwidth      → sadece eğitim lag-ACF'inden
  ✦ StandardScaler          → sadece train'de fit
  ✦ Sıcaklık T              → vol_regime lag-bazlı
  ✦ discovery_rate          → val tahminleri üzerinden
  ✦ %70/%15/%15 split       → iloc bazlı, zamana göre sıralı
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KDTree
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from copy import deepcopy
import time
import gc
import itertools

warnings.filterwarnings('ignore')

# ============================================================================
# 🧹 BELLEK YÖNETİMİ
# ============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
print(f"🎮 Device: {device} | GPU: {use_gpu}")


# ============================================================================
# 📐 METRİKLER
# ============================================================================

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                          noise_threshold: float = 0.0) -> float:
    if noise_threshold > 0:
        mask = np.abs(y_true) >= noise_threshold
    else:
        mask = y_true != 0
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def return_weighted_da(y_true: np.ndarray, y_pred: np.ndarray,
                        noise_threshold: float = 0.0) -> float:
    if noise_threshold > 0:
        mask = np.abs(y_true) >= noise_threshold
    else:
        mask = y_true != 0
    if mask.sum() == 0:
        return 0.5
    returns  = y_true[mask]
    preds    = y_pred[mask]
    abs_ret  = np.abs(returns)
    w        = abs_ret / (abs_ret.sum() + 1e-9)
    correct  = (np.sign(returns) == np.sign(preds)).astype(float)
    return float(np.dot(correct, w))


def da_penalty(da_score: float) -> float:
    """DA=0.50→1.50 | DA=0.60→1.00 | DA=0.70→0.75"""
    return float(max(0.50, 2.25 - 2.5 * da_score))


# ============================================================================
# 🗓️ TATİL YARDIMCISI
# ============================================================================

US_HOLIDAYS = {
    (1,1),(7,4),(11,11),(12,25),(12,24),(12,26),
    (1,15),(1,16),(1,17),(1,18),(1,19),
    (2,18),(2,19),(2,20),
    (5,26),(5,27),(5,28),
    (9,1),(9,2),(9,3),
    (11,27),(11,28),(11,29),
}

def is_holiday(dt_series: pd.Series) -> np.ndarray:
    return np.array([(dt.month, dt.day) in US_HOLIDAYS for dt in dt_series])


# ============================================================================
# [F] 🧠 PATTERN MEMORY — SADECE EĞİTİM VERİSİNDEN FİT EDİLİR
# ============================================================================

class PatternMemory:
    def __init__(self, n_bins: int = 10):
        self.n_bins         = n_bins
        self.bin_edges      : Optional[np.ndarray] = None
        self.bin_centers    : np.ndarray = np.zeros(n_bins)
        self.bin_counts     : np.ndarray = np.zeros(n_bins)
        self.bin_priors     : np.ndarray = np.ones(n_bins) / n_bins
        self.bin_values     : List[np.ndarray] = [np.array([]) for _ in range(n_bins)]
        self.transition_mat : np.ndarray = np.ones((n_bins, n_bins)) / n_bins
        self._fitted        = False

    def fit(self, y_train: np.ndarray) -> 'PatternMemory':
        quantiles       = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges  = np.percentile(y_train, quantiles)
        self.bin_edges[0]  -= 1e-6
        self.bin_edges[-1] += 1e-6

        labels = np.digitize(y_train, self.bin_edges[1:-1])
        labels = np.clip(labels, 0, self.n_bins - 1)

        self.bin_counts = np.zeros(self.n_bins)
        self.bin_values = [[] for _ in range(self.n_bins)]
        for i in range(self.n_bins):
            mask = labels == i
            self.bin_counts[i] = mask.sum()
            self.bin_values[i] = y_train[mask]
            self.bin_centers[i] = float(np.median(y_train[mask])) if mask.sum() > 0 else \
                float((self.bin_edges[i] + self.bin_edges[i+1]) / 2.0)
            self.bin_values[i] = np.array(self.bin_values[i])

        total = self.bin_counts.sum()
        self.bin_priors = (self.bin_counts + 1e-6) / (total + self.n_bins * 1e-6)

        trans = np.ones((self.n_bins, self.n_bins)) * 1e-6
        for t in range(1, len(labels)):
            trans[labels[t-1], labels[t]] += 1.0
        row_sums = trans.sum(axis=1, keepdims=True)
        self.transition_mat = trans / (row_sums + 1e-9)

        self._fitted = True
        print(f"   🧠 PatternMemory: {self.n_bins} kova | "
              f"min={int(self.bin_counts.min())} max={int(self.bin_counts.max())}")
        return self

    def get_bin(self, delta_norm: float) -> int:
        if not self._fitted:
            return 0
        idx = int(np.searchsorted(self.bin_edges[1:], delta_norm))
        return int(np.clip(idx, 0, self.n_bins - 1))

    def get_bin_array(self, deltas: np.ndarray) -> np.ndarray:
        return np.array([self.get_bin(float(d)) for d in deltas])

    def inherit_from(self, donors: List['PatternMemory'],
                     donor_weights: np.ndarray) -> 'PatternMemory':
        child = PatternMemory(self.n_bins)
        if not self._fitted or not donors:
            return self
        child.bin_edges    = self.bin_edges.copy()
        child.bin_centers  = self.bin_centers.copy()
        child.bin_counts   = self.bin_counts.copy()
        child.bin_priors   = self.bin_priors.copy()
        child.bin_values   = [v.copy() for v in self.bin_values]

        best_donor_idx = int(np.argmax(donor_weights))
        new_trans = self.transition_mat.copy()
        for i, (d, w) in enumerate(zip(donors, donor_weights)):
            if not d._fitted or d.transition_mat.shape != self.transition_mat.shape:
                continue
            if i == best_donor_idx:
                new_trans = 0.4 * self.transition_mat + 0.6 * d.transition_mat
            else:
                new_trans += w * d.transition_mat

        total_w = 1.0 + sum(donor_weights)
        child.transition_mat = new_trans / (total_w + 1e-9)
        row_sums = child.transition_mat.sum(axis=1, keepdims=True)
        child.transition_mat = child.transition_mat / (row_sums + 1e-9)
        child._fitted = True
        return child


# ============================================================================
# [K] 💹 FİNANSAL DURUM UZAYI MAHALLESİ (FSN) v3.2
# ============================================================================
#
# TASARIM FELSEFESİ — BİR FİNANS ANALİSTİ GİBİ DÜŞÜN:
#
# Klasik kNN problemleri:
#   1. Mesafe metriği özelliklerin ölçeğini ve korelasyonunu görmez
#   2. Tüm tarih aynı ağırlıkta — 2020 bear market ile 2024 bull market
#      benzer mikro sinyallere rağmen tamamen farklı dinamikler
#   3. Kırılma noktalarında (COVID crash, FTX, halving) analog arama anlamsız
#   4. Gecikmeli otokorelasyon yapısı band genişliğini etkilemeli
#
# FSN Çözümleri:
#   A) 4-Durum Rejim Detektörü:
#      TRENDING_UP  : momentum > eşik, vol orta
#      TRENDING_DOWN: momentum < -eşik, vol yüksek
#      MEAN_REV     : düşük momentum, düşük vol (yatay piyasa)
#      VOLATILE     : vol > yüksek eşik (herhangi momentum)
#
#   B) Mahalanobis Mesafesi:
#      Özellikler arasındaki korelasyonu ve ölçeği hesaba katar.
#      d_M(x,y) = sqrt((x-y)^T Σ^-1 (x-y))
#      → Yüksek korelasyonlu özellikler çift sayılmaz
#
#   C) Structural Break Detector:
#      CUSUM algoritması → büyük kümülatif sapma = kırılma
#      Kırılma anı komşusu seçildiğinde ağırlık azaltılır
#
#   D) Volatility-Normalized Temporal Decay:
#      Düşük vol dönemlerde daha uzun hafıza (stabil piyasa)
#      Yüksek vol dönemlerde kısa hafıza (rejim değişimi hızlı)
#      w_time = 1 / (1 + λ_eff * Δt)
#      λ_eff = λ_base * (1 + vol_neighbor)
#
#   E) Autocorrelation-Informed Prediction:
#      Eğitim ACF'ından "ortalama kalanma süresi" hesaplanır
#      Yüksek pozitif ACF → daha fazla momentum ağırlığı
#      Negatif ACF → mean-reversion beklentisi artar
#
# ============================================================================

class FinancialStateNeighborhood:
    """
    Finansal Piyasa Dinamikleri için Durum Uzayı Mahallesi (FSN) v3.2.

    Bir finans analistinin tarihsel analog arama mantığını uygular:
    "Bu piyasa ortamı geçmişte ne zaman oluştu ve o zaman ne oldu?"

    4 Rejim: TRENDING_UP (0), TRENDING_DOWN (1), MEAN_REV (2), VOLATILE (3)

    ⚠️ SIZMA KORUMALARI:
      - Tüm eğitim istatistikleri SADECE train bölümünden
      - Mahalanobis kovaryans matrisi → sadece train'den
      - Rejim eşikleri → train dağılımından percentile
      - Structural break threshold → train'den
      - ACF bandwidth → train'den
      - Her sorguda t_i < t_query maskesi zorunlu
    """

    # Rejim sabitleri
    REGIME_TREND_UP   = 0
    REGIME_TREND_DOWN = 1
    REGIME_MEAN_REV   = 2
    REGIME_VOLATILE   = 3

    # Komşu sayıları
    K_REGIME_POOL = 400   # Aynı rejimden aday sayısı
    K_FINAL       = 60    # Final Mahalanobis komşusu

    # Hiperbolik decay
    LAMBDA_BASE   = 1.0 / (24 * 365)   # 1 yıllık temel hafıza

    # Alpha aralığı
    ALPHA_MIN     = 0.08
    ALPHA_MAX     = 0.38
    ALPHA_ANOMALY = 0.01   # Kırılma anı

    # Bandwidth
    BANDWIDTH_MULT = 1.2

    def __init__(self):
        self._fitted            = False
        self._train_features    : Optional[np.ndarray] = None   # (N, D) ham özellikler
        self._train_regimes     : Optional[np.ndarray] = None   # (N,) rejim etiketleri
        self._train_y           : Optional[np.ndarray] = None
        self._train_t           : Optional[np.ndarray] = None
        self._train_vol         : Optional[np.ndarray] = None   # normalize vol
        self._train_breaks      : Optional[np.ndarray] = None   # kırılma skoru

        # Mahalanobis
        self._cov_inv           : Optional[np.ndarray] = None
        self._feature_scaler    = StandardScaler()

        # Rejim eşikleri (train'den)
        self._mom_thr_up        : float = 0.0
        self._mom_thr_down      : float = 0.0
        self._vol_thr_high      : float = 0.0
        self._vol_thr_mid       : float = 0.0

        # Anomali / kırılma eşiği
        self._break_thr         : float = np.inf

        # Otomatik bandwidth
        self._bandwidth         : float = 1.0

        # ACF-tabanlı momentum skoru (train'den)
        self._acf_momentum      : float = 0.5   # [0,1]: 0=mean-rev, 1=momentum

    def fit(self, features_train: np.ndarray,
            y_train            : np.ndarray,
            t_indices          : Optional[np.ndarray] = None) -> 'FinancialStateNeighborhood':
        """
        features_train : (N, D) — ham özellikler (normalize edilmemiş)
                         Sütunlar: [delta_lag1, delta_lag24, vol_accel,
                                    roll_168_mean, roll_168_std,
                                    momentum_12, momentum_48]
        y_train        : (N,)   — normalize Δ hedefleri
        t_indices      : (N,)   — mutlak zaman indeksleri

        ⚠️ SIZMA: Tüm girdiler SADECE eğitim setinden.
        """
        valid = ~(np.isnan(features_train).any(axis=1) | np.isnan(y_train))
        F = features_train[valid]
        y = y_train[valid]
        t = t_indices[valid] if t_indices is not None else np.arange(len(y))

        if len(F) < self.K_FINAL:
            print(f"   ⚠️  FSN: Yetersiz nokta ({len(F)}), devre dışı.")
            return self

        # Özellik normalizasyonu (sadece train'de fit)
        self._feature_scaler.fit(F)
        F_scaled = self._feature_scaler.transform(F)

        # Rejim tespiti için özellik indeksleri
        # [0]=delta_lag1, [1]=delta_lag24, [2]=vol_accel,
        # [3]=roll_168_mean, [4]=roll_168_std, [5]=momentum_12, [6]=momentum_48
        mom   = F_scaled[:, 5] if F_scaled.shape[1] > 5 else F_scaled[:, 0]
        vol   = F_scaled[:, 4] if F_scaled.shape[1] > 4 else np.abs(F_scaled[:, 2])

        # Eşikler: train dağılımından
        self._mom_thr_up   = float(np.percentile(mom, 65))
        self._mom_thr_down = float(np.percentile(mom, 35))
        self._vol_thr_high = float(np.percentile(vol, 80))
        self._vol_thr_mid  = float(np.percentile(vol, 50))

        regimes = self._detect_regime_batch(mom, vol)

        # Structural break: CUSUM üzerinden kırılma skoru
        breaks = self._cusum_break_score(y)
        self._break_thr = float(np.percentile(breaks, 90))

        # Mahalanobis kovaryans (regularized)
        try:
            cov = np.cov(F_scaled.T)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            cov += np.eye(cov.shape[0]) * 1e-4   # Tikhonov regularizasyon
            self._cov_inv = np.linalg.inv(cov)
        except Exception:
            self._cov_inv = np.eye(F_scaled.shape[1])

        # ACF-tabanlı momentum skoru
        self._acf_momentum = self._compute_acf_momentum(y, max_lag=24)

        # Bandwidth: Mahalanobis mesafeleri üzerinden
        sample_n = min(300, len(F_scaled))
        idx_s    = np.random.choice(len(F_scaled), sample_n, replace=False)
        dists    = []
        for i in idx_s[:50]:
            diff = F_scaled - F_scaled[i]
            d_sq = np.einsum('ij,jk,ik->i', diff, self._cov_inv, diff)
            dists.extend(d_sq[:10].tolist())
        self._bandwidth = float(np.sqrt(np.median(dists)) * self.BANDWIDTH_MULT + 1e-8)

        self._train_features = F_scaled
        self._train_regimes  = regimes
        self._train_y        = y
        self._train_t        = t
        self._train_vol      = vol
        self._train_breaks   = breaks
        self._fitted         = True

        regime_counts = {0: int((regimes==0).sum()), 1: int((regimes==1).sum()),
                         2: int((regimes==2).sum()), 3: int((regimes==3).sum())}
        print(f"   💹 FSN v3.2 Finansal Durum Uzayı Mahallesi:")
        print(f"       Eğitim noktası: {len(F)} | Özellik boyutu: {F_scaled.shape[1]}D")
        print(f"       Rejimler: ▲{regime_counts[0]} ▼{regime_counts[1]} ↔{regime_counts[2]} 🌊{regime_counts[3]}")
        print(f"       ACF Momentum skoru: {self._acf_momentum:.3f} "
              f"({'momentum' if self._acf_momentum > 0.5 else 'mean-rev'} piyasa)")
        print(f"       Structural break eşiği: {self._break_thr:.4f} (90p)")
        print(f"       Mahalanobis bandwidth: {self._bandwidth:.4f}")
        print(f"       K_pool={self.K_REGIME_POOL} K_final={self.K_FINAL}")
        return self

    def _detect_regime_batch(self, mom: np.ndarray, vol: np.ndarray) -> np.ndarray:
        """4-durum rejim tespiti (vektörize)."""
        regimes = np.full(len(mom), self.REGIME_MEAN_REV, dtype=int)
        # Önce volatile kontrol (dominant)
        volatile_mask = vol > self._vol_thr_high
        # Trend (volatile değilse veya çok güçlü momentum varsa)
        trend_up_mask   = (mom > self._mom_thr_up)   & (~volatile_mask | (mom > self._mom_thr_up * 1.5))
        trend_down_mask = (mom < self._mom_thr_down) & (~volatile_mask | (mom < self._mom_thr_down * 1.5))

        regimes[volatile_mask]   = self.REGIME_VOLATILE
        regimes[trend_up_mask]   = self.REGIME_TREND_UP
        regimes[trend_down_mask] = self.REGIME_TREND_DOWN
        return regimes

    def _detect_regime_single(self, mom: float, vol: float) -> int:
        """Tek nokta rejim tespiti."""
        if vol > self._vol_thr_high:
            if abs(mom) > abs(self._mom_thr_up) * 1.5:
                return self.REGIME_TREND_UP if mom > 0 else self.REGIME_TREND_DOWN
            return self.REGIME_VOLATILE
        if mom > self._mom_thr_up:
            return self.REGIME_TREND_UP
        if mom < self._mom_thr_down:
            return self.REGIME_TREND_DOWN
        return self.REGIME_MEAN_REV

    @staticmethod
    def _cusum_break_score(y: np.ndarray, window: int = 48) -> np.ndarray:
        """
        CUSUM tabanlı yapısal kırılma skoru.
        Kırılma skoru yüksekse o komşunun bilgisi güvenilmez.
        """
        n      = len(y)
        scores = np.zeros(n)
        mu     = np.mean(y)
        sigma  = np.std(y) + 1e-8
        cusum  = 0.0
        for i in range(n):
            cusum        = max(0, cusum + (abs(y[i] - mu) / sigma) - 0.5)
            scores[i]    = cusum
            if i % window == 0:
                cusum = 0.0  # Periyodik sıfırlama
        # Normalize
        max_s = scores.max()
        return scores / (max_s + 1e-8)

    @staticmethod
    def _compute_acf_momentum(y: np.ndarray, max_lag: int = 24) -> float:
        """
        ACF'den momentum skoru hesapla.
        Pozitif ortalama ACF → momentum piyasası → daha uzun hafıza önemli
        Negatif ortalama ACF → mean-reversion → kısa hafıza yeterli
        Döndürür: [0, 1] skoru (0=pure mean-rev, 1=pure momentum)
        """
        if len(y) < max_lag * 3:
            return 0.5
        acf_vals = []
        mu = np.mean(y)
        var = np.var(y) + 1e-12
        for lag in range(1, min(max_lag + 1, len(y) // 3)):
            cov = float(np.mean((y[lag:] - mu) * (y[:-lag] - mu)))
            acf_vals.append(cov / var)
        if not acf_vals:
            return 0.5
        mean_acf = float(np.mean(acf_vals))
        # [-1, 1] → [0, 1]
        return float(np.clip((mean_acf + 1.0) / 2.0, 0.0, 1.0))

    def _mahalanobis_sq(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        x : (D,), Y : (N, D)
        Döndürür: (N,) Mahalanobis kare mesafeleri
        """
        diff = Y - x
        return np.einsum('ij,jk,ik->i', diff, self._cov_inv, diff)

    def neighbor_estimate(self,
                           f_query  : np.ndarray,
                           t_query  : int,
                           vol_query: float) -> Tuple[float, int, bool]:
        """
        f_query  : (D,) — normalize edilmiş özellik vektörü
        t_query  : int  — sorgu zamanı (mutlak indeks)
        vol_query: float — normalize vol değeri

        Döndürür: (y_neighbor, etkin_komşu_sayısı, kırılma_var_mı)

        Finansal Analoji: "Bu an ile en çok hangi geçmiş anlar benzeşiyor?"
        """
        if not self._fitted:
            return 0.0, 0, False

        # ─── Geçmiş maskesi ─────────────────────────────────────────────
        past_mask = self._train_t < t_query
        if past_mask.sum() < max(5, self.K_FINAL):
            return 0.0, 0, False

        past_idx  = np.where(past_mask)[0]
        F_past    = self._train_features[past_idx]
        y_past    = self._train_y[past_idx]
        t_past    = self._train_t[past_idx]
        reg_past  = self._train_regimes[past_idx]
        vol_past  = self._train_vol[past_idx]
        brk_past  = self._train_breaks[past_idx]

        # ─── ADIM A: Rejim Tespiti ───────────────────────────────────────
        if self._cov_inv is not None and F_past.shape[1] > 5:
            q_mom = float(f_query[5])
            q_vol = float(f_query[4])
        else:
            q_mom = float(f_query[0])
            q_vol = abs(float(f_query[2]))
        q_regime = self._detect_regime_single(q_mom, q_vol)

        # ─── ADIM B: Rejim Havuzu ────────────────────────────────────────
        # Önce aynı rejim; yoksa yakın rejimler genişlet
        same_regime = reg_past == q_regime
        n_same = same_regime.sum()

        if n_same >= self.K_FINAL:
            pool_mask = same_regime
        elif n_same >= self.K_FINAL // 2:
            # Yakın rejimler de dahil et
            adjacent = {
                self.REGIME_TREND_UP:   [self.REGIME_MEAN_REV],
                self.REGIME_TREND_DOWN: [self.REGIME_VOLATILE],
                self.REGIME_MEAN_REV:   [self.REGIME_TREND_UP, self.REGIME_TREND_DOWN],
                self.REGIME_VOLATILE:   [self.REGIME_TREND_DOWN],
            }.get(q_regime, [])
            adj_mask  = np.isin(reg_past, adjacent)
            pool_mask = same_regime | adj_mask
        else:
            # Yeterli komşu yok — tüm geçmişi kullan (düşük ağırlıkla)
            pool_mask = np.ones(len(past_idx), dtype=bool)

        pool_n = pool_mask.sum()
        if pool_n < 3:
            return 0.0, 0, False

        # Havuzu K_REGIME_POOL ile sınırla (performans)
        pool_indices_local = np.where(pool_mask)[0]
        if len(pool_indices_local) > self.K_REGIME_POOL:
            # Mahalanobis ile en yakın K_REGIME_POOL'u seç
            d_sq_all = self._mahalanobis_sq(f_query, F_past[pool_indices_local])
            kp_idx   = np.argpartition(d_sq_all, self.K_REGIME_POOL)[:self.K_REGIME_POOL]
            pool_indices_local = pool_indices_local[kp_idx]

        F_pool   = F_past[pool_indices_local]
        y_pool   = y_past[pool_indices_local]
        t_pool   = t_past[pool_indices_local]
        vol_pool = vol_past[pool_indices_local]
        brk_pool = brk_past[pool_indices_local]

        # ─── ADIM C: Mahalanobis Mesafesi ───────────────────────────────
        d_sq = self._mahalanobis_sq(f_query, F_pool)
        k_eff = min(self.K_FINAL, len(d_sq))
        knn_local = np.argpartition(d_sq, k_eff)[:k_eff]

        y_knn    = y_pool[knn_local]
        t_knn    = t_pool[knn_local]
        d_sq_knn = d_sq[knn_local]
        vol_knn  = vol_pool[knn_local]
        brk_knn  = brk_pool[knn_local]

        # ─── ADIM D: Structural Break Penaltısı ─────────────────────────
        # Kırılma noktası komşusu → ağırlık azalt
        break_penalty = 1.0 / (1.0 + brk_knn * 3.0)

        # Kırılma varsa (ortalama break skoru eşiği geçiyorsa) anomali
        mean_brk = float(np.mean(brk_knn))
        is_break = mean_brk > self._break_thr * 0.7

        # ─── ADIM E: Vol-Normalized Hiperbolik Decay ────────────────────
        # Düşük vol dönemlerde uzun hafıza, yüksek vol'da kısa hafıza
        # λ_eff = λ_base * (1 + ortalama_vol_komşu)
        avg_vol_knn = float(np.mean(vol_knn))
        lambda_eff  = self.LAMBDA_BASE * (1.0 + avg_vol_knn)
        time_diff   = (t_query - t_knn).astype(float)
        w_time      = 1.0 / (1.0 + lambda_eff * time_diff)

        # ─── ADIM F: Gaussian Kernel (Mahalanobis mesafesi üzerinde) ────
        w_kernel = np.exp(-0.5 * d_sq_knn / (self._bandwidth ** 2 + 1e-12))

        # Toplam ağırlık
        weights = w_time * w_kernel * break_penalty
        w_sum   = weights.sum()
        if w_sum < 1e-12:
            return float(np.mean(y_knn)), k_eff, is_break

        # ─── ACF-Adjusted Prediction ─────────────────────────────────────
        # Yüksek momentum piyasasında: yakın komşular daha önemli
        # Mean-rev piyasasında: uzak (reverting) komşular da değerli
        if self._acf_momentum > 0.65:
            # Momentum: yakın zamanları ek ağırlıkla
            recency_bonus    = np.exp(-0.5 * time_diff / (24 * 30))  # 1 ay yarı-ömür
            weights          = weights * (1.0 + recency_bonus * 0.5)
            w_sum            = weights.sum()

        y_neighbor = float(np.dot(weights, y_knn) / (w_sum + 1e-12))
        return y_neighbor, k_eff, is_break

    def _compute_alpha(self, vol_norm: float) -> float:
        """Volatilite yüksekse α küçük (model baskın)."""
        ratio = float(np.clip((1.5 - vol_norm) / (1.5 - 0.5 + 1e-8), 0, 1))
        return self.ALPHA_MIN + (self.ALPHA_MAX - self.ALPHA_MIN) * ratio

    def correct(self, y_model_pred    : np.ndarray,
                F_query              : np.ndarray,
                t_query_array        : np.ndarray,
                vol_regime_array     : np.ndarray) -> Tuple[np.ndarray, int]:
        """
        y_model_pred   : (N,)   — ham model tahminleri
        F_query        : (N, D) — normalize edilmemiş özellikler
        t_query_array  : (N,)   — mutlak zaman indeksleri
        vol_regime_array: (N,)  — volatilite rejimi (lag-tabanlı)
        """
        if not self._fitted:
            return y_model_pred, 0

        # Normalize et (sadece transform — train'de fit edildi)
        F_scaled = self._feature_scaler.transform(F_query)

        corrected     = np.zeros_like(y_model_pred)
        break_count   = 0

        for i in range(len(y_model_pred)):
            y_nb, k_ef, is_break = self.neighbor_estimate(
                F_scaled[i], int(t_query_array[i]), float(vol_regime_array[i]))

            if is_break:
                alpha = self.ALPHA_ANOMALY
                break_count += 1
                corrected[i] = (1.0 - alpha) * y_model_pred[i]
            elif k_ef < 3:
                corrected[i] = y_model_pred[i]
            else:
                alpha        = self._compute_alpha(float(vol_regime_array[i]))
                corrected[i] = (1.0 - alpha) * y_model_pred[i] + alpha * y_nb

        return corrected, break_count


# ============================================================================
# [I] 🎯 NİŞ SKORU — AYKIRI DOĞRULUK v3.1 (korundu)
# ============================================================================

def calculate_niche_fitness(model_pred: np.ndarray,
                             all_preds: List[np.ndarray],
                             y_true: np.ndarray,
                             noise_threshold: float = 0.0,
                             discovery_bonus: float = 3.0
                             ) -> Tuple[float, float]:
    if not all_preds or len(y_true) == 0:
        base = mean_absolute_error(y_true, model_pred) if len(y_true) > 0 else np.inf
        return base, 0.0

    n     = min(len(y_true), len(model_pred), min(len(p) for p in all_preds))
    y_t   = y_true[:n]
    y_m   = model_pred[:n]
    all_p = np.column_stack([p[:n] for p in all_preds])

    mask = (np.abs(y_t) >= noise_threshold) if noise_threshold > 0 else (y_t != 0)
    base_mae = mean_absolute_error(y_t, y_m)

    if mask.sum() == 0:
        return base_mae, 0.0

    y_t_f   = y_t[mask]
    y_m_f   = y_m[mask]
    all_p_f = all_p[mask, :]

    pop_median_sign = np.sign(np.median(all_p_f, axis=1))
    true_sign       = np.sign(y_t_f)
    crowd_wrong     = (pop_median_sign != true_sign)
    model_right     = (np.sign(y_m_f) == true_sign)
    discovery_mask  = crowd_wrong & model_right
    n_discoveries   = int(discovery_mask.sum())
    n_filtered      = int(mask.sum())
    discovery_rate  = n_discoveries / float(n_filtered) if n_filtered > 0 else 0.0

    if n_discoveries == 0:
        return base_mae, 0.0

    niche_mae = base_mae * np.exp(-discovery_bonus * discovery_rate)
    return float(niche_mae), float(discovery_rate)


# ============================================================================
# [L] 🌡️ SICAKLIK KONTROLLÜ ENSEMBLE (korundu, T_min=0.20)
# ============================================================================

def temperature_ensemble(predictions: List[np.ndarray],
                          base_weights: np.ndarray,
                          vol_signal: float,
                          T_min: float = 0.20,
                          T_max: float = 1.0,
                          vol_low: float  = 0.3,
                          vol_high: float = 1.5) -> np.ndarray:
    ratio = float(np.clip((vol_high - vol_signal) / (vol_high - vol_low + 1e-8), 0, 1))
    T     = T_min + (T_max - T_min) * ratio
    log_w  = np.log(np.maximum(base_weights, 1e-9))
    scaled = log_w / (T + 1e-9)
    scaled -= scaled.max()
    exp_w  = np.exp(scaled)
    return exp_w / (exp_w.sum() + 1e-9)


# ============================================================================
# 📊 VERİ İŞLEYİCİ v3.2 — %70/%15/%15 YÜZDE BAZLI BÖLME
# ============================================================================

class TimeSeriesDataHandler:
    """
    v3.2 Değişiklikleri:
      FIX-5: Yıl bazlı split → Yüzde bazlı (iloc)
              Train=%70 | Val=%15 | Test=%15
      FIX-6: DynamicStateNeighborhood → FinancialStateNeighborhood (FSN)
              Finansal analog arama: Rejim + Mahalanobis + Break + ACF
    """

    FEATURE_COLS_BASE = [
        'lag_24_diff', 'lag_48_diff', 'lag_168_diff',
        'roll_24_mean_diff', 'roll_24_std_diff',
        'roll_168_mean_diff', 'roll_168_std_diff',
        'lag_interaction_diff',
        'acceleration',
        'volatility_regime',
        'hour_sin',  'hour_cos',
        'hour_sin2', 'hour_cos2',
        'hour_sin3', 'hour_cos3',
        'dow_sin',   'dow_cos',
        'month_sin', 'month_cos',
        'month_sin2','month_cos2',
        'month_sin3','month_cos3',
        'month', 'dow', 'quarter', 'is_weekend', 'is_holiday',
    ]

    # FSN özellik sütunları (7D)
    FSN_FEATURE_COLS = [
        'delta_lag1', 'delta_lag24', 'vol_accel',
        'roll_168_mean', 'roll_168_std',
        'momentum_12', 'momentum_48',
    ]

    def __init__(self,
                 filepath           : str,
                 sequence_length    : int   = 24,
                 target_col         : str   = 'price',
                 train_frac         : float = 0.70,
                 val_frac           : float = 0.15,
                 noise_sigma        : float = 0.01,
                 noise_threshold_pct: float = 0.30,
                 n_bins             : int   = 10,
                 # v3.1 uyumluluk parametreleri (artık kullanılmıyor)
                 train_years        : tuple = None,
                 val_year           : int   = None,
                 test_year          : int   = None,
                 ):
        self.filepath            = filepath
        self.sequence_length     = sequence_length
        self.target_col          = target_col
        self.train_frac          = train_frac
        self.val_frac            = val_frac
        # test_frac = 1 - train_frac - val_frac
        self.noise_sigma         = noise_sigma
        self.noise_threshold_pct = noise_threshold_pct
        self.n_bins              = n_bins

        self.feat_scaler         = StandardScaler()
        self.target_scaler       = StandardScaler()
        self.noise_threshold     : float = 0.0

        self.pattern_memory      = PatternMemory(n_bins)

        # [K→FSN] Finansal Durum Uzayı Mahallesi
        self.financial_neighborhood = FinancialStateNeighborhood()

        self.X_train_2d = self.y_train = None
        self.X_val_2d   = self.y_val   = None
        self.X_test_2d  = self.y_test  = None
        self.X_train_3d = self.X_val_3d = self.X_test_3d = None
        self._val_prices  : Optional[np.ndarray] = None
        self._test_prices : Optional[np.ndarray] = None

        # FSN özellikleri (7D, ham ölçekte)
        self._F_train : Optional[np.ndarray] = None
        self._F_val   : Optional[np.ndarray] = None
        self._F_test  : Optional[np.ndarray] = None
        self._t_train : Optional[np.ndarray] = None
        self._t_val   : Optional[np.ndarray] = None
        self._t_test  : Optional[np.ndarray] = None

        self._vol_regime_idx = self.FEATURE_COLS_BASE.index('volatility_regime')

        # Rolling splits için dataframe
        self.rolling_splits: List[Tuple] = []

    @property
    def FEATURE_COLS(self):
        return self.FEATURE_COLS_BASE

    # ──────────────────────────────────────────────────────────────────
    def load_and_prepare(self):
        print("\n" + "="*70)
        print("📊 VERİ YÜKLENİYOR | v3.2 ZORBA — FSN + Yüzde Bazlı Split")
        print("="*70)

        df = pd.read_csv(self.filepath)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)

        if self.target_col not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.target_col = num_cols[0]
            print(f"   ⚠️  Otomatik hedef: '{self.target_col}'")

        if self.target_col != 'price':
            df = df.rename(columns={self.target_col: 'price'})

        total_rows = len(df)
        date_min   = df['Datetime'].min()
        date_max   = df['Datetime'].max()
        print(f"   Toplam satır: {total_rows:,}")
        print(f"   Tarih aralığı: {date_min.date()} → {date_max.date()}")

        # ── FIX-5: YÜZDE BAZLI BÖLME ────────────────────────────────
        n_train = int(total_rows * self.train_frac)
        n_val   = int(total_rows * self.val_frac)
        n_test  = total_rows - n_train - n_val   # geriye kalan

        train_df = df.iloc[:n_train].copy().reset_index(drop=True)
        val_df   = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
        test_df  = df.iloc[n_train + n_val:].copy().reset_index(drop=True)

        print(f"\n   Split Yüzde Bazlı (%70/%15/%15):")
        print(f"   Train : {len(train_df):,} satır  "
              f"({train_df['Datetime'].min().date()} → {train_df['Datetime'].max().date()})")
        print(f"   Val   : {len(val_df):,} satır  "
              f"({val_df['Datetime'].min().date()} → {val_df['Datetime'].max().date()})")
        print(f"   Test  : {len(test_df):,} satır  "
              f"({test_df['Datetime'].min().date()} → {test_df['Datetime'].max().date()})")

        if len(test_df) < 24:
            raise ValueError(f"Test seti çok küçük: {len(test_df)} satır. "
                             f"Veri setini kontrol et.")

        df['price_diff'] = df['price'].diff()
        train_df['price_diff'] = train_df['price'].diff()
        val_df['price_diff']   = val_df['price'].diff()
        test_df['price_diff']  = test_df['price'].diff()

        # PatternMemory için ham fit
        X_raw, y_raw = self._feat(train_df, fit=True)
        self.pattern_memory.fit(y_raw)

        # Gerçek volreg ile ikinci geçiş
        self.X_train_2d, self.y_train = self._feat(train_df, fit=True,  recompute_volreg=True)
        self.X_val_2d,   self.y_val   = self._feat(val_df,   fit=False, recompute_volreg=True)
        self.X_test_2d,  self.y_test  = self._feat(test_df,  fit=False, recompute_volreg=True)

        self.noise_threshold = self.noise_threshold_pct

        # [K→FSN] FSN özellikleri üret (sızıntısız)
        self._build_fsn_features(train_df, val_df, test_df)

        # FSN fit — SADECE eğitim verisi
        n_tr      = len(self._F_train)
        y_for_fsn = self.y_train[-n_tr:] if len(self.y_train) > n_tr else self.y_train

        self.financial_neighborhood.fit(
            features_train = self._F_train,
            y_train        = y_for_fsn,
            t_indices      = self._t_train)

        # Val ve test mutlak zaman indeksleri
        n_train_abs = len(self.y_train)
        n_vl        = len(self._F_val)
        n_te        = len(self._F_test)
        self._t_val  = np.arange(n_train_abs, n_train_abs + n_vl)
        self._t_test = np.arange(n_train_abs + n_vl, n_train_abs + n_vl + n_te)

        self._val_prices  = self._get_aligned_prices(val_df,  self.y_val)
        self._test_prices = self._get_aligned_prices(test_df, self.y_test)

        self.X_train_3d = self._seq(self.X_train_2d)
        self.X_val_3d   = self._seq(self.X_val_2d)
        self.X_test_3d  = self._seq(self.X_test_2d)

        self._build_rolling(df, n_train, n_val)

        print(f"\n   Özellikler: {self.X_train_2d.shape[1]}")
        print(f"   Delta std : {np.std(self.y_train):.4f}  mean: {np.mean(self.y_train):.4f}")
        print(f"   DA eşiği  : |Δ_norm| ≥ {self.noise_threshold:.3f}")

    def _build_fsn_features(self, train_df, val_df, test_df):
        """
        [K→FSN] v3.2 — 7 Boyutlu Finansal Durum Vektörü

        Sütunlar:
          [0] delta_lag1     : Δ_{t-1}    (anlık momentum proxy)
          [1] delta_lag24    : Δ_{t-24}   (günlük fark)
          [2] vol_accel      : volatilite ivmesi (kısa - uzun vol)
          [3] roll_168_mean  : haftalık kayan ortalama (rejim merkezi)
          [4] roll_168_std   : haftalık kayan vol (rejim genişliği)
          [5] momentum_12    : 12 saatlik kümülatif momentum
          [6] momentum_48    : 48 saatlik kümülatif momentum

        Tüm özellikler shift(1) veya daha büyük lag → sızıntı sıfır.
        """
        def build_fsn_F(df):
            d = df.copy()
            pdiff = d['price_diff']

            d['delta_lag1']    = pdiff.shift(1)
            d['delta_lag24']   = pdiff.shift(24)

            # Volatilite ivmesi
            roll6              = pdiff.shift(1).rolling(6,  min_periods=1).std()
            roll24             = pdiff.shift(1).rolling(24, min_periods=1).std()
            d['vol_accel']     = roll6 - roll24

            # Haftalık rejim (shift=1 → t-1'den başlayan 168 pencere)
            lagged = pdiff.shift(1)
            d['roll_168_mean'] = lagged.rolling(168, min_periods=24).mean()
            d['roll_168_std']  = lagged.rolling(168, min_periods=24).std()

            # Kümülatif momentum (12h ve 48h — shift(1) bazlı)
            d['momentum_12']   = pdiff.shift(1).rolling(12, min_periods=1).sum()
            d['momentum_48']   = pdiff.shift(1).rolling(48, min_periods=1).sum()

            d = d.dropna(subset=self.FSN_FEATURE_COLS)
            return d[self.FSN_FEATURE_COLS].values.astype(float)

        F_tr = build_fsn_F(train_df)
        F_vl = build_fsn_F(val_df)
        F_te = build_fsn_F(test_df)

        n_tr = min(len(self.y_train), len(F_tr))
        n_vl = min(len(self.y_val),   len(F_vl))
        n_te = min(len(self.y_test),  len(F_te))

        self._F_train = F_tr[-n_tr:]
        self._F_val   = F_vl[-n_vl:]
        self._F_test  = F_te[-n_te:]
        self._t_train = np.arange(n_tr)

    def get_fsn_features(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """(F_features, t_indices) döndür."""
        if split == 'train':
            return self._F_train, self._t_train
        elif split == 'val':
            return self._F_val, self._t_val
        else:
            return self._F_test, self._t_test

    def _get_aligned_prices(self, split_df, y_split):
        prices  = split_df['price'].values
        n_model = len(y_split)
        if len(prices) > n_model:
            return prices[-n_model:]
        return prices

    def _compute_volatility_regime(self, price_diff_series: pd.Series,
                                    n_bins: int = 10) -> np.ndarray:
        if not self.pattern_memory._fitted:
            return np.zeros(len(price_diff_series))
        diff_values = price_diff_series.values.astype(float)
        result      = np.zeros(len(diff_values))
        for i in range(len(diff_values)):
            if i < 26:
                result[i] = 0.0
                continue
            window = diff_values[i-25:i-1]
            if len(window) < 2:
                result[i] = 0.0
                continue
            bins      = self.pattern_memory.get_bin_array(window)
            bin_diffs = np.abs(np.diff(bins.astype(float)))
            result[i] = float(np.sum(bin_diffs >= 2)) / 23.0
        return result

    def _feat(self, df: pd.DataFrame, fit: bool = False,
              feat_sc: StandardScaler = None,
              tgt_sc:  StandardScaler = None,
              add_noise: bool = False,
              recompute_volreg: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        fsc = feat_sc if feat_sc else self.feat_scaler
        tsc = tgt_sc  if tgt_sc  else self.target_scaler

        df = df.copy()
        dt = df['Datetime']

        df['hour']       = dt.dt.hour
        df['dow']        = dt.dt.dayofweek
        df['month']      = dt.dt.month
        df['quarter']    = dt.dt.quarter
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        df['is_holiday'] = is_holiday(dt).astype(int)

        for col, period in [('hour', 24), ('dow', 7), ('month', 12)]:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

        for k in [2, 3]:
            df[f'hour_sin{k}']  = np.sin(2 * np.pi * k * df['hour']  / 24)
            df[f'hour_cos{k}']  = np.cos(2 * np.pi * k * df['hour']  / 24)
            df[f'month_sin{k}'] = np.sin(2 * np.pi * k * df['month'] / 12)
            df[f'month_cos{k}'] = np.cos(2 * np.pi * k * df['month'] / 12)

        price_diff = df['price'].diff()
        for lag in [24, 48, 168]:
            df[f'lag_{lag}_diff'] = price_diff.shift(lag)
        shifted_diff = price_diff.shift(24)
        for window in [24, 168]:
            df[f'roll_{window}_mean_diff'] = shifted_diff.rolling(window, min_periods=1).mean()
            df[f'roll_{window}_std_diff']  = shifted_diff.rolling(window, min_periods=1).std()
        df['lag_interaction_diff'] = df['lag_24_diff'] - df['lag_168_diff']
        df['acceleration'] = price_diff.shift(1) - price_diff.shift(2)

        if recompute_volreg and self.pattern_memory._fitted:
            df['volatility_regime'] = self._compute_volatility_regime(price_diff, self.n_bins)
        else:
            df['volatility_regime'] = 0.0

        df = df.dropna().reset_index(drop=True)
        X  = df[self.FEATURE_COLS].values

        if 'price_diff' in df.columns:
            y_raw = df['price_diff'].values
        else:
            y_raw = df['price'].diff().values

        valid = ~np.isnan(y_raw)
        X     = X[valid]
        y_raw = y_raw[valid]

        if fit:
            y = tsc.fit_transform(y_raw.reshape(-1, 1)).ravel()
            X = fsc.fit_transform(X)
        else:
            y = tsc.transform(y_raw.reshape(-1, 1)).ravel()
            X = fsc.transform(X)

        if add_noise and self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, X.shape)
            X = X * (1.0 + noise)

        return X, y

    def get_vol_regime(self, split: str) -> np.ndarray:
        if split == 'train': X = self.X_train_2d
        elif split == 'val': X = self.X_val_2d
        else:                X = self.X_test_2d
        if X is None: return np.zeros(0)
        return X[:, self._vol_regime_idx]

    def _seq(self, X: np.ndarray) -> np.ndarray:
        n, f = X.shape
        S = np.zeros((n, self.sequence_length, f))
        for i in range(n):
            if i < self.sequence_length:
                pad = self.sequence_length - i - 1
                S[i, :pad, :] = X[0:1, :]
                S[i, pad:, :] = X[:i+1, :]
            else:
                S[i] = X[i - self.sequence_length + 1:i + 1]
        return S

    def _build_rolling(self, df: pd.DataFrame, n_train_base: int, n_val_base: int):
        """
        Rolling validation — yüzde bazlı split ile uyumlu.
        Son 3 val bloğu oluştur (expanding window).
        """
        total = len(df)
        # 3 fold için train bitiş noktaları
        folds = []
        for k in range(3, 0, -1):
            # k/4, k+1/4 şeklinde expanding
            tr_end = int(total * (0.70 - 0.08 * (k - 1)))
            vl_end = tr_end + int(total * 0.08)
            if tr_end > 200 and vl_end <= total and vl_end - tr_end > 24:
                folds.append((tr_end, vl_end))

        self.rolling_splits = []
        for tr_end, vl_end in folds:
            tr_df = df.iloc[:tr_end].copy().reset_index(drop=True)
            vl_df = df.iloc[tr_end:vl_end].copy().reset_index(drop=True)
            tr_df['price_diff'] = tr_df['price'].diff()
            vl_df['price_diff'] = vl_df['price'].diff()

            fsc = StandardScaler()
            tsc = StandardScaler()
            Xtr, ytr = self._feat(tr_df, fit=True,  feat_sc=fsc, tgt_sc=tsc, recompute_volreg=True)
            Xvl, yvl = self._feat(vl_df, fit=False, feat_sc=fsc, tgt_sc=tsc, recompute_volreg=True)
            self.rolling_splits.append((Xtr, ytr, Xvl, yvl))

        print(f"   Rolling splits: {len(self.rolling_splits)} fold")

    def get_data(self, model_type: str, split: str):
        is3d = (model_type == 'lstm')
        if split == 'train':
            return (self.X_train_3d if is3d else self.X_train_2d), self.y_train
        elif split == 'val':
            return (self.X_val_3d   if is3d else self.X_val_2d),   self.y_val
        elif split == 'test':
            return (self.X_test_3d  if is3d else self.X_test_2d),  self.y_test
        raise ValueError(f"Unknown split: {split}")

    def to_price_level_anchored(self, delta_pred_norm: np.ndarray,
                                 split: str) -> np.ndarray:
        real_delta  = self.target_scaler.inverse_transform(
            delta_pred_norm.reshape(-1, 1)).ravel()
        true_prices = self._val_prices if split == 'val' else self._test_prices
        n = min(len(true_prices), len(real_delta))
        prices = np.zeros(n)
        for t in range(n):
            anchor   = true_prices[0] if t == 0 else true_prices[t - 1]
            prices[t] = anchor + real_delta[t]
        return prices

    def get_true_prices(self, split: str) -> np.ndarray:
        return self._val_prices if split == 'val' else self._test_prices


# ============================================================================
# 🤖 LSTM
# ============================================================================

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ============================================================================
# 🏗️ MODEL CONFIG
# ============================================================================

@dataclass
class ModelConfig:
    model_type     : str
    params         : Dict[str, Any] = field(default_factory=dict)
    pattern_memory : Optional[PatternMemory] = field(default=None, compare=False, repr=False)
    epigenetic_source : Optional[str] = field(default=None, compare=False, repr=False)

    def clone(self):
        c = ModelConfig(self.model_type, deepcopy(self.params))
        c.pattern_memory     = self.pattern_memory
        c.epigenetic_source  = self.epigenetic_source
        return c

    def numeric_params(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.params.items()
                if isinstance(v, (int, float))}


# ============================================================================
# 🎯 MODEL WRAPPER v3.2 — FSN entegrasyonu
# ============================================================================

SKIP_ROLLING = {'lstm', 'svr'}

PARAM_BLOCKS: Dict[str, List[List[str]]] = {
    'lgbm':    [['learning_rate'], ['num_leaves', 'max_depth'], ['n_estimators']],
    'xgboost': [['learning_rate'], ['max_depth', 'subsample', 'colsample_bytree'], ['n_estimators']],
    'rf':      [['n_estimators'], ['max_depth', 'min_samples_split', 'min_samples_leaf']],
    'gbm':     [['learning_rate', 'subsample'], ['n_estimators', 'max_depth']],
    'ridge':   [['alpha']],
    'lasso':   [['alpha']],
    'elasticnet': [['alpha'], ['l1_ratio']],
    'svr':     [['C'], ['epsilon']],
    'lstm':    [['hidden_size', 'num_layers'], ['learning_rate'], ['epochs', 'batch_size']],
}


class ModelWrapper:
    def __init__(self, config: ModelConfig, data_handler: TimeSeriesDataHandler):
        self.config           = config
        self.data_handler     = data_handler
        self.model            = None
        self.train_time       = 0.0
        self.val_mae          = np.inf
        self.val_da           = 0.5
        self.val_rwda         = 0.5
        self.niche_fitness    = np.inf
        self.discovery_rate   = 0.0
        self.fitness          = np.inf
        self.pred_variance    = 0.0
        self.mcs_score        = 0.0
        self.rank             = 999
        self.predictions      : Dict[str, np.ndarray] = {}
        self.predictions_price: Dict[str, np.ndarray] = {}

    def train(self, use_rolling: bool = True, use_noise: bool = True):
        t0 = time.time()
        X_tr, y_tr = self.data_handler.get_data(self.config.model_type, 'train')
        X_vl, y_vl = self.data_handler.get_data(self.config.model_type, 'val')

        if use_noise and self.config.model_type != 'lstm' and self.data_handler.noise_sigma > 0:
            noise      = np.random.normal(0, self.data_handler.noise_sigma, X_tr.shape)
            X_tr_noisy = X_tr * (1.0 + noise)
        else:
            X_tr_noisy = X_tr

        {
            'ridge':      self._train_ridge,
            'lasso':      self._train_lasso,
            'elasticnet': self._train_elasticnet,
            'lgbm':       self._train_lgbm,
            'xgboost':    self._train_xgboost,
            'rf':         self._train_rf,
            'gbm':        self._train_gbm,
            'svr':        self._train_svr,
            'lstm':       self._train_lstm,
        }[self.config.model_type](X_tr_noisy, y_tr, X_vl, y_vl)

        self.train_time = time.time() - t0

        if use_rolling and self.config.model_type not in SKIP_ROLLING and self.data_handler.rolling_splits:
            maes = []
            for Xtr_f, ytr_f, Xvl_f, yvl_f in self.data_handler.rolling_splits:
                tmp = self._quick_predict(Xtr_f, ytr_f, Xvl_f)
                maes.append(mean_absolute_error(yvl_f, tmp))
            self.val_mae = float(np.mean(maes))
        else:
            self.val_mae = mean_absolute_error(y_vl, self.predict('val'))

        if 'val' not in self.predictions:
            self.predictions['val'] = self.predict('val')

        pv  = self.predictions['val']
        thr = self.data_handler.noise_threshold
        self.val_da   = directional_accuracy(y_vl, pv, noise_threshold=thr)
        self.val_rwda = return_weighted_da(y_vl,   pv, noise_threshold=thr)

        tgt_std = float(np.std(y_vl)) if np.std(y_vl) > 0 else 1.0
        self.pred_variance = float(np.std(pv)) / tgt_std

        avg_da      = (self.val_da + self.val_rwda) / 2.0
        dp          = da_penalty(avg_da)
        time_factor = 1.0 + np.log1p(self.train_time)
        var_penalty = 2.5 if self.pred_variance < 0.15 else (1.5 if self.pred_variance < 0.30 else 1.0)
        self.fitness = self.val_mae * time_factor * dp * var_penalty
        self.niche_fitness = self.fitness
        clear_memory()

    def _raw_predict(self, split: str) -> np.ndarray:
        X, _ = self.data_handler.get_data(self.config.model_type, split)
        if self.config.model_type == 'lstm':
            self.model.eval()
            with torch.no_grad():
                pred = self.model(torch.FloatTensor(X).to(device)).cpu().numpy().flatten()
        else:
            pred = self.model.predict(X)
        return pred

    def predict(self, split: str) -> np.ndarray:
        """[K→FSN] v3.2 Finansal Durum Uzayı Mahallesi ile düzeltilmiş tahmin."""
        if split in self.predictions:
            return self.predictions[split]

        raw_pred    = self._raw_predict(split)
        F, t_arr    = self.data_handler.get_fsn_features(split)
        vol_regime  = self.data_handler.get_vol_regime(split)

        n = min(len(raw_pred), len(F), len(t_arr), len(vol_regime))
        raw_pred   = raw_pred[:n]
        F          = F[:n]
        t_arr      = t_arr[:n]
        vol_regime = vol_regime[:n]

        # [K→FSN] FSN düzeltmesi
        corrected, n_breaks = self.data_handler.financial_neighborhood.correct(
            y_model_pred     = raw_pred,
            F_query          = F,
            t_query_array    = t_arr,
            vol_regime_array = vol_regime,
        )
        if n_breaks > 0:
            print(f"   ⚠️  FSN [{split}]: {n_breaks}/{n} yapısal kırılma "
                  f"(α={FinancialStateNeighborhood.ALPHA_ANOMALY})")
        self.predictions[split] = corrected
        return corrected

    def predict_price(self, split: str) -> np.ndarray:
        if split in self.predictions_price:
            return self.predictions_price[split]
        delta_pred = self.predict(split)
        price_pred = self.data_handler.to_price_level_anchored(delta_pred, split)
        self.predictions_price[split] = price_pred
        return price_pred

    def clone(self):
        return ModelWrapper(self.config.clone(), self.data_handler)

    def _quick_predict(self, Xtr, ytr, Xvl):
        mt = self.config.model_type
        p  = self.config.params
        if mt == 'ridge':
            m = Ridge(alpha=p.get('alpha', 1.0), random_state=42).fit(Xtr, ytr)
        elif mt == 'lasso':
            m = Lasso(alpha=p.get('alpha', 1.0), random_state=42, max_iter=2000).fit(Xtr, ytr)
        elif mt == 'elasticnet':
            m = ElasticNet(alpha=p.get('alpha', 1.0), l1_ratio=p.get('l1_ratio', 0.5),
                           random_state=42, max_iter=2000).fit(Xtr, ytr)
        elif mt == 'lgbm':
            m = lgb.LGBMRegressor(
                learning_rate=p.get('learning_rate', 0.05),
                num_leaves=int(p.get('num_leaves', 31)),
                max_depth=int(p.get('max_depth', 5)),
                n_estimators=int(p.get('n_estimators', 100)),
                random_state=42, verbose=-1).fit(Xtr, ytr)
        elif mt == 'xgboost':
            m = xgb.XGBRegressor(
                learning_rate=p.get('learning_rate', 0.05),
                max_depth=int(p.get('max_depth', 5)),
                n_estimators=int(p.get('n_estimators', 100)),
                subsample=p.get('subsample', 0.8),
                colsample_bytree=p.get('colsample_bytree', 0.8),
                random_state=42, verbosity=0).fit(Xtr, ytr)
        elif mt == 'rf':
            m = RandomForestRegressor(
                n_estimators=int(p.get('n_estimators', 100)),
                max_depth=int(p.get('max_depth', 10)),
                min_samples_split=int(p.get('min_samples_split', 5)),
                min_samples_leaf=int(p.get('min_samples_leaf', 2)),
                random_state=42, n_jobs=-1).fit(Xtr, ytr)
        elif mt == 'gbm':
            m = GradientBoostingRegressor(
                learning_rate=p.get('learning_rate', 0.1),
                n_estimators=int(p.get('n_estimators', 100)),
                max_depth=int(p.get('max_depth', 5)),
                subsample=p.get('subsample', 0.8),
                random_state=42).fit(Xtr, ytr)
        elif mt == 'svr':
            m = LinearSVR(C=p.get('C', 1.0), epsilon=p.get('epsilon', 0.1),
                          max_iter=3000, random_state=42).fit(Xtr, ytr)
        else:
            return np.zeros(len(Xvl))
        return m.predict(Xvl)

    def _train_ridge(self, Xtr, ytr, Xvl, yvl):
        self.model = Ridge(alpha=self.config.params.get('alpha', 1.0), random_state=42).fit(Xtr, ytr)

    def _train_lasso(self, Xtr, ytr, Xvl, yvl):
        self.model = Lasso(alpha=self.config.params.get('alpha', 1.0), random_state=42, max_iter=2000).fit(Xtr, ytr)

    def _train_elasticnet(self, Xtr, ytr, Xvl, yvl):
        p = self.config.params
        self.model = ElasticNet(alpha=p.get('alpha', 1.0), l1_ratio=p.get('l1_ratio', 0.5),
                                random_state=42, max_iter=2000).fit(Xtr, ytr)

    def _train_lgbm(self, Xtr, ytr, Xvl, yvl):
        p  = self.config.params
        kw = dict(learning_rate=p.get('learning_rate', 0.05),
                  num_leaves=int(p.get('num_leaves', 31)),
                  max_depth=int(p.get('max_depth', 5)),
                  n_estimators=int(p.get('n_estimators', 100)),
                  random_state=42, verbose=-1)
        if use_gpu: kw.update(device='gpu', gpu_use_dp=False)
        self.model = lgb.LGBMRegressor(**kw)
        self.model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])

    def _train_xgboost(self, Xtr, ytr, Xvl, yvl):
        p  = self.config.params
        kw = dict(learning_rate=p.get('learning_rate', 0.05),
                  max_depth=int(p.get('max_depth', 5)),
                  n_estimators=int(p.get('n_estimators', 100)),
                  subsample=p.get('subsample', 0.8),
                  colsample_bytree=p.get('colsample_bytree', 0.8),
                  random_state=42, verbosity=0)
        if use_gpu: kw.update(tree_method='gpu_hist', gpu_id=0)
        self.model = xgb.XGBRegressor(**kw)
        self.model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)

    def _train_rf(self, Xtr, ytr, Xvl, yvl):
        p = self.config.params
        self.model = RandomForestRegressor(
            n_estimators=int(p.get('n_estimators', 100)),
            max_depth=int(p.get('max_depth', 10)),
            min_samples_split=int(p.get('min_samples_split', 5)),
            min_samples_leaf=int(p.get('min_samples_leaf', 2)),
            random_state=42, n_jobs=-1).fit(Xtr, ytr)

    def _train_gbm(self, Xtr, ytr, Xvl, yvl):
        p = self.config.params
        self.model = GradientBoostingRegressor(
            learning_rate=p.get('learning_rate', 0.1),
            n_estimators=int(p.get('n_estimators', 100)),
            max_depth=int(p.get('max_depth', 5)),
            subsample=p.get('subsample', 0.8),
            random_state=42).fit(Xtr, ytr)

    def _train_svr(self, Xtr, ytr, Xvl, yvl):
        p = self.config.params
        self.model = LinearSVR(C=p.get('C', 1.0), epsilon=p.get('epsilon', 0.1),
                               max_iter=3000, random_state=42).fit(Xtr, ytr)

    def _train_lstm(self, Xtr, ytr, Xvl, yvl):
        p      = self.config.params
        hidden = int(p.get('hidden_size', 64))
        layers = int(p.get('num_layers', 1))
        lr     = p.get('learning_rate', 0.001)
        epochs = int(p.get('epochs', 30))
        bsize  = int(p.get('batch_size', 512))
        sigma  = self.data_handler.noise_sigma

        self.model = SimpleLSTM(Xtr.shape[2], hidden, layers).to(device)
        opt  = torch.optim.Adam(self.model.parameters(), lr=lr)
        crit = nn.MSELoss()

        for _ in range(epochs):
            self.model.train()
            if sigma > 0:
                noise  = np.random.normal(0, sigma, Xtr.shape).astype(np.float32)
                Xtr_ep = Xtr * (1.0 + noise)
            else:
                Xtr_ep = Xtr
            loader = DataLoader(LSTMDataset(Xtr_ep, ytr), batch_size=bsize, shuffle=True)
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                crit(self.model(bx), by).backward()
                opt.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# 🧬 GENETİK OPERATÖRLER v3.0 (korundu, [J] Epigenetik Miras dahil)
# ============================================================================

class GeneticOperators:

    @staticmethod
    def _rand_alpha(v, intensity):
        return float(np.clip(v * np.random.uniform(0.5**intensity, 2.0**intensity), 0.001, 100.0))

    @staticmethod
    def mutate(config: ModelConfig, intensity: float = 1.0) -> ModelConfig:
        mt  = config.model_type
        new = config.clone()
        p   = new.params

        if mt in ('ridge', 'lasso'):
            p['alpha'] = GeneticOperators._rand_alpha(p['alpha'], intensity)
        elif mt == 'elasticnet':
            p['alpha']    = GeneticOperators._rand_alpha(p['alpha'], intensity)
            p['l1_ratio'] = float(np.clip(p['l1_ratio'] + np.random.uniform(-0.15, 0.15)*intensity, 0, 1))
        elif mt in ('lgbm', 'xgboost', 'gbm'):
            p['learning_rate'] = float(np.clip(
                p['learning_rate'] * np.random.uniform(0.7**intensity, 1.3**intensity), 0.001, 0.3))
            p['max_depth']    = int(np.clip(p['max_depth']    + np.random.choice([-1,0,1])*round(intensity), 2, 12))
            p['n_estimators'] = int(np.clip(p['n_estimators'] + np.random.choice([-20,0,20])*round(intensity), 50, 400))
            if mt == 'lgbm':
                p['num_leaves'] = int(np.clip(p['num_leaves'] + np.random.choice([-5,0,5])*round(intensity), 10, 150))
            if mt in ('xgboost', 'gbm'):
                p['subsample'] = float(np.clip(p.get('subsample',0.8)+np.random.uniform(-0.1,0.1)*intensity, 0.5, 1.0))
            if mt == 'xgboost':
                p['colsample_bytree'] = float(np.clip(p.get('colsample_bytree',0.8)+np.random.uniform(-0.1,0.1)*intensity, 0.5, 1.0))
        elif mt == 'rf':
            p['n_estimators']      = int(np.clip(p['n_estimators']     + np.random.choice([-20,0,20])*round(intensity), 50, 400))
            p['max_depth']         = int(np.clip(p['max_depth']        + np.random.choice([-2,0,2])*round(intensity), 5, 25))
            p['min_samples_split'] = int(np.clip(p['min_samples_split']+ np.random.choice([-1,0,1])*round(intensity), 2, 10))
            p['min_samples_leaf']  = int(np.clip(p['min_samples_leaf'] + np.random.choice([-1,0,1])*round(intensity), 1, 5))
        elif mt == 'svr':
            p['C']       = float(np.clip(p['C']       * np.random.uniform(0.5**intensity, 2.0**intensity), 0.01, 200.0))
            p['epsilon'] = float(np.clip(p['epsilon'] * np.random.uniform(0.7**intensity, 1.3**intensity), 0.01, 1.0))
        elif mt == 'lstm':
            p['hidden_size']   = int(np.clip(p['hidden_size']   + np.random.choice([-16,0,16])*round(intensity), 32, 256))
            p['learning_rate'] = float(np.clip(p['learning_rate'] * np.random.uniform(0.7**intensity, 1.3**intensity), 1e-4, 0.01))
            if np.random.random() < 0.2*intensity:
                p['num_layers'] = int(np.random.choice([1, 2]))
        return new

    @staticmethod
    def epigenetic_crossover(a: ModelConfig, b: ModelConfig) -> ModelConfig:
        """[J] Epigenetik Blok Çaprazlaması."""
        if a.model_type != b.model_type:
            return (a if np.random.random() < 0.5 else b).clone()

        child = a.clone()
        mt    = a.model_type
        blocks = PARAM_BLOCKS.get(mt, [[k] for k in a.params.keys()])

        for block in blocks:
            source = a if np.random.random() < 0.5 else b
            for key in block:
                if key in source.params:
                    child.params[key] = deepcopy(source.params[key])

        child.epigenetic_source = f"{a.model_type}[A×B]"
        return child

    @staticmethod
    def type_mutate(config: ModelConfig) -> ModelConfig:
        types = ['ridge','elasticnet','lgbm','xgboost','rf','gbm','svr','lstm']
        nt    = np.random.choice([t for t in types if t != config.model_type])
        rng   = np.random.default_rng()
        D: Dict[str, Any] = {
            'ridge':      {'alpha': float(np.random.choice([0.01,0.1,1.0,10.0]))},
            'lasso':      {'alpha': float(np.random.choice([0.01,0.1,1.0]))},
            'elasticnet': {'alpha': float(rng.uniform(0.1,2.0)), 'l1_ratio': float(rng.uniform(0.1,0.9))},
            'lgbm':       {'learning_rate': float(rng.uniform(0.01,0.2)),
                           'num_leaves': int(np.random.choice([20,31,50,80])),
                           'max_depth': int(np.random.choice([4,5,6,8])),
                           'n_estimators': int(np.random.choice([100,150,200]))},
            'xgboost':    {'learning_rate': float(rng.uniform(0.01,0.2)),
                           'max_depth': int(np.random.choice([4,5,6,8])),
                           'n_estimators': int(np.random.choice([100,150,200])),
                           'subsample': float(rng.uniform(0.6,1.0)),
                           'colsample_bytree': float(rng.uniform(0.6,1.0))},
            'rf':         {'n_estimators': int(np.random.choice([80,100,150])),
                           'max_depth': int(np.random.choice([8,10,15])),
                           'min_samples_split': int(np.random.choice([2,5,8])),
                           'min_samples_leaf': int(np.random.choice([1,2,3]))},
            'gbm':        {'learning_rate': float(rng.uniform(0.01,0.2)),
                           'n_estimators': int(np.random.choice([80,100,150])),
                           'max_depth': int(np.random.choice([3,4,5])),
                           'subsample': float(rng.uniform(0.6,1.0))},
            'svr':        {'C': float(np.random.choice([0.1,1.0,10.0,50.0])),
                           'epsilon': float(rng.uniform(0.05,0.5))},
            'lstm':       {'hidden_size': int(np.random.choice([32,64,96,128])),
                           'num_layers': int(np.random.choice([1,2])),
                           'learning_rate': float(rng.uniform(0.0005,0.005)),
                           'epochs': int(np.random.choice([15,20,25])),
                           'batch_size': int(np.random.choice([256,512]))},
        }
        return ModelConfig(nt, D[nt])

    @staticmethod
    def socialist_inheritance(weak: ModelConfig,
                               ranked_pop: List[ModelWrapper],
                               weak_rank: int,
                               mutation_intensity: float = 0.5) -> ModelConfig:
        """[E] Kolektif Miras — blok bütünlüğü korunur."""
        donors = [w for w in ranked_pop[:max(1, weak_rank)]
                  if w.config.model_type == weak.model_type]
        if not donors:
            return GeneticOperators.mutate(weak, mutation_intensity)

        ranks  = np.array([r for r, w in enumerate(ranked_pop[:max(1, weak_rank)])
                           if w.config.model_type == weak.model_type], dtype=float)
        w_vals = 1.0 / (ranks + 1.0)
        w_vals = w_vals / (w_vals.sum() + 1e-9)

        child = weak.clone()
        mt    = weak.model_type
        blocks = PARAM_BLOCKS.get(mt, [[k] for k in weak.params.keys()])

        for block in blocks:
            best_donor = donors[0]
            for key in block:
                if key in best_donor.config.params:
                    d_val = float(best_donor.config.params[key]) if isinstance(best_donor.config.params[key], (int,float)) else best_donor.config.params[key]
                    w_val = float(weak.params.get(key, d_val)) if isinstance(weak.params.get(key, d_val), (int,float)) else d_val
                    if isinstance(d_val, float):
                        new_val = 0.5 * w_val + 0.5 * d_val
                        new_val += new_val * np.random.normal(0, 0.10 * mutation_intensity)
                        child.params[key] = float(new_val)
                    elif isinstance(d_val, int):
                        new_val = int(round(0.5 * w_val + 0.5 * d_val))
                        child.params[key] = max(1, new_val)

        best_pms = [d.config.pattern_memory for d in donors
                    if d.config.pattern_memory is not None and d.config.pattern_memory._fitted]
        if best_pms and weak.pattern_memory is not None and weak.pattern_memory._fitted:
            dw = w_vals[:len(best_pms)]
            dw = dw / (dw.sum() + 1e-9)
            child.pattern_memory = weak.pattern_memory.inherit_from(best_pms, dw)

        return child


# ============================================================================
# 🏛️ FRANKENSTEIN COUNCIL v3.2
# ============================================================================

class FrankensteinCouncil:

    MAX_PER_TYPE  = 2
    CORR_THRESH   = 0.95
    CORR_DAMP     = 0.30
    VAR_THRESHOLD = 0.30
    VAR_HARD_MIN  = 0.15

    @classmethod
    def update_niche_fitness(cls, population: List[ModelWrapper],
                              y_true: np.ndarray,
                              noise_threshold: float = 0.0):
        DISCOVERY_EXEMPT_THR = 0.05

        all_preds = [w.predictions.get('val', np.zeros_like(y_true))
                     for w in population]
        for w in population:
            pred   = w.predictions.get('val', np.zeros_like(y_true))
            others = [p for p in all_preds if p is not pred]

            niche_mae, discovery_rate = calculate_niche_fitness(
                pred, others, y_true,
                noise_threshold=noise_threshold,
                discovery_bonus=3.0)

            if discovery_rate > DISCOVERY_EXEMPT_THR:
                var_penalty = 1.0
            else:
                var_penalty = 2.5 if w.pred_variance < cls.VAR_HARD_MIN else (
                              1.5 if w.pred_variance < cls.VAR_THRESHOLD else 1.0)

            time_factor       = 1.0 + np.log1p(w.train_time)
            w.niche_fitness   = niche_mae * time_factor * var_penalty
            w.discovery_rate  = float(discovery_rate)

    @classmethod
    def select_council(cls, population: List[ModelWrapper], k: int = 6) -> List[ModelWrapper]:
        selected  : List[ModelWrapper] = []
        type_count: Dict[str, int]     = {}

        ranked_niche  = sorted(population, key=lambda w: w.niche_fitness)
        ranked_normal = sorted(population, key=lambda w: w.fitness)

        for w in ranked_niche:
            if len(selected) >= 2: break
            if type_count.get(w.config.model_type, 0) < cls.MAX_PER_TYPE:
                if w.pred_variance >= cls.VAR_THRESHOLD:
                    selected.append(w)
                    type_count[w.config.model_type] = type_count.get(w.config.model_type, 0) + 1

        valid_sel  = [w for w in selected if 'val' in w.predictions and len(w.predictions['val']) > 1]
        candidates = [w for w in population if w not in selected]

        if valid_sel and candidates:
            ref = np.column_stack([w.predictions['val'] for w in valid_sel])
            div = []
            for w in candidates:
                if 'val' not in w.predictions or len(w.predictions['val']) <= 1:
                    continue
                if w.pred_variance < cls.VAR_THRESHOLD:
                    continue
                cors = [abs(np.corrcoef(w.predictions['val'], ref[:, j])[0, 1])
                        for j in range(ref.shape[1])]
                div.append((float(np.mean(cors)), w))
            div.sort(key=lambda x: x[0])
            for _, w in div:
                if len(selected) >= k: break
                if type_count.get(w.config.model_type, 0) < cls.MAX_PER_TYPE:
                    selected.append(w)
                    type_count[w.config.model_type] = type_count.get(w.config.model_type, 0) + 1

        for w in ranked_normal:
            if len(selected) >= k: break
            if w not in selected:
                selected.append(w)

        return selected[:k]

    @classmethod
    def optimize_weights(cls, council: List[ModelWrapper],
                          y_true: np.ndarray,
                          l1_lambda: float = 0.02) -> Tuple[np.ndarray, float]:
        from scipy.optimize import nnls

        preds = [w.predictions['val'] for w in council]
        X     = np.column_stack(preds)
        n     = len(council)

        aug_X = np.vstack([X, np.sqrt(l1_lambda) * np.eye(n)])
        aug_y = np.concatenate([y_true, np.zeros(n)])
        try:
            weights, _ = nnls(aug_X, aug_y)
        except Exception:
            weights = np.ones(n)

        weights = np.maximum(weights, 0)
        s = weights.sum()
        weights = weights / s if s > 0 else np.ones(n) / n
        base_mae = mean_absolute_error(y_true, X @ weights)

        mcs = np.zeros(n)
        for i in range(n):
            if n == 1:
                mcs[i] = 1.0; break
            rest = [j for j in range(n) if j != i]
            Xr   = X[:, rest]
            wr   = weights[rest]
            wr   = wr / (wr.sum() + 1e-9)
            mcs[i] = max(0.0, mean_absolute_error(y_true, Xr @ wr) - base_mae)

        mcs = 0.1 + 0.9 * (mcs / mcs.max()) if mcs.max() > 0 else np.ones(n)
        for i, w in enumerate(council):
            w.mcs_score = float(mcs[i])

        niche_bonus = np.array([max(0.5, 1.0 / (w.niche_fitness + 1e-9)) for w in council])
        niche_bonus = niche_bonus / niche_bonus.mean()

        da_reward  = np.array([max(0.5, min(2.0, w.val_da + w.val_rwda)) for w in council])
        var_reward = np.array([max(0.2, min(1.0, w.pred_variance))        for w in council])

        adj = weights * mcs * da_reward * var_reward * niche_bonus
        s   = adj.sum()
        base_weights = adj / s if s > 0 else weights
        base_mae_adj = mean_absolute_error(y_true, X @ base_weights)

        return base_weights, base_mae_adj

    @staticmethod
    def temperature_ensemble_predict(predictions: List[np.ndarray],
                                      base_weights: np.ndarray,
                                      vol_regime_series: np.ndarray) -> np.ndarray:
        n = min(min(len(p) for p in predictions), len(vol_regime_series))
        result = np.zeros(n)
        for t in range(n):
            vol_t   = float(vol_regime_series[t])
            w_t     = temperature_ensemble(
                predictions, base_weights, vol_t,
                T_min=0.20, T_max=1.0,
                vol_low=0.3, vol_high=1.5)
            result[t] = sum(w * p[t] for w, p in zip(w_t, predictions))
        return result

    @staticmethod
    def correlation_penalty(predictions: List[np.ndarray], lam: float = 1.5) -> float:
        if len(predictions) < 2:
            return 0.0
        cors = [abs(np.corrcoef(predictions[i], predictions[j])[0, 1])
                for i, j in itertools.combinations(range(len(predictions)), 2)]
        return lam * float(np.mean(cors))


# ============================================================================
# 🧬 EVRİM ENJİNİ v3.2
# ============================================================================

class ZorbaEvolution:

    STAGNATION_LIMIT    = 3
    HYPER_RESET_FRAC    = 0.40
    TYPE_MUTATE_PROB    = 0.10
    CROSSOVER_PROB      = 0.25
    INHERITANCE_PROB    = 0.35
    INHERITANCE_CUTOFF  = 0.60

    def __init__(self, initial_population : List[ModelConfig],
                 data_handler            : TimeSeriesDataHandler,
                 population_size         : int = 20,
                 council_size            : int = 6):
        self.initial_population = initial_population
        self.data_handler       = data_handler
        self.population_size    = population_size
        self.council_size       = council_size

        self.generation  = 0
        self.population  : List[ModelWrapper] = []
        self.history     : List[Dict]         = []

        self.best_single_wrapper   : Optional[ModelWrapper] = None
        self.best_single_mae       = np.inf
        self.best_ensemble_weights : Optional[np.ndarray]  = None
        self.best_ensemble_mae     = np.inf
        self.best_ensemble_raw_mae = np.inf
        self.best_council          : List[ModelWrapper]     = []

        self._stagnation_counter = 0
        self._last_best_mae      = np.inf

    def _make(self, cfg: ModelConfig) -> ModelWrapper:
        cfg.pattern_memory = self.data_handler.pattern_memory
        return ModelWrapper(cfg, self.data_handler)

    def initialize_population(self):
        print("\n" + "="*70)
        print("🧬 BAŞLATMA — v3.2 ZORBA FSN + NICHE + EPİGENETİK")
        print("="*70)
        for cfg in self.initial_population:
            self.population.append(self._make(cfg))
        print(f"   Popülasyon   : {len(self.population)}  Council: {self.council_size}")
        print(f"   Gürültü σ    : {self.data_handler.noise_sigma}")
        print(f"   [I]  NİŞ SKORU + DISCOVERY RATE")
        print(f"   [J]  EPİGENETİK BLOK ÇAPRAZLAMA")
        print(f"   [K]  FİNANSAL DURUM UZAYI MAHALLESİ (FSN)")
        print(f"        4-Rejim + Mahalanobis + CUSUM Break + ACF-Aware")
        print(f"   [L]  SICAKLIK KONTROLLÜ SOFTMAX ENSEMBLE (T_min=0.20)")

    def evaluate_population(self):
        print(f"\n{'='*70}")
        print(f"⚡ GEN {self.generation} — {len(self.population)} MODEL")
        print(f"{'='*70}")
        for idx, w in enumerate(self.population):
            if w.model is None:
                print(f"   [{idx+1:2d}/{len(self.population)}] {w.config.model_type:10s} eğitiliyor...")
                w.train()
                flat = " 📉FLAT" if w.pred_variance < FrankensteinCouncil.VAR_HARD_MIN else (
                       " ⚠️LOW"  if w.pred_variance < FrankensteinCouncil.VAR_THRESHOLD else "")
                dr_tag = f" 🎯DR={w.discovery_rate:.2f}" if w.discovery_rate > 0.05 else ""
                print(f"         MAE={w.val_mae:.4f}  DA={w.val_da:.3f}  "
                      f"RWDA={w.val_rwda:.3f}  Var={w.pred_variance:.2f}"
                      f"{flat}{dr_tag}  Fit={w.fitness:.4f}  t={w.train_time:.1f}s")

        _, y_val = self.data_handler.get_data('ridge', 'val')
        FrankensteinCouncil.update_niche_fitness(
            self.population, y_val, self.data_handler.noise_threshold)

        self.population.sort(key=lambda w: w.niche_fitness)
        for rank, w in enumerate(self.population):
            w.rank = rank

        best = self.population[0]
        if best.val_mae < self.best_single_mae:
            self.best_single_mae     = best.val_mae
            self.best_single_wrapper = best.clone()
            self.best_single_wrapper.model             = best.model
            self.best_single_wrapper.predictions       = dict(best.predictions)
            self.best_single_wrapper.predictions_price = dict(best.predictions_price)
            self.best_single_wrapper.val_mae    = best.val_mae
            self.best_single_wrapper.val_da     = best.val_da
            self.best_single_wrapper.val_rwda   = best.val_rwda
            self.best_single_wrapper.fitness    = best.fitness
            self.best_single_wrapper.pred_variance = best.pred_variance
            print(f"\n   🏆 Yeni En İyi → {best.config.model_type}  "
                  f"MAE={best.val_mae:.4f}  DA={best.val_da:.3f}  "
                  f"RWDA={best.val_rwda:.3f}  NişFit={best.niche_fitness:.4f}")
        clear_memory()

    def run_council(self) -> Tuple[np.ndarray, float, List[ModelWrapper], float]:
        print(f"\n{'='*70}\n🏛️  COUNCIL v3.2 — GEN {self.generation}\n{'='*70}")

        council  = FrankensteinCouncil.select_council(self.population, k=self.council_size)
        _, y_val = self.data_handler.get_data('ridge', 'val')
        thr      = self.data_handler.noise_threshold
        vol_val  = self.data_handler.get_vol_regime('val')

        for w in council:
            if 'val' not in w.predictions:
                w.predictions['val'] = w.predict('val')

        preds = [w.predictions['val'] for w in council]

        if len(preds) > 1:
            cm        = np.corrcoef(np.row_stack(preds))
            mean_corr = float(np.mean(np.abs(cm[np.triu_indices(len(preds), k=1)])))
        else:
            mean_corr = 1.0

        base_weights, base_mae = FrankensteinCouncil.optimize_weights(council, y_val)

        n_min     = min(len(p) for p in preds)
        n_vol     = len(vol_val)
        n_ens     = min(n_min, n_vol)
        preds_cut = [p[:n_ens] for p in preds]
        vol_cut   = vol_val[:n_ens]

        ens_delta = FrankensteinCouncil.temperature_ensemble_predict(
            preds_cut, base_weights, vol_cut)
        ens_mae   = mean_absolute_error(y_val[:n_ens], ens_delta)
        ens_da    = directional_accuracy(y_val[:n_ens], ens_delta, noise_threshold=thr)
        ens_rwda  = return_weighted_da(y_val[:n_ens],   ens_delta, noise_threshold=thr)

        cor_penalty     = FrankensteinCouncil.correlation_penalty(preds, lam=1.5)
        penalised_score = ens_mae + cor_penalty

        print(f"\n   Council üyeleri ({len(council)}):")
        for i, (w, wt) in enumerate(zip(council, base_weights)):
            tag    = "⭐" if wt > 0.15 else "  "
            ftag   = "📉" if w.pred_variance < FrankensteinCouncil.VAR_HARD_MIN else (
                     "⚠️" if w.pred_variance < FrankensteinCouncil.VAR_THRESHOLD else "  ")
            nf_tag = "🎯" if w.niche_fitness < w.fitness * 0.9 else "  "
            dr_tag = f"DR={w.discovery_rate:.2f}" if w.discovery_rate > 0.05 else "DR=—  "
            print(f"   {tag}{ftag}{nf_tag} {i+1}. {w.config.model_type:10s}  "
                  f"MAE={w.val_mae:.4f}  DA={w.val_da:.3f}  RWDA={w.val_rwda:.3f}  "
                  f"MCS={w.mcs_score:.3f}  Var={w.pred_variance:.2f}  "
                  f"{dr_tag}  NişFit={w.niche_fitness:.4f}  w={wt:.4f}")

        bst = self.population[0].val_mae
        imp = (bst - ens_mae) / bst * 100 if bst > 0 else 0
        print(f"\n   En İyi Tek MAE : {bst:.4f}")
        print(f"   [L] Temp. Ens. MAE : {ens_mae:.4f}  (Δ {imp:+.2f}%)")
        print(f"   Ensemble DA    : {ens_da:.4f}  RWDA: {ens_rwda:.4f}")
        print(f"   Corr cezası    : +{cor_penalty:.4f}  Cezalı: {penalised_score:.4f}")
        print(f"   Ort. çift ρ    : {mean_corr:.4f}")

        if penalised_score < self.best_ensemble_mae:
            self.best_ensemble_mae     = penalised_score
            self.best_ensemble_raw_mae = ens_mae
            self.best_ensemble_weights = base_weights
            self.best_council          = council
            print(f"   🏆 Yeni En İyi! raw={ens_mae:.4f}  DA={ens_da:.4f}  "
                  f"cezalı={penalised_score:.4f}")

        return base_weights, ens_mae, council, mean_corr

    def _tournament(self, k: int = 3) -> ModelWrapper:
        idx = np.random.choice(len(self.population), size=min(k, len(self.population)), replace=False)
        return min([self.population[i] for i in idx], key=lambda w: w.niche_fitness)

    def evolve_population(self):
        print(f"\n{'='*70}\n🧬 EVRİM → GEN {self.generation+1}\n{'='*70}")

        if self.best_ensemble_mae < self._last_best_mae - 1e-4:
            self._stagnation_counter = 0
            self._last_best_mae      = self.best_ensemble_mae
        else:
            self._stagnation_counter += 1

        hyper = self._stagnation_counter >= self.STAGNATION_LIMIT
        if hyper:
            print(f"   ⚠️  HYPER-MUTASYON (durgunluk={self._stagnation_counter})")
            self._stagnation_counter = 0

        intensity = 2.0 if hyper else 1.0
        new_pop: List[ModelWrapper] = []

        elite = 0
        for w in sorted(self.population, key=lambda x: x.niche_fitness):
            if elite >= 2: break
            if w.pred_variance >= FrankensteinCouncil.VAR_THRESHOLD:
                c               = w.clone()
                c.model         = w.model
                c.val_mae       = w.val_mae
                c.val_da        = w.val_da
                c.val_rwda      = w.val_rwda
                c.fitness       = w.fitness
                c.niche_fitness = w.niche_fitness
                c.train_time    = w.train_time
                c.pred_variance = w.pred_variance
                c.predictions   = dict(w.predictions)
                c.predictions_price = dict(w.predictions_price)
                new_pop.append(c)
                elite += 1

        if len(new_pop) == 0:
            for w in self.population[:2]:
                c               = w.clone()
                c.model         = w.model
                c.val_mae       = w.val_mae
                c.val_da        = w.val_da
                c.val_rwda      = w.val_rwda
                c.fitness       = w.fitness
                c.niche_fitness = w.niche_fitness
                c.train_time    = w.train_time
                c.pred_variance = w.pred_variance
                c.predictions   = dict(w.predictions)
                c.predictions_price = dict(w.predictions_price)
                new_pop.append(c)

        if hyper:
            n_reset = int(self.population_size * self.HYPER_RESET_FRAC)
            for _ in range(n_reset):
                new_pop.append(self._make(GeneticOperators.type_mutate(self._tournament().config)))

        inheritance_cutoff = int(len(self.population) * self.INHERITANCE_CUTOFF)
        weak_candidates    = self.population[inheritance_cutoff:]

        inherit_count = 0
        cross_count   = 0
        while len(new_pop) < self.population_size:
            p1   = self._tournament()
            rand = np.random.random()

            if rand < self.CROSSOVER_PROB:
                p2  = self._tournament()
                cfg = GeneticOperators.epigenetic_crossover(p1.config, p2.config)
                cfg = GeneticOperators.mutate(cfg, intensity * 0.3)
                print(f"   [J] Epigenetic {p1.config.model_type} × {p2.config.model_type} → {cfg.model_type}")
                cross_count += 1

            elif rand < self.CROSSOVER_PROB + self.INHERITANCE_PROB and weak_candidates:
                weak_w = weak_candidates[np.random.randint(len(weak_candidates))]
                cfg    = GeneticOperators.socialist_inheritance(
                    weak_w.config, self.population, weak_w.rank,
                    mutation_intensity=intensity * 0.5)
                print(f"   [E] Miras: rank{weak_w.rank} {weak_w.config.model_type} → {cfg.model_type}")
                inherit_count += 1

            elif rand < self.CROSSOVER_PROB + self.INHERITANCE_PROB + self.TYPE_MUTATE_PROB:
                cfg = GeneticOperators.type_mutate(p1.config)
                print(f"   🔀 {p1.config.model_type} → {cfg.model_type}")

            else:
                cfg = GeneticOperators.mutate(p1.config, intensity)

            new_pop.append(self._make(cfg))

        self.population = new_pop[:self.population_size]
        self.generation += 1
        print(f"   ✅ Gen {self.generation}: {len(self.population)} model | "
              f"{cross_count} epigenetik | {inherit_count} miras")
        clear_memory()

    def run_evolution(self, n_generations: int = 30):
        print("\n" + "="*70)
        print("🚀 ZORBA EVRİM v3.2 — FSN + NİŞ + EPİGENETİK + SICAKLIK")
        print("="*70)
        self.initialize_population()

        for gen in range(n_generations):
            self.evaluate_population()
            weights, ens_mae, council, mean_corr = self.run_council()

            thr       = self.data_handler.noise_threshold
            _, y_val  = self.data_handler.get_data('ridge', 'val')
            mean_da   = float(np.mean([
                directional_accuracy(y_val, w.predictions.get('val', np.zeros_like(y_val)),
                                     noise_threshold=thr)
                for w in self.population if 'val' in w.predictions]))
            mean_rwda = float(np.mean([
                return_weighted_da(y_val, w.predictions.get('val', np.zeros_like(y_val)),
                                   noise_threshold=thr)
                for w in self.population if 'val' in w.predictions]))

            self.history.append({
                'generation'       : self.generation,
                'best_single_mae'  : self.population[0].val_mae,
                'best_single_type' : self.population[0].config.model_type,
                'ensemble_mae'     : ens_mae,
                'best_ens_mae'     : self.best_ensemble_raw_mae,
                'weights'          : weights.copy(),
                'council_types'    : [w.config.model_type for w in council],
                'council_mcs'      : [w.mcs_score         for w in council],
                'council_variance' : [w.pred_variance      for w in council],
                'council_da'       : [w.val_da             for w in council],
                'council_niche'    : [w.niche_fitness       for w in council],
                'all_maes'         : [w.val_mae            for w in self.population],
                'all_fitness'      : [w.fitness            for w in self.population],
                'all_niche'        : [w.niche_fitness       for w in self.population],
                'all_types'        : [w.config.model_type  for w in self.population],
                'all_variances'    : [w.pred_variance       for w in self.population],
                'all_da'           : [w.val_da              for w in self.population],
                'stagnation'       : self._stagnation_counter,
                'mean_train_time'  : float(np.mean([w.train_time for w in self.population])),
                'mean_corr'        : mean_corr,
                'pop_mean_da'      : mean_da,
                'pop_mean_rwda'    : mean_rwda,
            })

            tc: Dict[str, int] = {}
            for w in self.population:
                tc[w.config.model_type] = tc.get(w.config.model_type, 0) + 1
            print(f"\n   🧬 Çeşitlilik: {tc}")
            print(f"   📊 Gen {self.generation} | DA mean: {mean_da:.4f}  RWDA: {mean_rwda:.4f}")
            print(f"   📊 En İyi Tek MAE  : {self.best_single_mae:.4f} "
                  f"({self.best_single_wrapper.config.model_type if self.best_single_wrapper else '-'})")
            print(f"   📊 En İyi Ensemble : {self.best_ensemble_raw_mae:.4f}")

            if gen < n_generations - 1:
                self.evolve_population()

        print("\n" + "="*70 + "\n🏁 EVRİM TAMAMLANDI\n" + "="*70)

    def test_evaluation(self) -> Dict:
        print("\n" + "="*70 + "\n🧪 TEST SET DEĞERLENDİRME — v3.2\n" + "="*70)

        _, y_test_delta = self.data_handler.get_data('ridge', 'test')
        y_test_price    = self.data_handler.get_true_prices('test')
        thr             = self.data_handler.noise_threshold
        vol_test        = self.data_handler.get_vol_regime('test')

        bsw        = self.best_single_wrapper
        yps_delta  = bsw.predict('test')
        mae_s_delta = mean_absolute_error(y_test_delta, yps_delta)
        da_s        = directional_accuracy(y_test_delta, yps_delta, noise_threshold=thr)
        rwda_s      = return_weighted_da(y_test_delta,   yps_delta, noise_threshold=thr)

        yps_price   = bsw.predict_price('test')
        n           = min(len(y_test_price), len(yps_price))
        mae_s_price = mean_absolute_error(y_test_price[:n], yps_price[:n])
        r2_s        = r2_score(y_test_price[:n], yps_price[:n])

        print(f"\n🏆 En İyi Tek ({bsw.config.model_type}):")
        print(f"   Delta MAE={mae_s_delta:.4f}  DA={da_s:.4f}  RWDA={rwda_s:.4f}")
        print(f"   Price MAE={mae_s_price:.4f}  R²={r2_s:.4f}")

        cps_delta = [w.predict('test') for w in self.best_council]
        n_min     = min(len(p) for p in cps_delta)
        n_vol_t   = len(vol_test)
        n_ens     = min(n_min, n_vol_t, len(y_test_delta))
        cps_cut   = [p[:n_ens] for p in cps_delta]
        vol_cut   = vol_test[:n_ens]

        ype_delta = FrankensteinCouncil.temperature_ensemble_predict(
            cps_cut, self.best_ensemble_weights, vol_cut)

        mae_e_delta = mean_absolute_error(y_test_delta[:n_ens], ype_delta)
        da_e        = directional_accuracy(y_test_delta[:n_ens], ype_delta, noise_threshold=thr)
        rwda_e      = return_weighted_da(y_test_delta[:n_ens],   ype_delta, noise_threshold=thr)

        ype_price   = self.data_handler.to_price_level_anchored(ype_delta, 'test')
        mae_e_price = mean_absolute_error(y_test_price[:n_ens], ype_price[:n_ens])
        r2_e        = r2_score(y_test_price[:n_ens], ype_price[:n_ens])

        print(f"\n🏛️  [L] Temp. Ensemble ({len(self.best_council)} üye):")
        print(f"   Delta MAE={mae_e_delta:.4f}  DA={da_e:.4f}  RWDA={rwda_e:.4f}")
        print(f"   Price MAE={mae_e_price:.4f}  R²={r2_e:.4f}")
        for w, wt in zip(self.best_council, self.best_ensemble_weights):
            print(f"   {w.config.model_type:10s}  w={wt:.4f}  DA={w.val_da:.3f}  "
                  f"RWDA={w.val_rwda:.3f}  Niş={w.niche_fitness:.4f}  Var={w.pred_variance:.2f}")

        print(f"\n   DA iyileşmesi  : {(da_e - da_s)*100:+.2f}pp  ({da_s:.4f} → {da_e:.4f})")
        if mae_s_price > 0:
            print(f"   Price MAE iyil : {(mae_s_price-mae_e_price)/mae_s_price*100:.2f}%")

        return dict(
            single  =dict(mae_delta=mae_s_delta, mae_price=mae_s_price, r2=r2_s,  da=da_s,  rwda=rwda_s),
            ensemble=dict(mae_delta=mae_e_delta, mae_price=mae_e_price, r2=r2_e,  da=da_e,  rwda=rwda_e),
        )


# ============================================================================
# 📊 GÖRSELLEŞTİRİCİ v3.2
# ============================================================================

class FrankensteinVisualizer:

    @staticmethod
    def plot(history: List[Dict],
             data_handler: TimeSeriesDataHandler,
             output_path: str = 'frankenstein_v32_evolution.png'):
        fig = plt.figure(figsize=(28, 22))
        fig.suptitle(
            '🧬 Frankenstein v3.2 — FSN + Niş + Epigenetik + Sıcaklık\n'
            '[I] Niş Skoru  [J] Epigenetik Blok  '
            '[K] Finansal DSN (4-Rejim+Mahalanobis+CUSUM+ACF)  [L] Sıcaklık Ensemble',
            fontsize=12, fontweight='bold')
        gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.60, wspace=0.38)

        gens    = [h['generation']          for h in history]
        singles = [h['best_single_mae']     for h in history]
        ensembs = [h['ensemble_mae']        for h in history]
        best_e  = [h['best_ens_mae']        for h in history]
        pop_da  = [h.get('pop_mean_da',  0.5) for h in history]
        pop_rda = [h.get('pop_mean_rwda',0.5) for h in history]
        all_niche = [np.min(h.get('all_niche',[np.inf])) for h in history]

        ax = fig.add_subplot(gs[0, :2])
        ax.plot(gens, singles, 'o-',  label='En İyi Tek MAE (Δ)', lw=2)
        ax.plot(gens, ensembs, 's-',  label='[L] Temp. Ens. MAE', lw=2)
        ax.plot(gens, best_e,  '^--', label='En İyi Ensemble',     lw=2, color='green')
        ax.set_title('Δ-Hedef MAE Evrimi (v3.2)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax_nf = fig.add_subplot(gs[0, 2])
        ax_nf.plot(gens, all_niche, 'D-', color='darkred', lw=2, label='Min Niş Fit')
        ax_nf.set_title('[I] Niş Fitness\n(Aykırı Doğruluk)')
        ax_nf.legend(fontsize=8); ax_nf.grid(alpha=0.3)

        ax_da = fig.add_subplot(gs[0, 3])
        ax_da.plot(gens, pop_da,  's-',  color='blue', lw=2,   label='Mean DA')
        ax_da.plot(gens, pop_rda, 'D--', color='navy', lw=1.5, label='Mean RWDA')
        ax_da.axhline(0.50, color='red',    linestyle=':', lw=1.5, label='Rastgele')
        ax_da.axhline(0.52, color='orange', linestyle=':', lw=1.5, label='Hedef (0.52)')
        ax_da.axhline(0.55, color='yellow', linestyle=':', lw=1.0, label='İyi')
        ax_da.axhline(0.60, color='green',  linestyle=':', lw=1.0, label='Mükemmel')
        ax_da.set_title('DA Evrimi (Hedef: >0.52)')
        ax_da.legend(fontsize=6); ax_da.grid(alpha=0.3); ax_da.set_ylim(0.40, 0.75)

        ax3 = fig.add_subplot(gs[1, :2])
        all_types  = sorted(set(t for h in history for t in h['all_types']))
        colors     = plt.cm.tab10(np.linspace(0, 1, len(all_types)))
        type_color = {t: c for t, c in zip(all_types, colors)}
        for t in all_types:
            ax3.plot(gens, [h['all_types'].count(t) for h in history],
                     label=t, color=type_color[t], lw=1.5)
        ax3.set_title('[J] Popülasyon Bileşimi')
        ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(gens, [np.std(h['all_maes'])  for h in history], label='Std',   lw=2)
        ax4.plot(gens, [np.max(h['all_maes'])-np.min(h['all_maes']) for h in history],
                 label='Aralık', lw=2, linestyle='--')
        ax4.set_title('MAE Çeşitliliği'); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

        ax_var = fig.add_subplot(gs[1, 3])
        ax_var.plot(gens, [np.mean(h.get('all_variances',[1])) for h in history],
                    's-', color='teal', lw=2, label='Ort. Var')
        ax_var.plot(gens, [np.min(h.get('all_variances', [1])) for h in history],
                    'D--', color='red', lw=1.5, label='Min Var')
        ax_var.axhline(FrankensteinCouncil.VAR_THRESHOLD, color='orange',
                       linestyle=':', label='Düz eşik (0.30)')
        ax_var.axhline(FrankensteinCouncil.VAR_HARD_MIN, color='red',
                       linestyle='--', label='Sert ceza (0.15)')
        ax_var.set_title('Tahmin Varyansı'); ax_var.legend(fontsize=6); ax_var.grid(alpha=0.3)

        ax5 = fig.add_subplot(gs[2, :2])
        ft  = history[-1]['council_types']
        fw  = history[-1]['weights']
        ax5.bar(range(len(fw)), fw, color=[type_color.get(t,'grey') for t in ft])
        ax5.set_xticks(range(len(ft)))
        ax5.set_xticklabels([f"{t}\n{i+1}" for i, t in enumerate(ft)], fontsize=8)
        ax5.set_title('[L] Final Council Ağırlıkları')
        ax5.grid(alpha=0.3, axis='y')

        ax_mcs = fig.add_subplot(gs[2, 2])
        lm = history[-1].get('council_niche', [])
        lt = history[-1]['council_types']
        if lm:
            ax_mcs.bar(range(len(lm)), lm, color=[type_color.get(t,'grey') for t in lt])
            ax_mcs.set_xticks(range(len(lt)))
            ax_mcs.set_xticklabels([f"{t[:4]}\n{i+1}" for i,t in enumerate(lt)], fontsize=7)
        ax_mcs.set_title('[I] Council Niş Fitness'); ax_mcs.grid(alpha=0.3, axis='y')

        ax_cda = fig.add_subplot(gs[2, 3])
        lcd = history[-1].get('council_da', [])
        if lcd:
            ax_cda.bar(range(len(lcd)), lcd, color=[type_color.get(t,'grey') for t in lt])
            ax_cda.axhline(0.5,  color='red',    linestyle=':', lw=1.5)
            ax_cda.axhline(0.52, color='orange', linestyle=':', lw=1.0)
            ax_cda.axhline(0.55, color='green',  linestyle=':', lw=1.0)
            ax_cda.set_xticks(range(len(lt)))
            ax_cda.set_xticklabels([f"{t[:4]}\n{i+1}" for i,t in enumerate(lt)], fontsize=7)
        ax_cda.set_title('Final Council DA'); ax_cda.grid(alpha=0.3, axis='y')

        ax_pm = fig.add_subplot(gs[3, :2])
        pm = data_handler.pattern_memory
        if pm._fitted:
            ax_pm.bar(range(pm.n_bins), pm.bin_counts, color='steelblue', alpha=0.7)
            ax_pm.set_xticks(range(pm.n_bins))
            ax_pm.set_xticklabels([f"K{i+1}\n({pm.bin_centers[i]:.2f})"
                                    for i in range(pm.n_bins)], fontsize=7)
            ax_pm.set_title('[F] PatternMemory: Kova Dağılımı')
        ax_pm.grid(alpha=0.3, axis='y')

        ax_tm = fig.add_subplot(gs[3, 2:])
        if pm._fitted:
            im = ax_tm.imshow(pm.transition_mat, cmap='Blues', aspect='auto', vmin=0, vmax=0.3)
            ax_tm.set_title('[F] Transition Matrix')
            ax_tm.set_xlabel('Sonraki Kova'); ax_tm.set_ylabel('Mevcut Kova')
            plt.colorbar(im, ax=ax_tm, fraction=0.046)

        # [K→FSN] Rejim dağılımı
        ax_fsn = fig.add_subplot(gs[4, :2])
        fn = data_handler.financial_neighborhood
        if fn._fitted and fn._train_regimes is not None:
            regime_labels = ['Trend↑', 'Trend↓', 'MeanRev', 'Volatile']
            regime_colors = ['green', 'red', 'blue', 'orange']
            counts = [(fn._train_regimes == i).sum() for i in range(4)]
            ax_fsn.bar(regime_labels, counts, color=regime_colors, alpha=0.7)
            ax_fsn.set_title(
                f'[K] FSN v3.2 — 4-Rejim Dağılımı\n'
                f'ACF Momentum: {fn._acf_momentum:.3f} | '
                f'Break Eşiği: {fn._break_thr:.3f} | '
                f'Bandwidth: {fn._bandwidth:.4f}\n'
                f'K_pool={fn.K_REGIME_POOL} K_final={fn.K_FINAL} | '
                f'Mahalanobis + CUSUM + Vol-Normalized Decay')
        ax_fsn.grid(alpha=0.3, axis='y')

        ax_inh = fig.add_subplot(gs[4, 2:])
        last_da  = history[-1].get('all_da', [])
        first_da = history[0].get('all_da', [])
        if last_da and first_da:
            bins = np.linspace(0.4, 0.75, 20)
            ax_inh.hist(first_da, bins=bins, alpha=0.5, label='Gen 0', color='red')
            ax_inh.hist(last_da,  bins=bins, alpha=0.5,
                        label=f'Gen {history[-1]["generation"]}', color='blue')
            ax_inh.axvline(0.5,  color='black', linestyle=':', lw=1.5)
            ax_inh.axvline(0.52, color='orange', linestyle=':', lw=1.5, label='Hedef 0.52')
            ax_inh.axvline(0.55, color='green',  linestyle=':', lw=1.5)
        ax_inh.set_title('DA Dağılımı: Gen 0 vs Son Gen')
        ax_inh.legend(fontsize=7); ax_inh.grid(alpha=0.3)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Kaydedildi: {output_path}")
        return fig


# ============================================================================
# 🎯 MAIN — v3.2
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🧬 FRANKENSTEIN CO-EVOLUTIONARY FORECASTING v3.2")
    print("   Zorba — BTC Kripto Dinamikleri | %70/%15/%15 Yüzde Bazlı Split")
    print()
    print("   FIX-5: Yıl bazlı split KALDIRILDI → %70/%15/%15 iloc bazlı")
    print("   FIX-6: DSN → FSN (Finansal Durum Uzayı Mahallesi)")
    print()
    print("   [K] FSN Bileşenleri:")
    print("       • 4-Durum Rejim Detektörü (Trend↑/↓ + MeanRev + Volatile)")
    print("       • Mahalanobis Mesafesi (korelasyon ve ölçek aware)")
    print("       • CUSUM Structural Break Detector")
    print("       • Vol-Normalized Hiperbolik Temporal Decay")
    print("       • ACF-Aware Prediction (momentum vs mean-rev)")
    print()
    print("   VERİ SIZINTISI: SIFIR")
    print("="*70)

    data_handler = TimeSeriesDataHandler(
        filepath             = 'btc_hourly.csv',   # BTC saatlik veri (2020-2026)
        sequence_length      = 24,
        target_col           = 'close',             # CSV'deki fiyat sütunu
        train_frac           = 0.70,
        val_frac             = 0.15,
        # test_frac otomatik = 0.15
        noise_sigma          = 0.01,
        noise_threshold_pct  = 0.30,
        n_bins               = 10,
    )
    data_handler.load_and_prepare()

    initial_population = [
        ModelConfig('ridge',      {'alpha': 0.1}),
        ModelConfig('ridge',      {'alpha': 10.0}),
        ModelConfig('elasticnet', {'alpha': 0.5,  'l1_ratio': 0.3}),
        ModelConfig('elasticnet', {'alpha': 1.0,  'l1_ratio': 0.7}),
        ModelConfig('lgbm', {'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5, 'n_estimators': 100}),
        ModelConfig('lgbm', {'learning_rate': 0.10, 'num_leaves': 60, 'max_depth': 7, 'n_estimators': 200}),
        ModelConfig('lgbm', {'learning_rate': 0.02, 'num_leaves': 20, 'max_depth': 4, 'n_estimators': 150}),
        ModelConfig('xgboost', {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100,
                                 'subsample': 0.8, 'colsample_bytree': 0.8}),
        ModelConfig('xgboost', {'learning_rate': 0.10, 'max_depth': 7, 'n_estimators': 200,
                                 'subsample': 0.7, 'colsample_bytree': 0.7}),
        ModelConfig('xgboost', {'learning_rate': 0.02, 'max_depth': 4, 'n_estimators': 150,
                                 'subsample': 0.9, 'colsample_bytree': 0.9}),
        ModelConfig('rf', {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}),
        ModelConfig('rf', {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1}),
        ModelConfig('gbm', {'learning_rate': 0.10, 'n_estimators': 100, 'max_depth': 4, 'subsample': 0.8}),
        ModelConfig('gbm', {'learning_rate': 0.05, 'n_estimators': 150, 'max_depth': 5, 'subsample': 0.9}),
        ModelConfig('svr', {'C': 1.0,  'epsilon': 0.1}),
        ModelConfig('svr', {'C': 10.0, 'epsilon': 0.05}),
        ModelConfig('lstm', {'hidden_size': 64,  'num_layers': 1, 'learning_rate': 0.001,
                              'epochs': 30, 'batch_size': 512}),
        ModelConfig('lstm', {'hidden_size': 128, 'num_layers': 2, 'learning_rate': 0.0005,
                              'epochs': 25, 'batch_size': 256}),
    ]

    engine = ZorbaEvolution(
        initial_population = initial_population,
        data_handler       = data_handler,
        population_size    = 20,
        council_size       = 6,
    )
    engine.run_evolution(n_generations=30)
    results = engine.test_evaluation()

    FrankensteinVisualizer.plot(engine.history, data_handler)

    print("\n" + "="*70 + "\n📝 FİNAL ÖZET — v3.2\n" + "="*70)
    s, e = results['single'], results['ensemble']
    print(f"   Tek Model  → Price MAE={s['mae_price']:.4f}  "
          f"DA={s['da']:.4f}  RWDA={s['rwda']:.4f}  R²={s['r2']:.4f}")
    print(f"   [L] Ens.   → Price MAE={e['mae_price']:.4f}  "
          f"DA={e['da']:.4f}  RWDA={e['rwda']:.4f}  R²={e['r2']:.4f}")
    print(f"   DA iyileş  : {(e['da']-s['da'])*100:+.2f}pp")
    print(f"   Hedef DA   : >0.52 | Ulaşıldı mı: {'✅' if e['da'] > 0.52 else '❌'}")

    fn = data_handler.financial_neighborhood
    if fn._fitted:
        regime_labels = ['Trend↑', 'Trend↓', 'MeanRev', 'Volatile']
        counts = [(fn._train_regimes == i).sum() for i in range(4)]
        print(f"\n   [K] FSN v3.2 Özeti:")
        print(f"       Rejim Dağılımı: {dict(zip(regime_labels, counts))}")
        print(f"       ACF Momentum: {fn._acf_momentum:.3f} "
              f"({'momentum' if fn._acf_momentum > 0.5 else 'mean-rev'})")
        print(f"       CUSUM Break Eşiği: {fn._break_thr:.4f}")
        print(f"       Mahalanobis Bandwidth: {fn._bandwidth:.4f}")
        print(f"       K_pool={fn.K_REGIME_POOL} K_final={fn.K_FINAL}")

    print("\n" + "="*70)
    print("✅ FRANKENSTEIN v3.2 — ZORBA TAMAMLANDI")
    print("   FIX-5: %70/%15/%15 Yüzde Bazlı Split")
    print("   FIX-6: FSN — 4-Rejim+Mahalanobis+CUSUM+ACF+Vol-Decay")
    print("   FIX-1..4: v3.1'den korundu")
    print("   VERİ SIZINTISI: SIFIR | AKADEMİK STANDART: TAM")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()