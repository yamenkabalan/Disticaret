# Turkey Export Prediction — Linear Algebra Final Project 2026

**Course:** Linear Algebra  
**Author:** Mohammad Yamen Kabalan  
**Topic:** Turkey export forecasting using Linear Algebra methods (2000–2021)

---

## Overview

This project develops a **Linear Algebra-based Decision Support System** for predicting Turkey's annual export performance using five macroeconomic variables. Three methods are implemented and compared:

| Method | R² | RMSE | MAE |
|---|---|---|---|
| Normal Equation (LSQ) | 0.9981 | 3.70B USD | 2.96B USD |
| SVD Pseudo-Inverse | 0.9981 | 3.70B USD | 2.96B USD |
| PCA Regression (k=3) | 0.8657 | 31.21B USD | 25.38B USD |

> LSQ and SVD produce numerically identical results: ‖β_lstsq − β_svd‖ = 3.04 × 10⁻¹³

---

## Methods

- **Normal Equation** — Least Squares closed-form: β* = (XᵀX)⁻¹ Xᵀy, cond(XᵀX) = 61.62
- **SVD Pseudo-Inverse** — Moore-Penrose: β* = VΣ⁺Uᵀy, condition number κ(X) = 7.85
- **PCA Regression** — Dimensionality reduction to k=3 components (92.30% variance retained)

---

## Variables

| Symbol | Variable | Unit | Source |
|---|---|---|---|
| y | Export | Billion USD | DS1 |
| x₁ | Country Growth Rate | % | DS1 |
| x₂ | Inflation (CPI) | % | DS1+DS2 |
| x₃ | MFN Tariff Rate | % | DS1 |
| x₄ | Trade Openness | Index | DS1 |
| x₅ | Import/Export Ratio | Ratio | DS1 |

---

## Datasets

All data sourced from [Kaggle](https://www.kaggle.com):

- **DS1** — [World Export & Import Dataset](https://www.kaggle.com/datasets/muhammadtalhaawan/world-export-and-import-dataset) — Awan, M.T. (2023)
- **DS2** — [Macro-Economic Indicators Dataset](https://www.kaggle.com/datasets/veselagencheva/macro-economic-indicators-dataset-country-level) — Gencheva, V.
- **DS3** — [International Trade Database](https://www.kaggle.com/datasets/appetukhov/international-trade-database) — Appetukhov, A. (2022)

---

## Files

```
├── trade_analysis_real.py              # Main analysis script (LSQ, SVD, PCA)
├── trade_demo.html                     # Bilingual interactive dashboard (TR / EN)
├── mohammad yamen kabalan.pdf          # Full project report — Türkçe
├── mohammad yamen kabalan ENGLISH.pdf  # Full project report — English
└── README.md
```

---

## How to Run

**Requirements**
```bash
pip install numpy pandas
```

**Run Analysis**
```bash
python trade_analysis_real.py
```

**Interactive Demo**  
Open `trade_demo.html` directly in any browser — no server needed.  
Use the **TR / EN** toggle button in the top-right corner to switch languages.

---

## Key Results

| Finding | Value |
|---|---|
| Best model R² | 0.9981 |
| Strongest predictor | Trade Openness β₄ = +87.56 |
| LSQ vs SVD difference | ‖β_lstsq − β_svd‖ = 3.04 × 10⁻¹³ |
| SVD condition number | κ(X) = 7.85 |
| Gram matrix cond. number | cond(XᵀX) = 61.62 |
| Analysis period | 2000–2021 (n=22) |

---

## References

- Strang, G. (2006). *Linear Algebra and Its Applications* (4th ed.). Thomson Brooks/Cole.
- Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3), 211–218.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Helpman, E., & Krugman, P. R. (1985). *Market Structure and Foreign Trade*. MIT Press.
- Goldstein, M., & Khan, M. S. (1985). Income and price effects in foreign trade. *Handbook of International Economics*, Vol. 2.
- Tinbergen, J. (1962). *Shaping the World Economy*. Twentieth Century Fund.

---
---

# Türkiye İhracat Tahmini — Lineer Cebir Final Projesi 2026

**Ders:** Lineer Cebir  
**Hazırlayan:** Mohammad Yamen Kabalan  
**Konu:** Lineer Cebir yöntemleriyle Türkiye ihracat tahmini (2000–2021)

---

## Genel Bakış

Bu proje, beş makroekonomik değişken kullanarak Türkiye'nin yıllık ihracat performansını tahmin etmek amacıyla **Lineer Cebir tabanlı bir Karar Destek Sistemi** geliştirmektedir. Üç yöntem karşılaştırmalı biçimde uygulanmıştır:

| Yöntem | R² | RMSE | MAE |
|---|---|---|---|
| Normal Denklem (LSQ) | 0.9981 | 3.70 Milyar USD | 2.96 Milyar USD |
| SVD Pseudo-Inverse | 0.9981 | 3.70 Milyar USD | 2.96 Milyar USD |
| PCA Regresyon (k=3) | 0.8657 | 31.21 Milyar USD | 25.38 Milyar USD |

> LSQ ve SVD sayısal olarak özdeş sonuçlar üretmektedir: ‖β_lstsq − β_svd‖ = 3.04 × 10⁻¹³

---

## Yöntemler

- **Normal Denklem** — En Küçük Kareler kapalı form: β* = (XᵀX)⁻¹ Xᵀy, cond(XᵀX) = 61.62
- **SVD Pseudo-Inverse** — Moore-Penrose: β* = VΣ⁺Uᵀy, koşul sayısı κ(X) = 7.85
- **PCA Regresyon** — k=3 bileşene boyut indirgeme (%92.30 varyans korunmuştur)

---

## Değişkenler

| Sembol | Değişken | Birim | Kaynak |
|---|---|---|---|
| y | İhracat | Milyar USD | DS1 |
| x₁ | Ülke Büyüme Oranı | % | DS1 |
| x₂ | Enflasyon (TÜFE) | % | DS1+DS2 |
| x₃ | MFN Tarife Oranı | % | DS1 |
| x₄ | Ticaret Açıklığı | Endeks | DS1 |
| x₅ | İthalat/İhracat Oranı | Oran | DS1 |

---

## Veri Setleri

Tüm veriler [Kaggle](https://www.kaggle.com) platformundan elde edilmiştir:

- **DS1** — [World Export & Import Dataset](https://www.kaggle.com/datasets/muhammadtalhaawan/world-export-and-import-dataset) — Awan, M.T. (2023)
- **DS2** — [Macro-Economic Indicators Dataset](https://www.kaggle.com/datasets/veselagencheva/macro-economic-indicators-dataset-country-level) — Gencheva, V.
- **DS3** — [International Trade Database](https://www.kaggle.com/datasets/appetukhov/international-trade-database) — Appetukhov, A. (2022)

---

## Dosyalar

```
├── trade_analysis_real.py              # Ana analiz scripti (LSQ, SVD, PCA)
├── trade_demo.html                     # İki dilli interaktif dashboard (TR / EN)
├── mohammad yamen kabalan.pdf          # Tam proje raporu — Türkçe
├── mohammad yamen kabalan ENGLISH.pdf  # Tam proje raporu — İngilizce
└── README.md
```

---

## Nasıl Çalıştırılır

**Gereksinimler**
```bash
pip install numpy pandas
```

**Analizi Çalıştır**
```bash
python trade_analysis_real.py
```

**İnteraktif Demo**  
`trade_demo.html` dosyasını doğrudan tarayıcıda aç — sunucu gerekmez.  
Sağ üstteki **TR / EN** butonuyla dil değiştirilebilir.

---

## Temel Bulgular

| Bulgu | Değer |
|---|---|
| En iyi model R² | 0.9981 |
| En güçlü belirleyici | Ticaret Açıklığı β₄ = +87.56 |
| LSQ ile SVD farkı | ‖β_lstsq − β_svd‖ = 3.04 × 10⁻¹³ |
| SVD koşul sayısı | κ(X) = 7.85 |
| Gram matrisi koşul sayısı | cond(XᵀX) = 61.62 |
| Analiz dönemi | 2000–2021 (n=22) |

---

## Kaynakça

- Strang, G. (2006). *Linear Algebra and Its Applications* (4. baskı). Thomson Brooks/Cole.
- Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3), 211–218.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2. baskı). Springer.
- Helpman, E., & Krugman, P. R. (1985). *Market Structure and Foreign Trade*. MIT Press.
- Goldstein, M., & Khan, M. S. (1985). Income and price effects in foreign trade. *Handbook of International Economics*, Cilt 2.
- Tinbergen, J. (1962). *Shaping the World Economy*. Twentieth Century Fund.
