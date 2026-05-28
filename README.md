# Turkey Export Prediction — Linear Algebra Final Project 2026

**Course:** Linear Algebra (Lineer Cebir)  
**Author:** Mohammad Yamen Kabalan  
**Topic:** Turkey export forecasting using Linear Algebra methods (2000–2021)

---

## Overview

This project develops a **Linear Algebra-based Decision Support System** for predicting Turkey's annual export performance using five macroeconomic variables. Three methods are implemented and compared:

| Method | R² | RMSE |
|---|---|---|
| Normal Equation (LSQ) | 0.9981 | 3.70B USD |
| SVD Pseudo-Inverse | 0.9981 | 3.70B USD |
| PCA Regression (k=3) | 0.8657 | 31.21B USD |

---

## Methods

- **Normal Equation** — Least Squares closed-form solution: β* = (XᵀX)⁻¹ Xᵀy
- **SVD Pseudo-Inverse** — Moore-Penrose pseudo-inverse: β* = VΣ⁺Uᵀy, condition number κ(X) = 7.85
- **PCA Regression** — Dimensionality reduction to k=3 components (92.30% variance retained)

---

## Variables

| Symbol | Variable | Unit |
|---|---|---|
| y | Export | Billion USD |
| x₁ | Country Growth Rate | % |
| x₂ | Inflation (CPI) | % |
| x₃ | MFN Tariff Rate | % |
| x₄ | Trade Openness | Index |
| x₅ | Import/Export Ratio | Ratio |

---

## Datasets

All data sourced from [Kaggle](https://www.kaggle.com):

- **DS1** — [World Export & Import Dataset](https://www.kaggle.com/datasets/muhammadtalhaawan/world-export-and-import-dataset) — Awan, M.T. (2023)
- **DS2** — [Macro-Economic Indicators Dataset](https://www.kaggle.com/datasets/veselagencheva/macro-economic-indicators-dataset-country-level) — Gencheva, V.
- **DS3** — [International Trade Database](https://www.kaggle.com/datasets/appetukhov/international-trade-database) — Appetukhov, A. (2022)

---

## Files

```
├── trade_analysis_real.py     # Main analysis script (LSQ, SVD, PCA)
├── trade_demo.html            # Interactive decision support dashboard (Chart.js)
├── mohammad yamen kabalan.pdf # Full project report
└── README.md
```

---

## How to Run

### Requirements

```bash
pip install numpy pandas
```

### Run Analysis

```bash
python trade_analysis_real.py
```

### Interactive Demo

Open `trade_demo.html` directly in any browser — no server needed.

---

## Key Results

- LSQ and SVD produce numerically identical results: ‖β_lstsq − β_svd‖ = 3.04 × 10⁻¹³
- **Trade Openness** is the dominant predictor: β₄ = +87.56
- SVD condition number κ(X) = 7.85 vs cond(XᵀX) = 61.62 — confirms SVD numerical advantage
- Model covers Turkey's major economic events: 2001 crisis, 2008–2009 global crisis, COVID recovery

---

## References

- Strang, G. (2006). *Linear Algebra and Its Applications* (4th ed.). Thomson Brooks/Cole.
- Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3), 211–218.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
