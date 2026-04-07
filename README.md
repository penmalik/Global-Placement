# 🌍 Global Placement — Student Job Placement Prediction

Predicting whether a student gets placed and estimating their expected salary using **Random Forest** on a 10,000-record global dataset. The project covers end-to-end data science workflow — from EDA to dual model deployment.

---

## 📌 Project Overview

Two core questions drive this project:
- **Can we predict whether a student will be placed?** → Random Forest Classifier
- **If placed, how much will they earn?** → Random Forest Regressor

The dataset spans students across 5 countries, multiple college tiers, specializations, and industries — making it a genuinely global placement study.

---

## 📁 Project Structure

```
global-placement/
│
├── Data/
│   └── global_placement.csv
│
├── Images/
│   ├── salary_distribution.png
│   ├── cgpa_salary_by_uni_ranking.png
│   ├── backlogs_by_placement.png
│   ├── placement_pie.png
│   ├── placement_rate_by_tier.png
│   ├── salary_by_country.png
│   ├── internship_quality_by_placement.png
│   ├── internship_count_by_placement.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance_classifier.png
│   ├── feature_importance_regressor.png
│   └── feature_correlation.png
│
├── Notebook/
│   └── global_placement.ipynb
│
└── README.md
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `cgpa` | Float | Cumulative Grade Point Average |
| `backlogs` | Int | Number of failed/pending subjects |
| `college_tier` | Object | Tier 1 / Tier 2 / Tier 3 |
| `country` | Object | Country of study |
| `university_ranking_band` | Object | Top 100 / 100-300 / 300+ |
| `internship_count` | Int | Number of internships completed |
| `aptitude_score` | Float | Aptitude test score |
| `communication_score` | Float | Communication assessment score |
| `specialization` | Object | Field of study |
| `industry` | Object | Industry of placement |
| `internship_quality_score` | Float | Quality rating of internships |
| `placement_status` | Object | **Target 1** — Placed / Not Placed |
| `salary` | Float | **Target 2** — Salary in USD |

---

## 🔍 Key EDA Findings

- **Overall placement rate is ~60%** — consistent across all college tiers, countries, and specializations
- **CGPA and backlogs** are the strongest academic placement signals — placed students cluster around CGPA 7+ with 0–1 backlogs
- **Internship quality matters more than count** — the caliber of experience separates placed from unplaced students more distinctly than the number of internships
- **Country doesn't affect placement rate but dramatically affects salary** — USA median salary (~100k USD) is nearly 2.5x that of India (~40k USD)
- **University ranking affects salary, not CGPA** — all ranking bands average ~7.0 CGPA, yet Top 100 graduates earn ~19,000 USD more than 300+ graduates
- **A combined skill score of ~140** (aptitude + communication) emerges as the soft threshold for placed students

---

## 🤖 Models

### Random Forest Classifier — Placement Prediction

| Metric | Score |
|---|---|
| Train Accuracy | 0.796 |
| Test Accuracy | 0.731 |
| Train-Test Gap | 0.065 |
| F1 Score | 0.811 |
| Precision | 0.715 |
| Recall | 0.937 |
| ROC-AUC | 0.812 |

> Threshold adjusted to **0.4** to prioritize recall — missing a genuinely placeable student is a costlier error than over-predicting placement.

### Random Forest Regressor — Salary Prediction

| Metric | Score |
|---|---|
| Train R² | 0.982 |
| Test R² | 0.957 |
| Train-Test Gap | 0.026 |
| Test MAE | ~3,813 USD |

> ~5% average error across a salary range of 40k–100k USD.

---

## 🏆 Feature Importance Highlights

**Placement Classifier:**
- CGPA and Backlogs dominate (~0.22–0.23 each)
- Internship quality is the third strongest signal
- College tier and country rank near the bottom — placement is merit-driven, not institution-driven

**Salary Regressor:**
- country_India dominates at 0.65 — reflects dramatic market rate differences, not candidate quality
- college_tier_Tier 3 is a distant second at 0.15

---

## 🛠️ Tech Stack

- **Python** — Pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Modeling** — Scikit-learn (RandomForestClassifier, RandomForestRegressor, RandomizedSearchCV)
- **Evaluation** — Accuracy, F1, Precision, Recall, ROC-AUC, R², MAE

---

## ▶️ Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/global-placement.git
cd global-placement

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebook/global_placement.ipynb
```

---

## 💡 Insights Summary

> Placement outcome in this dataset is driven by individual merit CGPA, backlogs, internship quality, and skill scores not by where a student studied or what country they are from. However, once placed, geography becomes the dominant salary determinant, with country of study accounting for the largest share of salary variance in the model.

## 📈 Next Steps

- [ ] Build a simple placement and salary prediction pipeline to run on new student data

---

## 👤 Author

[![GitHub](https://img.shields.io/badge/GitHub-penmalik-black?logo=github)](https://github.com/penmalik)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/ugochukwu-nnamani-9b3a27255/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?logo=kaggle)](https://www.kaggle.com/ugonnamani)
