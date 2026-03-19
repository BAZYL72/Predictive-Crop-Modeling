# Crop Recommendation ML

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)

---

## Problem Statement

Soil testing is a prerequisite for optimal crop selection, yet the cost of measuring every soil metric — nitrogen (N), phosphorous (P), potassium (K), and pH — is prohibitive for smallholder farmers operating under tight budget constraints. This project addresses a practical question: **can a single, affordable soil measurement reliably predict the best crop to plant?**

Using the `soil_measures.csv` dataset, the objective is twofold:

1. Build multi-class classification models that predict the optimal crop from soil measurements.
2. Identify the **single most predictive soil feature**, enabling cost-effective, data-driven crop recommendations even when only one measurement can be taken.

---

## Dataset

Each row in `soil_measures.csv` represents a field observation with the following variables:

| Feature | Description |
|---|---|
| `N` | Nitrogen content ratio in the soil |
| `P` | Phosphorous content ratio in the soil |
| `K` | Potassium content ratio in the soil |
| `ph` | pH value of the soil |
| `crop` | Target variable — the optimal crop for those soil conditions |

---

## Exploratory Data Analysis

Prior to modelling, the dataset was inspected across four dimensions:

- **Structure**: `crops.info()` confirmed column data types and the absence of null values, validating the dataset's readiness for direct modelling without imputation.
- **Descriptive statistics**: `crops.describe()` revealed the range, mean, and spread of each soil metric, informing the choice of evaluation metrics sensitive to class distribution.
- **Missing value check**: `crops.isna().sum()` confirmed zero missing values across all features.
- **Class distribution**: The multi-crop target variable spans numerous crop types, making it essential to apply metrics that account for potential imbalance between common and rare crops.

---

## Model Selection Strategy

### Algorithm: Multinomial Logistic Regression

For the feature importance analysis, each soil metric was evaluated in isolation using:

```python
LogisticRegression(multi_class='multinomial')
```

The `multinomial` setting is the correct choice for this problem because the target variable contains more than two crop classes. Rather than reducing the problem to a series of binary one-vs-rest decisions, multinomial logistic regression models the joint probability distribution across all classes simultaneously using a softmax function. This yields a coherent probabilistic output across every crop class in a single model pass, making it both statistically appropriate and computationally efficient for a multi-class classification task.

### Evaluation Metrics: Weighted F1-Score and Balanced Accuracy

Two metrics were chosen to ensure evaluation is faithful to real-world performance, particularly for underrepresented crops:

**Weighted F1-Score** computes the harmonic mean of precision and recall for each class, then takes a weighted average proportional to class support. This prevents dominant classes from masking poor performance on minority crop types — a critical requirement when certain crops appear far less frequently in the dataset.

**Balanced Accuracy** calculates the average recall across all classes, assigning equal weight to each regardless of how many samples it contains. A model that correctly predicts only majority classes would score poorly on this metric, exposing it as unreliable for budget-conscious agricultural decisions where every crop type matters.

Together, these metrics ensure that a model cannot achieve high scores by ignoring difficult or infrequent crops, producing an honest assessment of predictive power.

### Single-Feature Evaluation Loop

Each of the four soil features (N, P, K, ph) was assessed independently:

```python
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class='multinomial')
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])

    f1  = f1_score(y_test, y_pred, average='weighted')
    bac = balanced_accuracy_score(y_test, y_pred)
```

Results were collected into a DataFrame sorted in descending order by Weighted F1-Score, with the bar chart in Cell 9 providing an immediate visual comparison of each feature's standalone predictive power.

---

## Final Discovery: The Single Best Predictive Feature

```python
best_predictive_feature = {best_row['Feature']: best_row['F1-Score']}
```

The analysis identified **K (Potassium)** as the single most predictive soil metric, achieving the highest Weighted F1-Score of the four features evaluated.

This result carries meaningful agricultural significance. Potassium governs a wide range of plant physiological functions — including water regulation, enzyme activation, and resistance to disease and drought stress — and its optimal range varies substantially across crop families. This wide inter-crop variation is precisely why K carries high discriminative power: knowing the potassium level of a field provides strong signal about which crops are viable and which are not.

For farmers who can only afford one soil test, this finding suggests that a single potassium measurement offers the greatest return on information: an actionable, data-backed crop recommendation at the lowest possible cost.

---

## How to Run

**Install dependencies:**

```bash
pip install pandas scikit-learn matplotlib
```

**Dependencies derived from notebook import statements:**

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and results tabulation |
| `scikit-learn` | `LogisticRegression`, `train_test_split`, `f1_score`, `balanced_accuracy_score` |
| `matplotlib` | Bar chart visualisation of per-feature predictive power |

**Run the notebook:**

```bash
jupyter notebook notebook.ipynb
```

Ensure `soil_measures.csv` is located in the same directory as the notebook before execution.

---

## Project Structure

```
crop-recommendation-ml/
├── notebook.ipynb          # Full analysis and modelling pipeline
├── soil_measures.csv       # Input dataset (soil measurements + crop labels)
└── README.md               # Project documentation
```

---

## Key Takeaways

- **Multinomial logistic regression** is the statistically correct baseline for joint multi-class classification over more than two target labels.
- **Weighted F1-Score and Balanced Accuracy** together provide an imbalance-resistant evaluation framework, ensuring performance on rare crops is not obscured.
- **Potassium (K)** is the most informative single soil metric for crop prediction — a finding with direct, practical value for resource-constrained agricultural decision-making.
