# 🌾 Crop-Recommendation-ML: Sowing Success with Data

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> *A machine learning pipeline that identifies the single most predictive soil metric for crop selection — empowering budget-conscious farmers to make data-driven decisions with minimal testing.*

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis-eda)
- [Model Selection Strategy](#-model-selection-strategy)
- [Evaluation Philosophy: Faithfulness via Dual Metrics](#-evaluation-philosophy-faithfulness-via-dual-metrics)
- [Final Discovery: The Single Best Feature](#-final-discovery-the-single-best-feature)
- [How to Run](#-how-to-run)

---

## 🌱 Problem Statement

Comprehensive soil analysis is a scientifically rigorous process — but it comes with a price tag that many smallholder farmers simply cannot afford. Measuring nitrogen (N), phosphorous (P), potassium (K), and soil pH individually requires separate testing kits, laboratory fees, or specialist equipment.

**The core question this project answers:**

> *If a farmer can only afford to test for **one** soil metric, which single measurement gives them the highest predictive power for choosing the optimal crop?*

This project builds a multi-class classification system on `soil_measures.csv`, trains individual models per soil feature, and uses rigorous evaluation to surface the single most cost-effective predictor. The result is a practical, budget-aware recommendation that a farmer can act on immediately.

---

## 📊 Dataset

The dataset `soil_measures.csv` contains soil measurement records, where each row represents a field's soil profile alongside its optimal crop label.

| Column | Type | Description |
|---|---|---|
| `N` | Numeric | Nitrogen content ratio in the soil |
| `P` | Numeric | Phosphorous content ratio in the soil |
| `K` | Numeric | Potassium content ratio in the soil |
| `ph` | Numeric | pH value of the soil |
| `crop` | Categorical | Target variable — the optimal crop for that field |

The dataset was verified to contain **no missing values**, confirmed via `crops.isna().sum()` — a critical pre-modelling step to ensure pipeline integrity.

---

## 🔍 Exploratory Data Analysis (EDA)

Before any modelling, a systematic EDA was performed to understand the data's structure and quality:

- **`crops.head()`** — Verified column naming, data types, and sample records.
- **`crops.info()`** — Confirmed non-null counts, data types per column, and memory usage.
- **`crops.describe()`** — Examined the statistical distribution (mean, std, min/max, quartiles) of each soil metric to flag potential outliers and understand feature ranges.
- **`crops.isna().sum()`** — Confirmed zero missing values across all columns, validating that no imputation strategy was needed.

This grounding step ensured that the features fed into training were clean, correctly typed, and representative of the underlying agricultural domain.

---

## ⚙️ Model Selection Strategy

### Why `LogisticRegression(multi_class='multinomial')`?

Standard binary logistic regression cannot natively handle problems with more than two classes. This dataset contains **multiple distinct crop types**, making it a true multi-class classification problem.

The `multi_class='multinomial'` parameter instructs scikit-learn to use the **Softmax (multinomial logistic) formulation** instead of a set of independent one-vs-rest classifiers. This is the correct choice here for several reasons:

1. **Jointly calibrated probabilities**: The softmax function produces a single probability distribution across *all* crop classes simultaneously, ensuring that probabilities sum to 1.0. This gives more coherent and interpretable predictions than independent binary classifiers.
2. **Cross-class competition**: In multinomial mode, the model learns that predicting "rice" higher necessarily means predicting "maize" lower — a realistic constraint in crop selection where one field grows one crop.
3. **Single-feature probing**: Because we are evaluating each soil metric (`N`, `P`, `K`, `ph`) in isolation, a linear model like Logistic Regression serves as an ideal, low-variance baseline. It is interpretable, fast, and produces stable results that cleanly reflect each feature's individual signal — without overfitting noise.

The training loop iterates over each feature independently:

```python
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class='multinomial')
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
```

This deliberate single-feature design is the core experimental insight: by isolating each metric, the comparison is fair and direct.

### Visualising the Results

A bar chart (`plt.bar`) was generated to compare the **Weighted F1-Score** of each individual feature, with the y-axis bounded at [0, 1] for consistent perspective across all features. This gave an immediate visual answer to the central question: *which metric stands tallest?*

---

## 📐 Evaluation Philosophy: Faithfulness via Dual Metrics

The evaluation deliberately combines **two complementary metrics** to guard against misleading conclusions in multi-class settings.

### 1. Weighted F1-Score

```python
f1 = f1_score(y_test, y_pred, average='weighted')
```

The weighted F1-Score computes the harmonic mean of precision and recall for each class, then aggregates them **weighted by class support** (i.e., how frequently each crop appears in the test set). This penalises a model that achieves high accuracy simply by excelling on common crops while ignoring rare ones.

### 2. Balanced Accuracy Score

```python
bac = balanced_accuracy_score(y_test, y_pred)
```

Balanced accuracy computes the average recall **across all classes equally**, regardless of how often each crop appears. This is the direct antidote to class imbalance: a model that completely ignores a rare crop will score 0% recall for that class, dragging the balanced accuracy down — even if its overall accuracy looks impressive.

### Why Both Together = "Faithfulness"

Using either metric alone can paint a misleading picture:

| Metric alone | Risk |
|---|---|
| Weighted F1 only | May reward ignoring minority crop classes if they are rare enough |
| Balanced Accuracy only | Does not capture precision — a model can guess freely and still score |

Together, they form a **faithful evaluation framework**: a model must demonstrate both *precision and recall across the full crop distribution* and *equitable performance regardless of crop frequency*. Only a genuinely informative feature can score well on both simultaneously. The results were stored per-feature and ranked by F1-Score to surface the clearest winner:

```python
results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
```

---

## 🏆 Final Discovery: The Single Best Feature

After evaluating all four soil metrics individually, the results were clear:

```python
best_row = results_df.iloc[0]  # DataFrame sorted by F1-Score descending

best_predictive_feature = {
    best_row['Feature']: best_row['F1-Score']
}

print(f"Final Selection: {best_predictive_feature}")
```

**`{'K': <score>}`** — **Potassium (K)** emerged as the single best predictive feature.

### Agricultural Significance of Potassium (K)

This result is not just a statistical artefact — it is agronomically meaningful:

- **Cellular regulation**: Potassium governs osmotic balance within plant cells, controlling water uptake and drought resilience. Different crops have radically different potassium demands, making K a natural discriminator.
- **Crop-specific signatures**: Crops like banana, coconut, and grapes are notably high-potassium consumers, while cereals like rice and wheat operate at far lower K levels. This creates distinct, separable clusters in K-space that the model can exploit.
- **Yield quality, not just quantity**: Unlike nitrogen (which broadly governs vegetative growth across many crop types), potassium's effects are more crop-specific — governing fruit quality, disease resistance, and root development in ways that differ significantly between species.
- **Practical implication**: A farmer with a single soil-testing budget should **prioritise a potassium test**. The K reading alone provides more information about optimal crop selection than any other individual soil metric in this dataset.

---

## 🚀 How to Run

### Prerequisites

All dependencies are drawn directly from the project's `import` statements:

```bash
pip install pandas scikit-learn matplotlib
```

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.5 | Data loading, EDA, and results aggregation |
| `scikit-learn` | ≥ 1.1 | `LogisticRegression`, `train_test_split`, `f1_score`, `balanced_accuracy_score` |
| `matplotlib` | ≥ 3.5 | Bar chart visualisation of per-feature F1 scores |

### Running the Notebook

```bash
# Clone the repository
git clone https://github.com/your-username/Crop-Recommendation-ML.git
cd Crop-Recommendation-ML

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebook.ipynb
```

Ensure `soil_measures.csv` is present in the root directory alongside `notebook.ipynb` before running.

---

## 📁 Project Structure

```
Crop-Recommendation-ML/
│
├── notebook.ipynb          # Main analysis notebook
├── soil_measures.csv       # Dataset (N, P, K, pH, crop)
├── farmer_in_a_field.jpg   # Cover image
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🤝 Acknowledgements

Dataset sourced as part of a structured ML curriculum. The problem framing — identifying a *single* actionable soil metric — reflects real-world agricultural constraints faced by subsistence and smallholder farmers globally.

---

*Built with 🌱 and scikit-learn.*
