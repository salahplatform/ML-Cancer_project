# Breast Cancer Diagnosis — ML Classification Study

Comparing Logistic Regression and K-Nearest Neighbors for breast cancer diagnosis using cell nucleus measurements from the Wisconsin Breast Cancer Diagnostic (WBCD) dataset.

## Overview

This project frames breast cancer diagnosis as a binary classification problem: given 30 numeric measurements extracted from a fine needle aspirate (FNA) of a breast mass, predict whether the tumor is **malignant** or **benign**.

I chose this problem because my brother is a doctor, and I wanted to see whether a machine learning model could meaningfully contribute to something as high-stakes as medical diagnosis — and to understand not just which model performs better, but *why*.

## Dataset

- **Source:** [Wisconsin Breast Cancer Diagnostic dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) (UCI ML Repository / scikit-learn)
- **Samples:** 569 (212 malignant, 357 benign)
- **Features:** 30 numeric features — 10 physical properties of the cell nucleus (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each summarized as a mean, standard error, and "worst" (most extreme) value
- **Missing values:** None

## Approach

1. **Exploratory Data Analysis** — class distribution, feature distributions by F-score, and correlation analysis to identify multicollinearity (e.g., radius/perimeter/area correlate at r > 0.98)
2. **Dimensionality Reduction** — PCA and t-SNE to visualize class separability. PCA showed the two classes are nearly linearly separable (top 2 components explain ~63% of variance), which correctly predicted that a linear model would perform well
3. **Feature Selection** — used `SelectKBest` with ANOVA F-scores to reduce 30 features down to the 15 most discriminative, dominated by concavity/shape-irregularity and "worst-value" measurements
4. **Model Training** — compared two models covered in coursework:
   - **Logistic Regression** — tuned regularization strength `C` via grid search + 5-fold cross-validation
   - **K-Nearest Neighbors (KNN)** — tuned `k` via grid search + 5-fold cross-validation
5. **Evaluation** — 5-fold stratified cross-validation, confusion matrices, ROC curves, and a full metric comparison (accuracy, precision, recall, F1, AUC), with particular attention to **recall**, since false negatives (missed malignancies) are the more dangerous error in a medical context

## Results

| Metric | Logistic Regression (C=10) | KNN (k=18) |
|---|---|---|
| CV AUC | 0.9914 ± 0.0036 | 0.9850 ± 0.0108 |
| Accuracy | 0.9561 | 0.9298 |
| Precision | 0.9589 | 0.9444 |
| Recall | 0.9722 | 0.9444 |
| F1 Score | 0.9655 | 0.9444 |
| Test AUC | 0.9931 | 0.9785 |

**Logistic Regression outperformed KNN on every metric.** This lines up with the PCA analysis — since the two classes are nearly linearly separable, a linear decision boundary generalizes better than a distance-based method, especially with only 455 training samples spread across 15 features (where KNN's local neighborhoods become sparse and less reliable).

## Key Takeaways

- **Shape irregularity beats raw size.** The most predictive features were concavity/concave-points and "worst-value" measurements — not the mean measurements. This matches clinical intuition: malignant cells are more irregularly shaped, and a single highly abnormal cell can be a stronger signal than an average.
- **Simpler models can win.** When the underlying data is close to linearly separable, a well-tuned Logistic Regression model can match or beat more flexible methods like KNN — while also being more interpretable and more stable across cross-validation folds (lower variance).
- **Recall matters more than accuracy here.** In a diagnostic setting, a missed malignant case (false negative) is far costlier than a false alarm — so recall and AUC were weighted more heavily than raw accuracy throughout model selection.

## Limitations

- Dataset is relatively small (569 samples) and from a single institution — generalization to other patient populations/imaging equipment is untested
- Only two model families were compared; SVM in particular seems like a natural next step given the near-linear separability with some noise
- No probability calibration was performed — AUC is high, but predicted probabilities haven't been validated as true likelihoods

## Tech Stack

Python, scikit-learn, pandas, NumPy, Matplotlib/Seaborn

## How to Run

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run

The dataset loads directly from scikit-learn — no external data file needed.

```bash
python ml_pipeline.py
```

### What it does

Running `ml_pipeline.py` will:
1. Load the Wisconsin Breast Cancer dataset and print basic dataset stats to the console
2. Run the full EDA → feature selection → model training → evaluation pipeline described above
3. Generate all 9 figures (EDA, correlation heatmap, PCA/t-SNE, feature F-scores, hyperparameter tuning curves, confusion matrices, ROC curves, metric comparison) and save them to a `figures/` directory (created automatically)
4. Print cross-validation results, test-set metrics, classification reports, and a final side-by-side summary table for both models to the console

No arguments or configuration needed — it runs end-to-end as a single script.

## References

1. Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). *Machine learning techniques to diagnose breast cancer from fine-needle aspirates.* Cancer Letters, 77(2-3), 163-171.
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825-2830.

---
*Course project for CS 4375 – Introduction to Machine Learning, UT Dallas, Spring 2026*
