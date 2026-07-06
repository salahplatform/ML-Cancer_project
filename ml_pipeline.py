"""
Breast Cancer Diagnosis using KNN and Logistic Regression
CS 4375 - Intro to Machine Learning
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve,
                              classification_report)

np.random.seed(42)
os.makedirs('figures', exist_ok=True)


# ==========================================================
# 1. LOAD DATA
# ===========================================================
print("Loading dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = malignant, 1 = benign

X = df.drop('target', axis=1)
y = df['target']
feature_names = list(X.columns)

print(f"Total samples  : {len(df)}")
print(f"Malignant (0)  : {(y==0).sum()}")
print(f"Benign    (1)  : {(y==1).sum()}")
print(f"Features       : {X.shape[1]}")
print(f"Missing values : {df.isnull().sum().sum()}")


# ==========================================================
# 2. EDA - FIGURE 1: Class distribution + top feature histograms
# ===========================================================
print("\nGenerating EDA figures...")

fscores, _ = f_classif(X, y)
top5_idx   = np.argsort(fscores)[-5:][::-1]
top5_names = [feature_names[i] for i in top5_idx]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Exploratory Data Analysis – Breast Cancer Dataset',
             fontsize=13, fontweight='bold')

# Class distribution bar chart
ax = axes[0, 0]
counts = y.value_counts()
bars = ax.bar(['Malignant', 'Benign'], [counts[0], counts[1]],
              color=['#e74c3c', '#2ecc71'], edgecolor='black', width=0.5)
for bar, count in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
            str(count), ha='center', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Number of Samples')
ax.set_ylim(0, 420)

# Top 5 feature histograms
for i, (ax, idx, name) in enumerate(
        zip([axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]],
            top5_idx, top5_names)):
    for label, color, lname in zip([0, 1], ['#e74c3c', '#2ecc71'],
                                    ['Malignant', 'Benign']):
        ax.hist(X.iloc[y == label, idx], bins=25, alpha=0.6,
                color=color, label=lname, density=True)
    ax.set_title(name, fontsize=8, fontweight='bold')
    ax.set_xlabel('Value', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig1_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/fig1_eda.png")


# =========================================================
# 3. FIGURE 2: Correlation heatmap
# ==========================================================
top12 = X.var().nlargest(12).index.tolist() + ['target']
corr  = df.corr().loc[top12, top12]

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.4, annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix (Top 12 Features + Target)',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)
plt.tight_layout()
plt.savefig('figures/fig2_corr.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/fig2_corr.png")


# ==========================================================
# 4. FIGURE 3: PCA + t-SNE
# =========================================================
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

tsne   = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Visualizing the Data: PCA and t-SNE', fontsize=13, fontweight='bold')

for ax, X_2d, title in zip(axes,
                             [X_pca, X_tsne],
                             ['PCA', 't-SNE']):
    for label, color, name in zip([0, 1], ['#e74c3c', '#2ecc71'],
                                   ['Malignant', 'Benign']):
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name,
                   alpha=0.6, s=30, edgecolors='none')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend()
    if title == 'PCA':
        pct = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({pct[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pct[1]*100:.1f}% variance)')
    else:
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

plt.tight_layout()
plt.savefig('figures/fig3_pca_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/fig3_pca_tsne.png")


# ===========================================================
# 5. FEATURE SELECTION - top 15 by F-score
# ==========================================================
selector         = SelectKBest(f_classif, k=15)
X_selected       = selector.fit_transform(X_scaled, y)
selected_mask    = selector.get_support()
selected_features = [feature_names[i] for i in range(len(feature_names))
                     if selected_mask[i]]

print(f"\nSelected features: {selected_features}")

# FIGURE 4: Feature F-scores bar chart
fig, ax = plt.subplots(figsize=(12, 5))
sorted_idx  = np.argsort(fscores)[::-1]
sorted_scores = fscores[sorted_idx]
sorted_names  = [feature_names[i] for i in sorted_idx]
colors = ['#2ecc71' if selected_mask[i] else '#bdc3c7' for i in sorted_idx]

ax.bar(range(len(feature_names)), sorted_scores, color=colors, edgecolor='white')
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('F-Score')
ax.set_title('Feature F-Scores (Green = Selected Top 15)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Selected (top 15)'),
                   Patch(facecolor='#bdc3c7', label='Not selected')]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('figures/fig4_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/fig4_features.png")


# ==========================================================
# 6. TRAIN / TEST SPLIT
# ===========================================================
X_sel = pd.DataFrame(X_selected)
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")


# ===========================================================
# 7. MODEL 1 - LOGISTIC REGRESSION
# ==========================================================
print("\n" + "="*50)
print("MODEL 1: Logistic Regression")
print("="*50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search over C values
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(LogisticRegression(max_iter=5000, random_state=42),
                       lr_param_grid, cv=cv, scoring='roc_auc')
lr_grid.fit(X_train, y_train)

best_lr    = lr_grid.best_estimator_
best_lr_C  = lr_grid.best_params_['C']
print(f"Best C value: {best_lr_C}")

# CV scores
lr_cv_scores = cross_val_score(best_lr, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"CV AUC: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

# Test evaluation
best_lr.fit(X_train, y_train)
y_pred_lr = best_lr.predict(X_test)
y_prob_lr = best_lr.predict_proba(X_test)[:, 1]

print(f"Test Accuracy  : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Test Precision : {precision_score(y_test, y_pred_lr):.4f}")
print(f"Test Recall    : {recall_score(y_test, y_pred_lr):.4f}")
print(f"Test F1        : {f1_score(y_test, y_pred_lr):.4f}")
print(f"Test AUC       : {roc_auc_score(y_test, y_prob_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Malignant', 'Benign']))


# =========================================================
# 8. MODEL 2 - KNN
# ============================================================
print("\n" + "="*50)
print("MODEL 2: K-Nearest Neighbors")
print("="*50)

# Grid search over k values
knn_param_grid = {'n_neighbors': list(range(1, 21))}
knn_grid = GridSearchCV(KNeighborsClassifier(),
                        knn_param_grid, cv=cv, scoring='roc_auc')
knn_grid.fit(X_train, y_train)

best_knn = knn_grid.best_estimator_
best_k   = knn_grid.best_params_['n_neighbors']
print(f"Best k: {best_k}")

# CV scores
knn_cv_scores = cross_val_score(best_knn, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"CV AUC: {knn_cv_scores.mean():.4f} ± {knn_cv_scores.std():.4f}")

# Test evaluation
best_knn.fit(X_train, y_train)
y_pred_knn = best_knn.predict(X_test)
y_prob_knn = best_knn.predict_proba(X_test)[:, 1]

print(f"Test Accuracy  : {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Test Precision : {precision_score(y_test, y_pred_knn):.4f}")
print(f"Test Recall    : {recall_score(y_test, y_pred_knn):.4f}")
print(f"Test F1        : {f1_score(y_test, y_pred_knn):.4f}")
print(f"Test AUC       : {roc_auc_score(y_test, y_prob_knn):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=['Malignant', 'Benign']))


# ===========================================================
# 9. FIGURE 5: KNN - effect of k on AUC
# ==========================================================
k_values = list(range(1, 21))
k_aucs   = []
for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             X_train, y_train, cv=cv, scoring='roc_auc')
    k_aucs.append(scores.mean())

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(k_values, k_aucs, marker='o', color='#3498db', linewidth=2, markersize=6)
ax.axvline(x=best_k, color='#e74c3c', linestyle='--', linewidth=2,
           label=f'Best k={best_k} (AUC={max(k_aucs):.4f})')
ax.set_xlabel('Number of Neighbors (k)', fontsize=11)
ax.set_ylabel('Cross-Validation AUC', fontsize=11)
ax.set_title('KNN: Effect of k on Model Performance', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xticks(k_values)
plt.tight_layout()
plt.savefig('figures/fig5_knn_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: figures/fig5_knn_k.png")


# ==========================================================
# 10. FIGURE 6: Logistic Regression  effect of C on AUC
# ==========================================================
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
C_aucs   = []
for C in C_values:
    scores = cross_val_score(LogisticRegression(C=C, max_iter=5000, random_state=42),
                             X_train, y_train, cv=cv, scoring='roc_auc')
    C_aucs.append(scores.mean())

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([str(c) for c in C_values], C_aucs, marker='s', color='#e67e22',
        linewidth=2, markersize=7)
ax.axvline(x=str(best_lr_C), color='#e74c3c', linestyle='--', linewidth=2,
           label=f'Best C={best_lr_C} (AUC={max(C_aucs):.4f})')
ax.set_xlabel('Regularization Parameter C', fontsize=11)
ax.set_ylabel('Cross-Validation AUC', fontsize=11)
ax.set_title('Logistic Regression: Effect of C on Performance', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig6_lr_C.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/fig6_lr_C.png")


# =========================================================
# 11. FIGURE 7: Confusion matrices side by side
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('Confusion Matrices – Logistic Regression vs KNN',
             fontsize=13, fontweight='bold')

for ax, y_pred, title in zip(axes,
                               [y_pred_lr, y_pred_knn],
                               [f'Logistic Regression (C={best_lr_C})',
                                f'KNN (k={best_k})']):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                linewidths=1, linecolor='gray', annot_kws={'size': 14})
    acc = accuracy_score(y_test, y_pred)
    ax.set_title(f'{title}\nAccuracy = {acc:.3f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('figures/fig7_confmat.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/fig7_confmat.png")


# =========================================================
# 12. FIGURE 8: ROC curves
# ==========================================================
fig, ax = plt.subplots(figsize=(8, 6))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)

ax.plot(fpr_lr,  tpr_lr,  color='#e67e22', linewidth=2.5,
        label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})')
ax.plot(fpr_knn, tpr_knn, color='#3498db', linewidth=2.5,
        label=f'KNN k={best_k} (AUC = {roc_auc_score(y_test, y_prob_knn):.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Guess')

ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves – Model Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig8_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/fig8_roc.png")


# =========================================================
# 13. FIGURE 9: Side by side metric comparison
# ===========================================================
metrics      = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
lr_scores    = [accuracy_score(y_test, y_pred_lr), precision_score(y_test, y_pred_lr),
                recall_score(y_test, y_pred_lr),   f1_score(y_test, y_pred_lr),
                roc_auc_score(y_test, y_prob_lr)]
knn_scores   = [accuracy_score(y_test, y_pred_knn), precision_score(y_test, y_pred_knn),
                recall_score(y_test, y_pred_knn),    f1_score(y_test, y_pred_knn),
                roc_auc_score(y_test, y_prob_knn)]

x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, lr_scores,  width, label='Logistic Regression',
               color='#e67e22', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, knn_scores, width, label=f'KNN (k={best_k})',
               color='#3498db', alpha=0.85, edgecolor='white')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0.88, 1.02)
ax.set_title('Model Comparison: All Metrics', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig9_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/fig9_comparison.png")


# ==========================================================
# 14. FINAL SUMMARY
# ==========================================================
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
print(f"{'Metric':<15} {'Logistic Reg':>15} {'KNN':>10}")
print("-"*42)
print(f"{'Accuracy':<15} {accuracy_score(y_test,y_pred_lr):>15.4f} {accuracy_score(y_test,y_pred_knn):>10.4f}")
print(f"{'Precision':<15} {precision_score(y_test,y_pred_lr):>15.4f} {precision_score(y_test,y_pred_knn):>10.4f}")
print(f"{'Recall':<15} {recall_score(y_test,y_pred_lr):>15.4f} {recall_score(y_test,y_pred_knn):>10.4f}")
print(f"{'F1 Score':<15} {f1_score(y_test,y_pred_lr):>15.4f} {f1_score(y_test,y_pred_knn):>10.4f}")
print(f"{'AUC':<15} {roc_auc_score(y_test,y_prob_lr):>15.4f} {roc_auc_score(y_test,y_prob_knn):>10.4f}")
print(f"{'Best Param':<15} {'C='+str(best_lr_C):>15} {'k='+str(best_k):>10}")
print("\nAll figures saved to ./figures/")
