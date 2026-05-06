"""
Raisin Classification using Naive Bayes and Exploratory Data Analysis
This script performs data exploration, visualization, preprocessing, 
and builds a Gaussian Naive Bayes classifier for raisin type classification.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# ============================================================
# 1. LOAD AND EXPLORE THE DATASET
# ============================================================
print("="*60)
print("RAISIN CLASSIFICATION - DATA EXPLORATION & ANALYSIS")
print("="*60)

# Load the dataset
df = pd.read_excel(r"Raisin_Dataset.xlsx")

print("\n[1] DATASET OVERVIEW")
print("-" * 60)
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\n[2] DATASET INFO")
print("-" * 60)
print(df.info())

print("\n[3] STATISTICAL SUMMARY")
print("-" * 60)
print(df.describe())

print("\n[4] DATA QUALITY CHECK")
print("-" * 60)
print("Missing values per column:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
# ============================================================
print("\n\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS & VISUALIZATION")
print("="*60)

# Pairplot
print("\n[5] Generating Pairplot (Feature Relationships)...")
sns.pairplot(df, hue='Class', plot_kws={'alpha': 0.8})
plt.suptitle('Pairplot of Raisin Features by Class', fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
plt.show()

# Correlation Matrix Heatmap
print("[6] Generating Correlation Matrix Heatmap...")
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', square=True, fmt='.2f')
plt.title("Correlation Matrix of Features", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 3. DIMENSIONALITY REDUCTION - PCA
# ============================================================
print("\n[7] Performing PCA Analysis...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.iloc[:, :-1])

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative Variance: {sum(pca.explained_variance_ratio_):.4f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Class'], s=100, palette='viridis', alpha=0.7, edgecolor='k')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
plt.title('PCA: 2D Visualization of Raisin Dataset', fontsize=12, fontweight='bold')
plt.legend(title='Class')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 4. OUTLIER DETECTION AND REMOVAL
# ============================================================
print("\n" + "="*60)
print("OUTLIER DETECTION & REMOVAL")
print("="*60)

features = df.columns[:-1]
df_filtered = df.copy()

print(f"\n[8] Using IQR Method to detect outliers...")
for f in features:
    Q1 = df_filtered[f].quantile(0.25)
    Q3 = df_filtered[f].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers_count = len(df_filtered[(df_filtered[f] < lower) | (df_filtered[f] > upper)])
    if outliers_count > 0:
        print(f"  {f}: {outliers_count} outliers detected (range: [{lower:.2f}, {upper:.2f}])")
    
    df_filtered = df_filtered[(df_filtered[f] >= lower) & (df_filtered[f] <= upper)]

print(f"\nOriginal dataset size: {len(df)}")
print(f"After outlier removal: {len(df_filtered)}")
print(f"Rows removed: {len(df) - len(df_filtered)}")

# Boxplot for feature-wise outliers
print("\n[9] Generating Boxplot visualization...")
plt.figure(figsize=(12, 5))
sns.boxplot(data=df.iloc[:, :-1], palette='Set2')
plt.xticks(rotation=45)
plt.title("Feature-wise Boxplot (Before Outlier Removal)", fontsize=12, fontweight='bold')
plt.ylabel('Values')
plt.tight_layout()
plt.show()

# ============================================================
# 5. DATA PREPARATION FOR MODELING
# ============================================================
print("\n" + "="*60)
print("DATA PREPARATION FOR MODELING")
print("="*60)

X = df_filtered.iloc[:, :-1]  # Features
y = df_filtered.iloc[:, -1]   # Target (Class)

print(f"\n[10] Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")
print(f"\nTarget classes: {sorted(y.unique())}")
print(f"Class distribution:\n{y.value_counts()}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n[11] Train-Test Split:")
print(f"  Training set shape: {X_train.shape}")
print(f"  Test set shape: {X_test.shape}")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# ============================================================
# 6. BUILD AND TRAIN GAUSSIAN NAIVE BAYES MODEL
# ============================================================
print("\n" + "="*60)
print("GAUSSIAN NAIVE BAYES MODEL")
print("="*60)

print("\n[12] Training Gaussian Naive Bayes Model...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred_train = nb_model.predict(X_train)
y_pred_test = nb_model.predict(X_test)

print("Model training completed!")

# ============================================================
# 7. MODEL EVALUATION
# ============================================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Accuracy Scores
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n[13] Accuracy Scores:")
print(f"  Train Accuracy: {train_acc * 100:.2f}%")
print(f"  Test Accuracy: {test_acc * 100:.2f}%")

# Classification Report
print(f"\n[14] Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
print(f"\n[15] Confusion Matrix:")
labels = sorted(df['Class'].unique())
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)
plt.title("Confusion Matrix - Gaussian Naive Bayes", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 8. SUMMARY REPORT
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY REPORT")
print("="*60)
print(f"\nDataset Information:")
print(f"  Original Size: {len(df)} samples")
print(f"  After Outlier Removal: {len(df_filtered)} samples")
print(f"  Number of Features: {X.shape[1]}")
print(f"  Number of Classes: {len(np.unique(y))}")

print(f"\nModel Performance:")
print(f"  Train Accuracy: {train_acc * 100:.2f}%")
print(f"  Test Accuracy: {test_acc * 100:.2f}%")
print(f"  Overfitting Gap: {(train_acc - test_acc) * 100:.2f}%")

print(f"\nPCA Analysis:")
print(f"  Variance Explained (2 components): {sum(pca.explained_variance_ratio_):.2%}")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
