"""
Raisin Classification using Naive Bayes and Exploratory Data Analysis
Optimized for Google Colab - with Model Comparison and Enhanced Analysis

This script performs data exploration, visualization, preprocessing, 
and builds multiple classifiers for raisin type classification.
"""

# ============================================================
# SETUP FOR GOOGLE COLAB (Uncomment to run in Colab)
# ============================================================
# !pip install pandas openpyxl scikit-learn seaborn matplotlib numpy -q
# from google.colab import files
# print("Upload your Raisin_Dataset.xlsx file:")
# uploaded = files.upload()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD AND EXPLORE THE DATASET
# ============================================================
print("="*60)
print("RAISIN CLASSIFICATION - DATA EXPLORATION & ANALYSIS")
print("="*60)

# Load the dataset (works in Colab after upload)
try:
    df = pd.read_excel("Raisin_Dataset.xlsx")
except FileNotFoundError:
    print("File not found. Please upload Raisin_Dataset.xlsx")
    exit()

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

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n[11] Train-Test Split:")
print(f"  Training set shape: {X_train.shape}")
print(f"  Test set shape: {X_test.shape}")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# ============================================================
# 6. BUILD AND TRAIN MULTIPLE MODELS
# ============================================================
print("\n" + "="*60)
print("BUILDING AND TRAINING MULTIPLE CLASSIFIERS")
print("="*60)

models = {
    'Gaussian Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
}

trained_models = {}
results = {}

for model_name, model in models.items():
    print(f"\n[12] Training {model_name}...")
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    results[model_name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_pred_test': y_pred_test,
        'model': model
    }
    
    print(f"  Train Accuracy: {train_acc * 100:.2f}%")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Overfitting Gap: {(train_acc - test_acc) * 100:.2f}%")

# ============================================================
# 7. MODEL EVALUATION & COMPARISON
# ============================================================
print("\n" + "="*60)
print("MODEL EVALUATION & COMPARISON")
print("="*60)

# Model Comparison Table
print("\n[13] Model Performance Summary:")
print("-" * 60)
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_acc'] * 100 for m in results.keys()],
    'Test Accuracy': [results[m]['test_acc'] * 100 for m in results.keys()],
    'Overfitting Gap': [(results[m]['train_acc'] - results[m]['test_acc']) * 100 for m in results.keys()]
})
print(comparison_df.to_string(index=False))

# Visualize Model Comparison
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(results))
train_accs = [results[m]['train_acc'] * 100 for m in results.keys()]
test_accs = [results[m]['test_acc'] * 100 for m in results.keys()]

plt.bar(x_pos - 0.2, train_accs, 0.4, label='Train Accuracy', alpha=0.8)
plt.bar(x_pos + 0.2, test_accs, 0.4, label='Test Accuracy', alpha=0.8)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Comparison - Train vs Test Accuracy', fontsize=12, fontweight='bold')
plt.xticks(x_pos, results.keys(), rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Detailed evaluation for best model
best_model_name = max(results, key=lambda x: results[x]['test_acc'])
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred_test']

print(f"\n[14] Best Model: {best_model_name}")
print(f"  Test Accuracy: {results[best_model_name]['test_acc'] * 100:.2f}%")

print(f"\n[15] Classification Report ({best_model_name}):")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix
print(f"\n[16] Confusion Matrix ({best_model_name}):")
labels = sorted(df['Class'].unique())
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)
plt.title(f"Confusion Matrix - {best_model_name}", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 8. CROSS-VALIDATION
# ============================================================
print("\n" + "="*60)
print("CROSS-VALIDATION ANALYSIS")
print("="*60)

print("\n[17] 5-Fold Cross-Validation Scores:")
for model_name, model in trained_models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"  {model_name}: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# ============================================================
# 9. SUMMARY REPORT
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY REPORT")
print("="*60)
print(f"\nDataset Information:")
print(f"  Original Size: {len(df)} samples")
print(f"  After Outlier Removal: {len(df_filtered)} samples")
print(f"  Number of Features: {X.shape[1]}")
print(f"  Number of Classes: {len(np.unique(y))}")

print(f"\nBest Model Performance:")
print(f"  Model: {best_model_name}")
print(f"  Train Accuracy: {results[best_model_name]['train_acc'] * 100:.2f}%")
print(f"  Test Accuracy: {results[best_model_name]['test_acc'] * 100:.2f}%")

print(f"\nPCA Analysis:")
print(f"  Variance Explained (2 components): {sum(pca.explained_variance_ratio_):.2%}")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
