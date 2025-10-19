"""
Task 1: Classical ML with Scikit-learn
Dataset: Iris Species Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("IRIS SPECIES CLASSIFICATION")
print("="*60)

# Load Data
iris = load_iris()
X = iris.data
y = iris.target

print(f"\nDataset shape: {X.shape}")
print(f"Classes: {iris.target_names}")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train Model
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

print("\n✓ Model trained!")

# Predictions
y_pred = dt.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\n📊 RESULTS:")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔢 Confusion Matrix:")
print(cm)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Decision Tree
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, 
          filled=True, ax=axes[0,0], fontsize=10)
axes[0,0].set_title('Decision Tree Structure', fontsize=14, fontweight='bold')

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[0,1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('True Label')
axes[0,1].set_xlabel('Predicted Label')

# Feature Importance
importance = dt.feature_importances_
indices = np.argsort(importance)[::-1]
axes[1,0].bar(range(len(importance)), importance[indices], color='steelblue')
axes[1,0].set_xticks(range(len(importance)))
axes[1,0].set_xticklabels([iris.feature_names[i] for i in indices], rotation=45, ha='right')
axes[1,0].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Importance')

# Actual vs Predicted
axes[1,1].scatter(range(len(y_test)), y_test, alpha=0.7, label='Actual', s=100)
axes[1,1].scatter(range(len(y_pred)), y_pred, alpha=0.7, label='Predicted', s=100, marker='x')
axes[1,1].set_yticks([0,1,2])
axes[1,1].set_yticklabels(iris.target_names)
axes[1,1].set_xlabel('Sample Index')
axes[1,1].set_ylabel('Species')
axes[1,1].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/iris_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Results saved to 'outputs/iris_results.png'")
plt.show()

print("\n" + "="*60)
print("TASK 1 COMPLETE ✓")
print("="*60)