import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns

# Fake prediction scores and true labels
y_true = np.random.randint(0, 2, 1000)
y_scores_old = np.random.rand(1000)
y_scores_new = np.clip(y_scores_old + np.random.normal(0, 0.05, 1000), 0, 1)

# 1. ROC Curve
fpr_old, tpr_old, _ = roc_curve(y_true, y_scores_old)
fpr_new, tpr_new, _ = roc_curve(y_true, y_scores_new)
roc_auc_old = auc(fpr_old, tpr_old)
roc_auc_new = auc(fpr_new, tpr_new)

plt.figure(figsize=(8, 5))
plt.plot(fpr_old, tpr_old, label=f'Current Model (AUC = {roc_auc_old:.2f})', linestyle='--')
plt.plot(fpr_new, tpr_new, label=f'New GAN Model (AUC = {roc_auc_new:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Precision-Recall Curve
precision_old, recall_old, _ = precision_recall_curve(y_true, y_scores_old)
precision_new, recall_new, _ = precision_recall_curve(y_true, y_scores_new)

plt.figure(figsize=(8, 5))
plt.plot(recall_old, precision_old, label='Current Model', linestyle='--')
plt.plot(recall_new, precision_new, label='New GAN Model', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Confusion Matrix (fake predictions)
y_pred_old = (y_scores_old > 0.5).astype(int)
y_pred_new = (y_scores_new > 0.5).astype(int)

conf_matrix_old = confusion_matrix(y_true, y_pred_old)
conf_matrix_new = confusion_matrix(y_true, y_pred_new)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_old, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Current Model Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_new, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('New GAN Model Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

# 4. Bar Chart for Metric Comparison
metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
current = [0.94, 0.95, 0.945, 0.97]
new = [0.96, 0.97, 0.965, 0.98]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, current, width, label='Current Model')
plt.bar(x + width/2, new, width, label='New GAN Model')
plt.xticks(x, metrics)
plt.ylim(0.9, 1.0)
plt.ylabel('Score')
plt.title('Metric Comparison Between Models')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
