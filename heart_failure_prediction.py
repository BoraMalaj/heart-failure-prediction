#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('heart.csv')
print("Dataset loaded successfully!")
print("Shape:", df.shape)
df.head()


# In[9]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Separate features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Encode categorical columns
X = pd.get_dummies(X)

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeature columns:")
print(X.columns.tolist())


# In[33]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=70)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

print("Decision Tree trained successfully!")
print("Accuracy:", round(dt.score(X_test, y_test), 2))


# In[35]:


plt.figure(figsize=(25, 10))
tree.plot_tree(dt,
               feature_names=X.columns.tolist(),
               class_names=['No Disease', 'Heart Disease'],
               filled=True,
               rounded=True,
               fontsize=10)
plt.title('Decision Tree - Heart Failure Prediction', fontsize=16)
plt.tight_layout()
plt.show()


# In[37]:


importances = pd.Series(dt.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
importances.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Feature Importance - Decision Tree')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
print(importances.head())


# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Separate features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Encode categorical columns
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr = LogisticRegression(max_iter=10000, random_state=42)
lr.fit(X_train, y_train)

print("Logistic Regression trained successfully!")
print("Accuracy:", round(dt.score(X_test, y_test), 2))


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = lr.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['No Disease', 'Heart Disease']))


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# In[45]:


from sklearn.metrics import roc_curve, roc_auc_score

y_prob = lr.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', linewidth=2, 
         label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

print(f"AUC Score: {round(auc, 2)}")


# In[31]:


coefficients = pd.Series(lr.coef_[0], index=X.columns)
coefficients = coefficients.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
coefficients.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Feature Coefficients - Logistic Regression')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nTop 5 Positive Predictors:")
print(coefficients.head())
print("\nTop 5 Negative Predictors:")
print(coefficients.tail())


# In[47]:


from sklearn.naive_bayes import GaussianNB

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

print("Naive Bayes trained successfully!")
print("Accuracy:", round(nb.score(X_test, y_test), 2))


# In[49]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred_nb = nb.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred_nb,
      target_names=['No Disease', 'Heart Disease']))


# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns

cm_nb = confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# In[53]:


from sklearn.metrics import roc_curve, roc_auc_score

y_prob_nb = nb.predict_proba(X_test)[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
auc_nb = roc_auc_score(y_test, y_prob_nb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='green', linewidth=2,
         label=f'ROC Curve (AUC = {auc_nb:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.title('ROC Curve - Naive Bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

print(f"AUC Score: {round(auc_nb, 2)}")


# In[55]:


y_prob_lr = lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='steelblue', linewidth=2,
         label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot(fpr_nb, tpr_nb, color='green', linewidth=2,
         label=f'Naive Bayes (AUC = {auc_nb:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

print("\nModel Comparison:")
print(f"Logistic Regression Accuracy : {round(lr.score(X_test, y_test), 2)}")
print(f"Naive Bayes Accuracy         : {round(nb.score(X_test, y_test), 2)}")
print(f"Logistic Regression AUC      : {round(auc_lr, 2)}")
print(f"Naive Bayes AUC              : {round(auc_nb, 2)}")


# In[57]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get predictions
y_pred_dt = dt.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_nb = nb.predict(X_test)

# Build comparison table
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Logistic Regression', 'Naive Bayes'],
    'Accuracy': [
        round(accuracy_score(y_test, y_pred_dt), 2),
        round(accuracy_score(y_test, y_pred_lr), 2),
        round(accuracy_score(y_test, y_pred_nb), 2)
    ],
    'Precision': [
        round(precision_score(y_test, y_pred_dt), 2),
        round(precision_score(y_test, y_pred_lr), 2),
        round(precision_score(y_test, y_pred_nb), 2)
    ],
    'Recall': [
        round(recall_score(y_test, y_pred_dt), 2),
        round(recall_score(y_test, y_pred_lr), 2),
        round(recall_score(y_test, y_pred_nb), 2)
    ],
    'F1 Score': [
        round(f1_score(y_test, y_pred_dt), 2),
        round(f1_score(y_test, y_pred_lr), 2),
        round(f1_score(y_test, y_pred_nb), 2)
    ]
})

print("Model Comparison Table:")
print(results.to_string(index=False))


# In[59]:


import matplotlib.pyplot as plt
import numpy as np

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
models = ['Decision Tree', 'Logistic Regression', 'Naive Bayes']
colors = ['steelblue', 'green', 'orange']

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

for i, (model, color) in enumerate(zip(models, colors)):
    values = results[results['Model'] == model][metrics].values[0]
    ax.bar(x + i * width, values, width, label=model, 
           color=color, edgecolor='black', alpha=0.8)

ax.set_title('Model Comparison - Accuracy, Precision, Recall, F1 Score', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[63]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_list = [
    (dt, y_pred_dt, 'Decision Tree', 'Blues'),
    (lr, y_pred_lr, 'Logistic Regression', 'Greens'),
    (nb, y_pred_nb, 'Naive Bayes', 'Oranges')
]

for ax, (model, y_pred, name, color) in zip(axes, models_list):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                xticklabels=['No Disease', 'Heart Disease'],
                yticklabels=['No Disease', 'Heart Disease'])
    ax.set_title(f'Confusion Matrix - {name}', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices Comparison', fontsize=14)
plt.tight_layout()
plt.show()


# In[65]:


y_prob_dt = dt.predict_proba(X_test)[:, 1]
y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_prob_nb = nb.predict_proba(X_test)[:, 1]

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)

auc_dt = roc_auc_score(y_test, y_prob_dt)
auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_nb = roc_auc_score(y_test, y_prob_nb)

plt.figure(figsize=(10, 6))
plt.plot(fpr_dt, tpr_dt, color='steelblue', linewidth=2,
         label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot(fpr_lr, tpr_lr, color='green', linewidth=2,
         label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot(fpr_nb, tpr_nb, color='orange', linewidth=2,
         label=f'Naive Bayes (AUC = {auc_nb:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', 
         label='Random Classifier')
plt.title('ROC Curves Comparison - All Models', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[67]:


from sklearn.model_selection import cross_val_score
import numpy as np

models_cv = {
    'Decision Tree': dt,
    'Logistic Regression': lr,
    'Naive Bayes': nb
}

print("Cross Validation Results (5-Fold):")
print("=" * 50)

for name, model in models_cv.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n{name}:")
    print(f"  Scores     : {scores.round(2)}")
    print(f"  Mean       : {scores.mean().round(2)}")
    print(f"  Std        : {scores.std().round(2)}")


# In[73]:


import matplotlib.pyplot as plt
import numpy as np

models_cv = {
    'Decision Tree': dt,
    'Logistic Regression': lr,
    'Naive Bayes': nb
}

colors = ['steelblue', 'green', 'orange']
all_scores = []
names = []

for name, model in models_cv.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    all_scores.append(scores)
    names.append(name)

plt.figure(figsize=(10, 6))
plt.boxplot(all_scores, tick_labels=names, patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.6),
            medianprops=dict(color='red', linewidth=2))
plt.title('Cross Validation - 5 Fold Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[75]:


cv_results = pd.DataFrame({
    'Model': ['Decision Tree', 'Logistic Regression', 'Naive Bayes'],
    'Test Accuracy': [
        round(accuracy_score(y_test, y_pred_dt), 2),
        round(accuracy_score(y_test, y_pred_lr), 2),
        round(accuracy_score(y_test, y_pred_nb), 2)
    ],
    'CV Mean Accuracy': [
        round(cross_val_score(dt, X, y, cv=5).mean(), 2),
        round(cross_val_score(lr, X, y, cv=5).mean(), 2),
        round(cross_val_score(nb, X, y, cv=5).mean(), 2)
    ],
    'CV Std': [
        round(cross_val_score(dt, X, y, cv=5).std(), 2),
        round(cross_val_score(lr, X, y, cv=5).std(), 2),
        round(cross_val_score(nb, X, y, cv=5).std(), 2)
    ]
})

print("Final Model Comparison with Cross Validation:")
print(cv_results.to_string(index=False))

