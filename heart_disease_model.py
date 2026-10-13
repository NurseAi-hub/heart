import pandas as pd

# Load real UCI Heart Disease dataset directly
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print(df.head())
print("Missing values per column:")
print(df.isnull().sum())
# Check value counts (frequency of each class)
target_counts = df['target'].value_counts()
print("Target Value Counts:")
print(target_counts)

# Check proportions/percentages
target_percentages = df['target'].value_counts(normalize=True) * 100
print("\\nTarget Percentages:")
print(target_percentages.round(2))
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
print(df[continuous_cols].describe())
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f'Distribution of {col}', fontsize=12)

# Hide the empty 6th subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

# 1. Separate target variable from features
X = df.drop(columns=['target'])
y = df['target']

# 2. Define categorical features to dummy-encode
categorical_cols = ['cp', 'restecg', 'slope', 'thal']

# 3. Apply One-Hot Encoding using pandas
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 4. Scale numerical continuous columns
scaler = StandardScaler()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

print("New dataset shape after encoding:", X_encoded.shape)
X_encoded.head()
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state= 42, stratify=y)
print(f"Training set size: {X_train.shape[0]} patients")
print(f"Testing set size: {X_test.shape[0]} patients")
cv_strategy = StratifiedKFold(n_splits = 5, shuffle= True, random_state= 42)
print("Cross-validation strategy ready: 5 Folds")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

#random forest
rf_pro = RandomForestClassifier(
    n_estimators=300,                 
    class_weight='balanced_subsample',
    max_depth=12,                      
    min_samples_leaf=5,               
    n_jobs=-1,                         
    random_state=42
)
rf_pro.fit(X_train, y_train)
y_pred_pro = rf_pro.predict(X_test)
y_prob_pro = rf_pro.predict_proba(X_test)[:, 1] # نحتاج الاحتمالات لحساب ROC-AUC
print("--- Classification Report ---")
print(classification_report(y_test, y_pred_pro))
accuracy = rf_pro.score(X_test, y_test) * 100
print("--- Random Forest Results ---")
print(f"Accuracy: {accuracy:.2f}%")
roc_auc = roc_auc_score(y_test, y_prob_pro) * 100
print(f"ROC-AUC Score (The true performance): {roc_auc:.2f}%")

#logistic regression:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
# Initialize the model (no need to scale again, it was done in Step 2!)
model = LogisticRegression(max_iter=1000, random_state=42)
# Fit using ONLY the training data
model.fit(X_train, y_train)
# Predict using ONLY the testing data
y_pred = model.predict(X_test)
y_prod = model.predict_proba(X_test)[:, 1]
# Calculate and print metrics
accur = accuracy_score(y_test, y_pred)
print("--- Logistic Regression Results ---")
print(f"Accuracy: {accur * 100:.2f}%")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prod) * 100:.2f}%")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

import pandas as pd
import matplotlib.pyplot as plt

# Extract feature importances from Random Forest
importances = rf_pro.feature_importances_
feature_names = X_train.columns

# Create DataFrame and sort
df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df_importance = df_importance.sort_values(by='Importance', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 5))
plt.barh(df_importance['Feature'][:10], df_importance['Importance'][:10], color='teal')
plt.gca().invert_yaxis()
plt.title('Top 10 Clinical Features Driving Heart Disease Predictions')
plt.xlabel('Importance Score')
plt.show()

from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(rf_pro, X_test, y_test, ax=ax, name="Random Forest")
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="Logistic Regression")
plt.title("ROC Curve Comparison")
plt.show()
