# ❤️ Clinical Heart Disease Risk Stratification & Model Evaluation

## 🎯 Executive Summary
This evaluation compared a Random Forest Classifier against a baseline Logistic Regression model using a 303-patient clinical heart disease dataset. Both models achieved identical baseline accuracy (80.33%), but the Random Forest demonstrated superior overall discriminatory power with a ROC-AUC of 91.13% (compared to 89.18% for Logistic Regression). While Logistic Regression yielded a slightly higher sensitivity for detecting active heart disease (Recall of 88% vs. 85%), Random Forest remains the recommended primary architecture due to its higher overall feature interaction handling and superior class separation.

---

## ⚖️ Clinical Trade-off Analysis
From a clinical and nursing triage perspective, **Recall (Sensitivity)** is the single most critical performance metric in heart disease screening.

- **The Danger of False Negatives (Low Recall):** A False Negative occurs when the model predicts a patient is healthy (0) when they actually have underlying heart disease (1). In a clinical workflow, sending a high-risk cardiac patient home without follow-up care or immediate intervention can lead to unmonitored ischemic events, heart failure exacerbation, or fatal cardiac arrest.
- **The Cost of False Positives (Low Precision):** Conversely, a False Positive results in a healthy patient receiving unnecessary diagnostic workups (e.g., troponin draws, echocardiograms, stress testing). While this causes minor resource strain and patient anxiety, it carries zero mortality risk.

Therefore, clinical model tuning must explicitly prioritize high Recall to minimize False Negatives, accepting higher rates of False Positives as an acceptable safety trade-off.

---

## ⚠️ Deployment Safety & Risk Note
Deploying this specific model in a real-time Emergency Department (ED) triage setting introduces several critical safety risks:

1. **Inappropriate Triage Downgrading:** ED triage relies on rapid identification of life-threatening conditions (e.g., Acute Coronary Syndrome). If nurses rely on an automated AI score that misclassifies a high-risk atypical presentation (e.g., elderly or diabetic patients with non-standard chest pain) as "Low Risk," patient care will be delayed.
2. **Automation Bias:** Busy clinical staff facing high patient volumes may over-rely on the AI output, leading to "cognitive offloading" and reducing independent clinical assessment.
3. **Dataset Size & Population Shift:** The model was trained on a small sample of 303 patients. It lacks generalizability across diverse demographics, comorbidity profiles, and real-time artifact-laden diagnostic inputs.

---

## 📊 Key Clinical Insights & Feature Importance
Based on the Random Forest feature importance analysis, the model identified key clinical indicators as the primary drivers for predicting heart disease risk[cite: 1]:
* **`thal_2` (Thalassemia defect type 2 / Reversible defect):** Rank #1 feature strongly correlating with myocardial ischemia.
* **`oldpeak` (ST depression induced by exercise relative to rest):** A classic ECG sign indicating ischemic heart disease during physical stress.
* **`thal_3` (Fixed defect):** Critical category indicating prior myocardial infarction.
* **`ca` (Number of major vessels colored by fluoroscopy):** Directly measures coronary artery blockage.
* **`exang` (Exercise-induced angina):** Chest pain triggered by exertion, a primary symptom of coronary artery disease.

---

## 📈 Model Performance & Visualizations
* **Random Forest Classifier (AUC = 0.91):** Demonstrated superior overall class separation and non-linear feature interaction handling.
* **Logistic Regression (AUC = 0.89):** Provided a strong, interpretable linear baseline with high sensitivity.

---

## 🛠️ Tech Stack & Model Files
- **Python:** Pandas, NumPy, Scikit-Learn
- **Models:** Logistic Regression, Random Forest Classifier
- **Evaluation:** ROC-AUC Curves, Confusion Matrices, Classification Reports
- **Code Implementation:** `heart_disease_model.py`
