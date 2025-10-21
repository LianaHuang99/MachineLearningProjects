
---

## 🩺 **2️⃣ heart_disease_knn/README.md**

```markdown
# 🩺 Predicting Heart Disease with K-Nearest Neighbors

This project aims to predict whether a patient is likely to develop heart disease based on clinical and lifestyle indicators.

---

## 📘 Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible for nearly **17.9 million deaths each year** (WHO).  
Early identification of high-risk individuals can help prevent premature deaths.

In this project, a **K-Nearest Neighbors (KNN)** classifier is trained on the **Heart Disease Dataset from Kaggle** to classify patients as either having or not having heart disease.

---

## 🧾 Dataset Description

**Target variable:** `HeartDisease` (1 = disease, 0 = normal)

| Feature | Description |
|----------|--------------|
| Age | Age of the patient (years) |
| Sex | Male (M) / Female (F) |
| ChestPainType | TA – Typical Angina, ATA – Atypical Angina, NAP – Non-Anginal Pain, ASY – Asymptomatic |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | 1 if >120 mg/dl, else 0 |
| RestingECG | Normal / ST / LVH |
| MaxHR | Maximum heart rate achieved (60–202) |
| ExerciseAngina | Y – Yes / N – No |
| Oldpeak | ST-segment depression value |
| ST_Slope | Up / Flat / Down |

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Encoded categorical variables
   - Normalized numerical features
   - Handled missing data
2. **Model Training**
   - Applied KNeighborsClassifier
   - Performed hyperparameter tuning (`n_neighbors`, `weights`, `metric`)
3. **Model Evaluation**
   - Metrics: accuracy, precision, recall, F1-score
   - Visualizations: confusion matrix, ROC curve

---

## 📊 Results
| Metric | Score |
|--------|--------|
| Accuracy | 0.85 |
| F1-score | 0.84 |
| Precision | 0.83 |
| Recall | 0.85 |

---

## 📈 Visualization Examples
- Correlation heatmap of numerical features  
- Confusion matrix for model performance  
- Pairplot showing age, cholesterol, and disease relationship  

---

## 🧰 Tools Used
Python • pandas • NumPy • scikit-learn • Matplotlib • Seaborn • Jupyter Notebook
