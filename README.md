# 🧠 Machine Learning Projects

This repository contains three machine learning projects that demonstrate **supervised**, **unsupervised**, and **regression-based** learning workflows using Python and scikit-learn.  
Each project includes end-to-end data preprocessing, feature engineering, model building, evaluation, and visualization steps.

---

## 📚 Projects Overview

| Project | Description | Algorithm | Link |
|----------|--------------|------------|------|
| 🩺 **Heart Disease Prediction** | Predicts the likelihood of heart disease based on clinical and lifestyle data | K-Nearest Neighbors (KNN) | [View Project](./heart_disease_prediction/README.md) |
| 🛍️ **Mall Customer Classification** | Clusters mall customers by income and spending patterns to identify customer segments | K-Means Clustering | [View Project](./mall_customer_classification/README.md) |
| 🌦️ **Weather Prediction** | Predicts tomorrow’s maximum temperature using historical weather features and regression modeling | Ridge Regression | [View Project](./weather_prediction/README.md) |

---

## ⚙️ Environment Setup

```bash
# Clone the repository
git clone https://github.com/LianaHuang99/MachineLearningProjects.git
cd MachineLearningProjects

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # for macOS/Linux
# venv\Scripts\activate   # for Windows

# Install dependencies
pip install -r requirements.txt
