# 🛍️ Mall Customer Classification using K-Means Clustering

This project uses **unsupervised learning** to segment mall customers based on spending patterns and annual income.

---

## 📘 Overview
Customer segmentation helps businesses design targeted marketing campaigns.  
In this project, the **Mall Customers Dataset** is analyzed using **K-Means clustering** to identify groups with similar purchasing behavior.

---

## 🧾 Dataset Description

| Feature | Description |
|----------|--------------|
| CustomerID | Unique customer ID |
| Gender | Male / Female |
| Age | Age of the customer |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1–100) | Score assigned by mall based on spending behavior |

---

## ⚙️ Methodology
1. **Data Cleaning and EDA**
   - Checked for missing values and outliers  
   - Visualized distributions and correlations
2. **Feature Scaling**
   - Standardized data using `StandardScaler`
3. **K-Means Clustering**
   - Determined optimal k using the **Elbow Method**
   - Applied clustering and labeled customer segments
4. **Visualization**
   - Plotted clusters by **Annual Income vs Spending Score**

---

## 📊 Key Insights
- 5 customer segments were identified  
- **High-income, low-spending** group represents potential marketing opportunities  
- **Young, high-spending** cluster indicates loyal premium shoppers  

---

## 🧰 Tools Used
Python • pandas • NumPy • scikit-learn • Matplotlib • Seaborn • Jupyter Notebook
