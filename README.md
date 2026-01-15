# Telco Customer Churn Prediction

An end-to-end machine learning project to predict customer churn in the telecom industry using structured customer data.  
The focus of this project is **business-oriented modeling**, emphasizing recall optimization and decision-threshold tuning rather than accuracy alone.

---

## 🌐 Live Demo
🚀 **Streamlit App**:  
👉 https://telco-customer-churn-classifier.streamlit.app/

---

## 📌 Problem Statement
Customer churn is a major challenge for telecom companies.  
The goal of this project is to identify customers who are likely to churn so that retention strategies can be applied proactively.

---

## 📊 Dataset
- **Source**: Telco Customer Churn Dataset (Kaggle)
- **Rows**: 7,043
- **Target Variable**: `Churn` (Yes / No)
- **Feature Types**:
  - Numerical: tenure, MonthlyCharges, TotalCharges
  - Binary categorical
  - Nominal categorical
  - Ordinal categorical

---

## ⚙️ Workflow
1. Exploratory Data Analysis (EDA)
2. Feature preprocessing using `Pipeline` and `ColumnTransformer`
3. Handling class imbalance using:
   - `class_weight='balanced'`
   - `scale_pos_weight` for XGBoost
4. Model training and comparison:
   - Logistic Regression
   - SVM
   - Random Forest
   - XGBoost
5. Hyperparameter tuning using `GridSearchCV`
6. Threshold tuning to maximize churn recall
7. Final model selection and evaluation
8. Model serialization using `pickle`

---

## 📈 Model Performance (Tuned)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.735 | 0.500 | 0.783 | 0.610 | 0.840 |
| SVM | 0.722 | 0.486 | 0.818 | 0.610 | 0.839 |
| Random Forest | **0.770** | **0.548** | 0.767 | **0.639** | 0.843 |
| XGBoost | 0.744 | 0.512 | **0.799** | 0.624 | **0.845** |

---

## 🎯 Final Model
- **Model**: XGBoost
- **Decision Threshold**: `0.35`
- **Recall Achieved**: ~89%
- **Why**: Maximizes churn capture while keeping precision acceptable

---

## 🧠 Key Insights
- Accuracy alone is misleading for churn prediction
- Class imbalance handling significantly improves recall
- Threshold tuning is critical for real business impact
- Tree-based models perform best after proper tuning

---

## 🛠️ Tech Stack
- Python
- pandas, numpy
- scikit-learn
- XGBoost
- matplotlib, seaborn

---

