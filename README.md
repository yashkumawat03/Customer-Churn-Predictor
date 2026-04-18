# 📉 Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer is likely to churn, explains the reasons, and suggests retention strategies.

🔗 **Live App:** https://customer-churn-predictor-yash.streamlit.app/

---

## 📌 Problem Statement
Customer churn is one of the biggest challenges in the telecom industry. This project builds an end-to-end ML pipeline to identify at-risk customers before they leave.

---

## 🛠️ Tech Stack
- **Python** — pandas, numpy, scikit-learn, XGBoost
- **ML** — Random Forest with GridSearchCV tuning
- **Explainability** — SHAP
- **Imbalance Handling** — SMOTE
- **Deployment** — Streamlit Community Cloud

---

## 📊 Model Performance
| Metric | Score |
|---|---|
| Accuracy | 84.03% |
| Precision | 0.84 |
| Recall | 0.84 |
| F1 Score | 0.84 |
| ROC-AUC | **0.92** |

---

## 🔍 Key Insights from SHAP
- **Contract type** is the #1 churn driver
- **High monthly charges** significantly increase churn risk
- **Low tenure** customers are most vulnerable
- Customers without **Online Security** or **Tech Support** churn more

---

## 🗂️ Project Structure
customer-churn-predictor/
├── data/                  # Dataset
├── notebooks/             # EDA notebook
├── src/
│   ├── preprocess.py      # Data preprocessing pipeline
│   ├── train.py           # Model training + tuning
│   ├── evaluate.py        # Evaluation metrics + plots
│   └── explain.py         # SHAP explainability
├── app.py                 # Streamlit web app
└── requirements.txt

---

## 🚀 Run Locally
git clone https://github.com/yourusername/customer-churn-predictor
cd customer-churn-predictor
pip install -r requirements.txt
streamlit run app.py

---

## 📬 Contact
**Yash Kumawat** — [LinkedIn](https://linkedin.com/in/yash-kumawat03/)