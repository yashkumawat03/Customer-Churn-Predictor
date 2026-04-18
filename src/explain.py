import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd
from preprocess import load_and_preprocess

# Load data and model
X_train, X_test, y_train, y_test, scaler = load_and_preprocess(
    'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature names
import pandas as pd_orig
df = pd_orig.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.drop(columns=['customerID', 'Churn'], inplace=True)
feature_names = df.columns.tolist()

# Create SHAP explainer
print("Calculating SHAP values... (takes ~30 seconds)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot 1 — Summary Bar Plot (most important features overall)
shap.summary_plot(
    shap_values[:, :, 1],
    X_test,
    feature_names=feature_names,
    plot_type='bar',
    show=False
)
plt.title('Top Features Driving Churn')
plt.tight_layout()
plt.savefig('models/shap_bar.png')
plt.show()
print("SHAP bar plot saved.")

# Plot 2 — Beeswarm Plot (how each feature affects predictions)
shap.summary_plot(
    shap_values[:, :, 1],
    X_test,
    feature_names=feature_names,
    show=False
)
plt.title('SHAP Beeswarm Plot')
plt.tight_layout()
plt.savefig('models/shap_beeswarm.png')
plt.show()
print("SHAP beeswarm plot saved.")