import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess

# Load preprocessed data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess(
    'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)

# GridSearchCV — tries all combinations, picks the best
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

print("Starting Grid Search... (this will take 2-4 minutes)")
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred = best_model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(f"\nBest Parameters: {best_params}")
print(f"Tuned Accuracy: {score:.4f}")

# Save
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Tuned model saved to models/best_model.pkl")