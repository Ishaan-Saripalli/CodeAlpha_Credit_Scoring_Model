import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import numpy as np
data = pd.read_csv(r'C:\Users\DELL\Desktop\Credit_Score_Builder\credit_risk_dataset.csv')
print("Dataset loaded successfully.")
print(data.head())
print(data.info())
print("Missing values per column:\n", data.isnull().sum())
imputer = SimpleImputer(strategy='median')
data['loan_int_rate'] = imputer.fit_transform(data[['loan_int_rate']])

data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent', 'cb_person_default_on_file'], drop_first=True)
scaler = StandardScaler()
numeric_features = ['person_age', 'person_income', 'person_emp_length', 
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                    'cb_person_cred_hist_length']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

X = data.drop('loan_status', axis=1)
y = data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)
y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score:", roc_auc)
