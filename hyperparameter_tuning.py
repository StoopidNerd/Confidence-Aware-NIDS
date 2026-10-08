from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib 
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [20, None],
    "min_samples_split": [2, 5]
}
rf=RandomForestClassifier(random_state=42,n_jobs=-1)
rf_grid =   RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid_rf,
    scoring="f1_weighted",
    cv=5,
    n_jobs=-1,
    verbose=2,random_state=42
)
df=pd.read_parquet("cleaned_dataset.parquet")
df=df.sample(n=200000,random_state=42)
encoder=LabelEncoder()
df["Target"]=encoder.fit_transform(df["Label"])
X = df.drop(columns=["Label","ClassLabel",  "Target"])
y = df["Target"]
print(X.select_dtypes(include="object").columns)
print(df.columns)
print(df.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
rf_grid.fit(X_train, y_train)
print("Best Parameters:")
print(rf_grid.best_params_)

print("Best Cross Validation F1:")
print(rf_grid.best_score_)
best_rf = rf_grid.best_estimator_

y_pred = best_rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1:", f1_score(y_test, y_pred, average="weighted"))
joblib.dump(best_rf, "best_random_forest.pkl")
xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist"
)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
xgb_grid = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    scoring="f1_weighted",
    cv=5,
    n_jobs=-1,
    verbose=2,random_state=42
)
xgb_grid.fit(X_train, y_train)
print("Best Parameters:")
print(xgb_grid.best_params_)

print("Best F1 Score:")
print(xgb_grid.best_score_)
best_xgb = xgb_grid.best_estimator_
y_pred = best_xgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted",zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1:", f1_score(y_test, y_pred, average="weighted"))
joblib.dump(best_xgb, "best_xgboost.pkl")
best_xgb = joblib.load("best_xgboost.pkl")
