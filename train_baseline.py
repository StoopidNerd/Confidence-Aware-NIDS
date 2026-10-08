import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
import time
results=[]
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start

    print(f"\n========== {model_name} ==========")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        xticks_rotation=90
    )
    plt.title(model_name)
    plt.show()
    results.append({
    "Model": model_name,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="weighted"),
    "Recall": recall_score(y_test, y_pred, average="weighted"),
    "F1 Score": f1_score(y_test, y_pred, average="weighted"),
    "Training Time": training_time,
    "Prediction Time": prediction_time
})
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
print("Unique classes in y:", sorted(y.unique()))
print("Number of classes:", y.nunique())

print("Unique classes in y_train:", sorted(y_train.unique()))
print("Number of classes in y_train:", y_train.nunique())
rf = RandomForestClassifier(n_estimators=200, random_state=42)
evaluate_model(rf, "Random Forest")

xgb = XGBClassifier(
    n_estimators=200,
    random_state=42,
    eval_metric="mlogloss",tree_method="hist"
)
evaluate_model(xgb, "XGBoost")

lgbm = LGBMClassifier(
    n_estimators=200,
    random_state=42
)
evaluate_model(lgbm, "LightGBM")
print(results)
comparison = pd.DataFrame(results)

print(comparison)

comparison.to_csv(
    "model_comparison.csv",
    index=False
)
print("CSV saved successfully.")