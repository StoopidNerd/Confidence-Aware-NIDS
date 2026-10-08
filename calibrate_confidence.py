import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    log_loss
)

from xgboost import XGBClassifier


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_parquet("cleaned_dataset.parquet")

df = df.sample(
    n=200000,
    random_state=42
)


# ==================================================
# 2. CREATE TARGET
# ==================================================

encoder = LabelEncoder()

df["Target"] = encoder.fit_transform(
    df["Label"]
)

# ==================================================
# 3. CREATE X AND y
# ==================================================

X = df.drop(
    columns=[
        "Label",
        "ClassLabel",
        "Target"
    ]
)

y = df["Target"]

print("Number of classes:", y.nunique())

print("\nSmallest classes:")
print(
    y.value_counts()
      .sort_values()
      .head(10)
)
# ==================================================
# 4. TRAIN / CALIBRATION / TEST SPLIT
# ==================================================

# First split:
# 80% = training + calibration
# 20% = final test

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split:
# 75% of 160,000 = 120,000 training
# 25% of 160,000 = 40,000 calibration

X_train, X_calibration, y_train, y_calibration = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\nTraining samples:", len(X_train))
print("Calibration samples:", len(X_calibration))
print("Test samples:", len(X_test))


# ==================================================
# 5. LOAD BEST MODEL
# ==================================================

old_model = joblib.load(
    "best_xgboost.pkl"
)

best_params = old_model.get_params()

model = XGBClassifier(
    **best_params
)


# ==================================================
# 6. TRAIN XGBOOST
# ==================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ==================================================
# 7. GET CALIBRATION PREDICTIONS
# ==================================================

print("\nGenerating calibration predictions...")

calibration_probabilities = model.predict_proba(
    X_calibration
)

calibration_predictions = model.predict(
    X_calibration
)

calibration_confidence = np.max(
    calibration_probabilities,
    axis=1
)

# Was each prediction correct?
calibration_correct = (
    calibration_predictions ==
    y_calibration.values
).astype(int)


# ==================================================
# 8. GET FINAL TEST PREDICTIONS
# ==================================================

test_probabilities = model.predict_proba(
    X_test
)

test_predictions = model.predict(
    X_test
)

test_confidence = np.max(
    test_probabilities,
    axis=1
)


# ==================================================
# 9. MANUAL CONFIDENCE CALIBRATION
# ==================================================

print("\nCalibrating confidence...")

from sklearn.linear_model import LogisticRegression

epsilon = 1e-6

# Prevent log(0)
calibration_confidence = np.clip(
    calibration_confidence,
    epsilon,
    1 - epsilon
)

# Convert confidence to log-odds
calibration_log_odds = np.log(
    calibration_confidence /
    (1 - calibration_confidence)
)

# Learn relationship:
# raw confidence → probability prediction is correct

calibrator = LogisticRegression()

calibrator.fit(
    calibration_log_odds.reshape(-1, 1),
    calibration_correct
)

print("Confidence calibration completed.")


# ==================================================
# 10. CALIBRATE FINAL TEST CONFIDENCE
# ==================================================

test_confidence_clipped = np.clip(
    test_confidence,
    epsilon,
    1 - epsilon
)

test_log_odds = np.log(
    test_confidence_clipped /
    (1 - test_confidence_clipped)
)

calibrated_confidence = calibrator.predict_proba(
    test_log_odds.reshape(-1, 1)
)[:, 1]


# ==================================================
# 11. MODEL ACCURACY
# ==================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    "XGBoost Accuracy:",
    accuracy
)


# ==================================================
# 12. CONFIDENCE ANALYSIS
# ==================================================

correct = (
    test_predictions ==
    y_test.values
)

print("\n================================")
print("CONFIDENCE ANALYSIS")
print("================================")

print(
    "Average raw confidence:",
    np.mean(test_confidence)
)

print(
    "Average calibrated confidence:",
    np.mean(calibrated_confidence)
)

print(
    "\nCalibrated confidence - CORRECT:",
    np.mean(
        calibrated_confidence[correct]
    )
)

print(
    "Calibrated confidence - WRONG:",
    np.mean(
        calibrated_confidence[~correct]
    )
)


# ==================================================
# 13. CONFIDENCE BINS
# ==================================================

results = pd.DataFrame({

    "Actual":
        y_test.values,

    "Predicted":
        test_predictions,

    "Raw_Confidence":
        test_confidence,

    "Calibrated_Confidence":
        calibrated_confidence,

    "Correct":
        correct
})


results["Confidence_Range"] = pd.cut(
    results["Calibrated_Confidence"],
    bins=[
        0.0,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        1.0
    ]
)


analysis = (
    results
    .groupby(
        "Confidence_Range",
        observed=True
    )["Correct"]
    .agg(["count", "mean"])
)

analysis["Accuracy (%)"] = (
    analysis["mean"] * 100
)


print("\n================================")
print("CALIBRATION RESULTS")
print("================================")

print(analysis)


# ==================================================
# 14. SAVE RESULTS
# ==================================================

results.to_csv(
    "calibrated_confidence_results.csv",
    index=False
)


# ==================================================
# 15. SAVE MODELS
# ==================================================

joblib.dump(
    model,
    "final_xgboost.pkl"
)

joblib.dump(
    calibrator,
    "confidence_calibrator.pkl"
)


print("\n================================")
print("FILES SAVED")
print("================================")

print("calibrated_confidence_results.csv")
print("final_xgboost.pkl")
print("confidence_calibrator.pkl")