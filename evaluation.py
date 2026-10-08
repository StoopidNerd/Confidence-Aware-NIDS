import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD CALIBRATED CONFIDENCE RESULTS
# ============================================================

df = pd.read_csv("calibrated_confidence_results.csv")

print("Dataset loaded successfully.")
print("Total predictions:", len(df))


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Actual",
    "Predicted",
    "Correct",
    "Raw_Confidence",
    "Calibrated_Confidence"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nMissing columns:")
    print(missing_columns)
    print("\nAvailable columns:")
    print(df.columns.tolist())
    raise ValueError("Required columns are missing.")


# ============================================================
# 3. BASIC CONFIDENCE ANALYSIS
# ============================================================

print("\n================================")
print("CONFIDENCE ANALYSIS")
print("================================")

correct_predictions = df[df["Correct"] == True]
wrong_predictions = df[df["Correct"] == False]

print("\nTotal Correct Predictions:",
      len(correct_predictions))

print("Total Wrong Predictions:",
      len(wrong_predictions))

print("\nAverage Raw Confidence:")

print("Correct Predictions:",
      correct_predictions["Raw_Confidence"].mean())

print("Wrong Predictions:",
      wrong_predictions["Raw_Confidence"].mean())

print("\nAverage Calibrated Confidence:")

print("Correct Predictions:",
      correct_predictions["Calibrated_Confidence"].mean())

print("Wrong Predictions:",
      wrong_predictions["Calibrated_Confidence"].mean())


# ============================================================
# 4. CONFIDENCE RANGE ANALYSIS
# ============================================================

confidence_bins = [
    0.0,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    1.0
]

df["Confidence_Range"] = pd.cut(
    df["Calibrated_Confidence"],
    bins=confidence_bins
)

confidence_analysis = (
    df
    .groupby("Confidence_Range", observed=False)["Correct"]
    .agg(["count", "mean"])
)

confidence_analysis["Accuracy (%)"] = (
    confidence_analysis["mean"] * 100
)

print("\n================================")
print("CALIBRATED CONFIDENCE ANALYSIS")
print("================================")

print(confidence_analysis)


# ============================================================
# 5. THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.99
]

threshold_results = []

total_predictions = len(df)
total_wrong_predictions = len(wrong_predictions)


for threshold in thresholds:

    # Predictions trusted automatically
    automatic = df[
        df["Calibrated_Confidence"] >= threshold
    ]

    # Predictions sent to analyst
    analyst = df[
        df["Calibrated_Confidence"] < threshold
    ]

    if len(automatic) == 0:
        continue

    # Accuracy of automatically trusted predictions
    automatic_accuracy = (
        automatic["Correct"].mean() * 100
    )

    # Wrong predictions that were still automatically trusted
    wrong_auto_alerts = (
        automatic["Correct"] == False
    ).sum()

    # Wrong predictions sent to analyst
    wrong_predictions_caught = (
        analyst["Correct"] == False
    ).sum()

    automatic_coverage = (
        len(automatic) / total_predictions
    ) * 100

    analyst_percentage = (
        len(analyst) / total_predictions
    ) * 100

    wrong_predictions_caught_percentage = (
        wrong_predictions_caught /
        total_wrong_predictions
    ) * 100

    threshold_results.append({

        "Threshold":
            threshold,

        "Automatic Alerts":
            len(automatic),

        "Automatic Coverage (%)":
            automatic_coverage,

        "Automatic Accuracy (%)":
            automatic_accuracy,

        "Wrong Auto Alerts":
            wrong_auto_alerts,

        "Analyst Reviews":
            len(analyst),

        "Analyst Review (%)":
            analyst_percentage,

        "Wrong Predictions Caught":
            wrong_predictions_caught,

        "Wrong Predictions Caught (%)":
            wrong_predictions_caught_percentage
    })


threshold_df = pd.DataFrame(threshold_results)


# ============================================================
# 6. DISPLAY THRESHOLD RESULTS
# ============================================================

print("\n================================")
print("ADAPTIVE THRESHOLD RESULTS")
print("================================")

print(
    threshold_df.to_string(index=False)
)


# ============================================================
# 7. SELECT OPERATIONAL THRESHOLD
# ============================================================

# Final threshold selected for the NIDS system.
#
# At 0.99:
# - Automatically trusted predictions are very accurate
# - Most wrong predictions are sent to analyst review
#
# This threshold is intentionally chosen as the
# operational threshold rather than using the old
# accuracy + coverage scoring formula.

OPERATIONAL_THRESHOLD = 0.99


# ============================================================
# 8. APPLY FINAL DECISION
# ============================================================

def analyst_action(confidence):

    if confidence >= OPERATIONAL_THRESHOLD:
        return "Automatic Alert"

    else:
        return "Analyst Review"


df["Action"] = df["Calibrated_Confidence"].apply(
    analyst_action
)


# ============================================================
# 9. FINAL SYSTEM PERFORMANCE
# ============================================================

automatic_predictions = df[
    df["Action"] == "Automatic Alert"
]

analyst_predictions = df[
    df["Action"] == "Analyst Review"
]

automatic_accuracy = (
    automatic_predictions["Correct"].mean()
    * 100
)

wrong_predictions_sent_to_analyst = (
    analyst_predictions["Correct"] == False
).sum()

wrong_predictions_caught_percentage = (
    wrong_predictions_sent_to_analyst /
    total_wrong_predictions
) * 100


print("\n================================")
print("FINAL NIDS DECISION SYSTEM")
print("================================")

print(
    f"\nOperational Threshold: "
    f"{OPERATIONAL_THRESHOLD:.2f}"
)

print(
    f"Automatic Alerts: "
    f"{len(automatic_predictions)}"
)

print(
    f"Automatic Coverage: "
    f"{len(automatic_predictions) / total_predictions * 100:.2f}%"
)

print(
    f"Automatic Accuracy: "
    f"{automatic_accuracy:.2f}%"
)

print(
    f"\nAnalyst Reviews: "
    f"{len(analyst_predictions)}"
)

print(
    f"Analyst Review Percentage: "
    f"{len(analyst_predictions) / total_predictions * 100:.2f}%"
)

print(
    f"\nWrong Predictions Caught "
    f"by Analyst Review: "
    f"{wrong_predictions_caught_percentage:.2f}%"
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

threshold_df.to_csv(
    "adaptive_threshold_results.csv",
    index=False
)

df["Operational_Threshold"] = (
    OPERATIONAL_THRESHOLD
)

df.to_csv(
    "prediction_confidence.csv",
    index=False
)


# ============================================================
# 11. SAVE ONLY MISCLASSIFIED PREDICTIONS
# ============================================================

misclassified = df[
    df["Correct"] == False
]

misclassified.to_csv(
    "misclassified_predictions.csv",
    index=False
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n================================")
print("FILES SAVED")
print("================================")

print("adaptive_threshold_results.csv")
print("prediction_confidence.csv")
print("misclassified_predictions.csv")

print("\nEvaluation completed successfully.")