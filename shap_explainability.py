import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df = pd.read_parquet("cleaned_dataset.parquet")

encoder = LabelEncoder()
df["Target"] = encoder.fit_transform(df["Label"])

X = df.drop(columns=["Label", "ClassLabel", "Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
best_xgb = joblib.load("best_xgboost.pkl")
explainer = shap.TreeExplainer(best_xgb)
X_sample = X_test.sample(500, random_state=42)

shap_values = explainer.shap_values(X_sample)
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
sample = X_sample.iloc[[0]]

explanation = explainer(sample)

predicted_class = int(best_xgb.predict(sample)[0])

shap.plots.force(
    explanation[0, :, predicted_class],
    matplotlib=True,
    show=False
)

plt.savefig(
    "force_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
sample = X_sample.iloc[[0]]

# Get explanation
explanation = explainer(sample)

# Predicted class
predicted_class = int(best_xgb.predict(sample)[0])

# Plot explanation only for the predicted class
shap.plots.waterfall(
    explanation[0, :, predicted_class],
    show=False
)

plt.savefig(
    "waterfall_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
sample = X_sample.iloc[[0]]

explanation = explainer(sample)

print(type(explanation))
print(explanation.values.shape)