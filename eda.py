import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
df = pd.read_parquet("cleaned_dataset.parquet")
plt.figure(figsize=(15,8))

df["Label"].value_counts().plot(kind="bar")

plt.title("Attack Class Distribution")
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.xticks(rotation=90)

plt.show()
class_percentage = (
    df["Label"].value_counts(normalize=True) * 100
)

print(class_percentage)
print(df.describe().T)
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()

selector.fit(df.select_dtypes(include="number"))

print(selector.variances_)
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(18,15))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")

plt.show()

encoder = LabelEncoder()

df["Target"] = encoder.fit_transform(df["Label"])
df.to_parquet("cleaned_dataset.parquet", index=False)