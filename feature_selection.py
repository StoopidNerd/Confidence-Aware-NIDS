import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
df = pd.read_parquet("cleaned_dataset.parquet")
df=df.sample(n=200000,random_state=42)
X = df.drop(columns=["Label", "ClassLabel","Target"])
y=df["Target"]
selector = VarianceThreshold(threshold=0)

X = pd.DataFrame(selector.fit_transform(X),columns=X.columns[selector.get_support()])

print(X.shape)
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X,y,random_state=42)

mi = pd.Series(mi_scores,index=X.columns).sort_values(ascending=False)

print(mi.head(20))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print(mi.head(20))
print(importance.head(20))
importance.to_csv("feature_importance.csv")