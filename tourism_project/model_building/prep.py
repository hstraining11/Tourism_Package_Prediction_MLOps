import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(RAW_PATH)

df.drop(columns=["CustomerID"], inplace=True)

# NOTE: Categorical columns like 'TypeofContact', 'Occupation', 'Gender',
#'MaritalStatus', 'ProductPitched' are intentionally left as raw strings. The training pipeline will one-hot-encode them.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) ProdTaken ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
