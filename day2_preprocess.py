import pandas as pd

train_df = pd.read_parquet("data/train_raw.parquet")
test_df = pd.read_parquet("data/test_raw.parquet")

# Drop the KDD-specific 'difficulty' column — not a real traffic feature
train_df = train_df.drop(columns=["difficulty"])
test_df = test_df.drop(columns=["difficulty"])

# Binary target: 0 = normal, 1 = attack
train_df["target"] = (train_df["label"] != "normal").astype(int)
test_df["target"] = (test_df["label"] != "normal").astype(int)

print(train_df["target"].value_counts())
print(test_df["target"].value_counts())

from sklearn.preprocessing import LabelEncoder

cat_cols = ["protocol_type", "service", "flag"]

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    # test set may have unseen categories — map them safely
    test_df[col] = test_df[col].map(
        lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
    )

print(train_df[cat_cols].head())
print(test_df[cat_cols].head())

from sklearn.preprocessing import StandardScaler

feature_cols = [c for c in train_df.columns if c not in ["label", "target"]]

scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

print(train_df[feature_cols].describe().loc[["mean", "std"]])

feature_cols = [c for c in train_df.columns if c not in ["label", "target"]]

X_train = train_df[feature_cols]
y_train = train_df["target"]
X_test = test_df[feature_cols]
y_test = test_df["target"]

X_train.to_parquet("data/X_train.parquet", index=False)
y_train.to_frame().to_parquet("data/y_train.parquet", index=False)
X_test.to_parquet("data/X_test.parquet", index=False)
y_test.to_frame().to_parquet("data/y_test.parquet", index=False)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Saved processed files to data/")
