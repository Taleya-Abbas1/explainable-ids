"""
Day 1 — Explainable Network Intrusion Detection
Setup + Dataset loading & exploration (NSL-KDD)
"""

import pandas as pd

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

TRAIN_PATH = "data/KDDTrain+.txt"
TEST_PATH = "data/KDDTest+.txt"


def load_nsl_kdd(path: str) -> pd.DataFrame:
    return pd.read_csv(path, names=COLUMN_NAMES)


def explore(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'=' * 60}\n{name}  —  shape: {df.shape}\n{'=' * 60}")
    print("\nMissing values:", df.isnull().sum().sum())

    print("\nCategorical columns:")
    for col in ["protocol_type", "service", "flag"]:
        print(f"  {col}: {df[col].nunique()} unique -> {df[col].unique()[:8]}")

    print("\nTop 15 labels:")
    print(df["label"].value_counts().head(15))

    binary = df["label"].apply(lambda x: "normal" if x == "normal" else "attack")
    print("\nBinary class balance:")
    print(binary.value_counts(normalize=True).round(3))


if __name__ == "__main__":
    train_df = load_nsl_kdd(TRAIN_PATH)
    test_df = load_nsl_kdd(TEST_PATH)

    explore(train_df, "KDDTrain+")
    explore(test_df, "KDDTest+")

    train_df.to_parquet("data/train_raw.parquet", index=False)
    test_df.to_parquet("data/test_raw.parquet", index=False)
    print("\nSaved: data/train_raw.parquet, data/test_raw.parquet")
