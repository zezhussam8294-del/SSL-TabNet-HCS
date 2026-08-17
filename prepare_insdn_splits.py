import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# 1) SETTINGS
# =========================================================

# Repository-relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATH = os.path.join(
    SCRIPT_DIR,
    "data",
    "InSDN",
    "InSDN_Normal_and_Attack_Combined.csv"
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "data",
    "InSDN"
)

LABEL_COL = "Label"
BENIGN_LABEL = "Normal"

# Open-set configuration
KNOWN_ATTACK_LABELS = ["DDoS", "Probe"]
UNSEEN_ATTACK_LABELS = ["DoS", "BFA", "U2R"]

# Exact duplicate removal
REMOVE_DUPLICATES = True

RANDOM_STATE = 42

# 20% test for known classes and Normal
TEST_SIZE = 0.20

# 25% of the remaining 80% = 20% of the total dataset
# Final split for known classes: 60% train, 20% validation, 20% test
VAL_SIZE_FROM_REMAIN = 0.25

# Non-feature columns removed before creating the split files
DROP_NON_NUMERIC = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp"
]

# Constant columns identified during dataset inspection
CONSTANT_COLS = [
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
    "Init Fwd Win Byts",
    "Fwd Seg Size Min"
]


# =========================================================
# 2) CREATE OUTPUT DIRECTORY
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 3) LOAD DATA
# =========================================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"InSDN source file was not found:\n{FILE_PATH}\n\n"
        "Place InSDN_Normal_and_Attack_Combined.csv inside data/InSDN/ "
        "and run this script again."
    )

df = pd.read_csv(FILE_PATH)

print("Original shape:", df.shape)

# Clean label spacing
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Drop rows with missing labels
df = df.dropna(subset=[LABEL_COL]).copy()

# Optional exact duplicate removal
if REMOVE_DUPLICATES:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    print(f"Duplicates removed: {before - after}")


# =========================================================
# 4) CHECK AND FILTER LABELS
# =========================================================
print("\nLabel distribution before filtering:")
print(df[LABEL_COL].value_counts())

target_labels = (
    [BENIGN_LABEL]
    + KNOWN_ATTACK_LABELS
    + UNSEEN_ATTACK_LABELS
)

df = df[df[LABEL_COL].isin(target_labels)].copy()

print("\nLabel distribution after filtering:")
print(df[LABEL_COL].value_counts())


# =========================================================
# 5) DROP NON-FEATURE COLUMNS
# =========================================================
feature_drop_cols = [
    c for c in DROP_NON_NUMERIC + CONSTANT_COLS
    if c in df.columns
]

# Preserve the original class label
df["OriginalLabel"] = df[LABEL_COL]

# Build feature dataframe
X_df = df.drop(
    columns=feature_drop_cols,
    errors="ignore"
).copy()

# Ensure that only numeric features remain
feature_cols = [
    c for c in X_df.columns
    if c not in [LABEL_COL, "OriginalLabel"]
]

non_numeric_after_drop = (
    X_df[feature_cols]
    .select_dtypes(exclude=[np.number])
    .columns
    .tolist()
)

if non_numeric_after_drop:
    raise ValueError(
        "Still found non-numeric feature columns: "
        f"{non_numeric_after_drop}"
    )

print(
    "\nNumber of final numeric feature columns:",
    len(feature_cols)
)


# =========================================================
# 6) SPLIT DATA BY OPEN-SET ROLE
# =========================================================
normal_df = X_df[
    X_df[LABEL_COL] == BENIGN_LABEL
].copy()

known_df = X_df[
    X_df[LABEL_COL].isin(KNOWN_ATTACK_LABELS)
].copy()

unseen_df = X_df[
    X_df[LABEL_COL].isin(UNSEEN_ATTACK_LABELS)
].copy()

print("\nCounts by role:")
print("Normal :", len(normal_df))
print("Known  :", len(known_df))
print("Unseen :", len(unseen_df))


# =========================================================
# 7) SPLIT NORMAL -> TRAIN / VALIDATION / TEST
# =========================================================
normal_trainval, normal_test = train_test_split(
    normal_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

normal_train, normal_val = train_test_split(
    normal_trainval,
    test_size=VAL_SIZE_FROM_REMAIN,
    random_state=RANDOM_STATE,
    shuffle=True
)


# =========================================================
# 8) SPLIT KNOWN ATTACKS -> TRAIN / VALIDATION / TEST
# =========================================================
known_trainval, known_test = train_test_split(
    known_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=known_df[LABEL_COL]
)

known_train, known_val = train_test_split(
    known_trainval,
    test_size=VAL_SIZE_FROM_REMAIN,
    random_state=RANDOM_STATE,
    stratify=known_trainval[LABEL_COL]
)


# =========================================================
# 9) BUILD FINAL EXPERIMENTAL SPLITS
# =========================================================

# Self-supervised pretraining:
# Normal training samples only
pretrain_train = normal_train.copy()

# Supervised fine-tuning:
# Normal + known attack training samples
finetune_train = pd.concat(
    [normal_train, known_train],
    axis=0
).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Validation:
# Normal + known attack validation samples
finetune_val = pd.concat(
    [normal_val, known_val],
    axis=0
).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Open-set test:
# Known test samples + all unseen attacks
open_test = pd.concat(
    [normal_test, known_test, unseen_df],
    axis=0
).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# =========================================================
# 10) ADD OPEN-SET TARGET
# =========================================================

# Known classes preserve their original labels.
# Unseen attack categories are grouped as Unknown.
open_test["OpenSetLabel"] = open_test[
    "OriginalLabel"
].where(
    open_test["OriginalLabel"].isin(
        [BENIGN_LABEL] + KNOWN_ATTACK_LABELS
    ),
    "Unknown"
)

# Training and validation contain known classes only
finetune_train["OpenSetLabel"] = (
    finetune_train["OriginalLabel"]
)

finetune_val["OpenSetLabel"] = (
    finetune_val["OriginalLabel"]
)

pretrain_train["OpenSetLabel"] = (
    pretrain_train["OriginalLabel"]
)


# =========================================================
# 11) SAVE FILES
# =========================================================
pretrain_path = os.path.join(
    OUTPUT_DIR,
    "pretrain_normal_train.csv"
)

finetune_train_path = os.path.join(
    OUTPUT_DIR,
    "finetune_train.csv"
)

finetune_val_path = os.path.join(
    OUTPUT_DIR,
    "finetune_val.csv"
)

open_test_path = os.path.join(
    OUTPUT_DIR,
    "open_test.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "split_summary.json"
)

pretrain_train.to_csv(
    pretrain_path,
    index=False
)

finetune_train.to_csv(
    finetune_train_path,
    index=False
)

finetune_val.to_csv(
    finetune_val_path,
    index=False
)

open_test.to_csv(
    open_test_path,
    index=False
)


# =========================================================
# 12) SAVE SPLIT SUMMARY
# =========================================================
summary = {
    "source_file": "InSDN_Normal_and_Attack_Combined.csv",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "validation_size_from_remaining": VAL_SIZE_FROM_REMAIN,
    "duplicate_removal": REMOVE_DUPLICATES,
    "original_shape_after_cleaning": list(df.shape),
    "feature_columns_count": len(feature_cols),
    "feature_columns": feature_cols,
    "dropped_non_feature_columns": feature_drop_cols,
    "benign_label": BENIGN_LABEL,
    "known_attack_labels": KNOWN_ATTACK_LABELS,
    "unseen_attack_labels": UNSEEN_ATTACK_LABELS,

    "split_sizes": {
        "pretrain_train": len(pretrain_train),
        "finetune_train": len(finetune_train),
        "finetune_val": len(finetune_val),
        "open_test": len(open_test)
    },

    "label_distributions": {
        "pretrain_train": (
            pretrain_train["OriginalLabel"]
            .value_counts()
            .to_dict()
        ),

        "finetune_train": (
            finetune_train["OriginalLabel"]
            .value_counts()
            .to_dict()
        ),

        "finetune_val": (
            finetune_val["OriginalLabel"]
            .value_counts()
            .to_dict()
        ),

        "open_test_original": (
            open_test["OriginalLabel"]
            .value_counts()
            .to_dict()
        ),

        "open_test_openset": (
            open_test["OpenSetLabel"]
            .value_counts()
            .to_dict()
        )
    }
}

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# =========================================================
# 13) PRINT SUMMARY
# =========================================================
print("\n" + "=" * 80)
print("FINAL SPLIT SUMMARY")
print("=" * 80)

print("\nPretraining set (Normal only):")
print(
    pretrain_train["OriginalLabel"]
    .value_counts()
)

print("\nFine-tuning train:")
print(
    finetune_train["OriginalLabel"]
    .value_counts()
)

print("\nFine-tuning validation:")
print(
    finetune_val["OriginalLabel"]
    .value_counts()
)

print("\nOpen-set test (original labels):")
print(
    open_test["OriginalLabel"]
    .value_counts()
)

print("\nOpen-set test (OpenSetLabel):")
print(
    open_test["OpenSetLabel"]
    .value_counts()
)

print("\nSaved files:")
print(pretrain_path)
print(finetune_train_path)
print(finetune_val_path)
print(open_test_path)
print(summary_path)
