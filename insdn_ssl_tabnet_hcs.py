import os
import json
import random
import joblib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef
)


def patch_only_numpy_unsup_metric(eps: float = 1e-8) -> None:
    import pytorch_tabnet.metrics as ptm

    def safe_unsupervised_loss_numpy(y_pred, embedded_x, obf_vars, eps_local=eps):
        y_pred = np.asarray(y_pred, dtype=np.float64)
        embedded_x = np.asarray(embedded_x, dtype=np.float64)
        obf_vars = np.asarray(obf_vars, dtype=np.float64)

        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        embedded_x = np.nan_to_num(embedded_x, nan=0.0, posinf=1e6, neginf=-1e6)
        obf_vars = np.nan_to_num(obf_vars, nan=0.0, posinf=1.0, neginf=0.0)

        errors = np.clip(y_pred - embedded_x, -1e6, 1e6)
        reconstruction_errors = np.multiply(errors, obf_vars) ** 2
        reconstruction_errors = np.nan_to_num(
            reconstruction_errors,
            nan=0.0,
            posinf=1e6,
            neginf=0.0
        )
        reconstruction_errors = np.clip(reconstruction_errors, 0.0, 1e6)

        batch_means = np.mean(embedded_x, axis=0, dtype=np.float64)
        batch_means = np.nan_to_num(batch_means, nan=1.0, posinf=1.0, neginf=1.0)

        batch_vars = np.var(embedded_x, axis=0, ddof=0, dtype=np.float64)
        batch_vars = np.nan_to_num(batch_vars, nan=0.0, posinf=0.0, neginf=0.0)

        fallback = np.where(np.abs(batch_means) > eps_local, np.abs(batch_means), 1.0)
        denom = np.where(batch_vars > eps_local, batch_vars, fallback)
        denom = np.nan_to_num(
            denom,
            nan=eps_local,
            posinf=1.0 / eps_local,
            neginf=eps_local
        )
        denom = np.maximum(denom, eps_local)

        inv_denom = np.empty_like(denom, dtype=np.float64)
        np.divide(1.0, denom, out=inv_denom, where=denom > 0)
        inv_denom = np.nan_to_num(inv_denom, nan=0.0, posinf=1e6, neginf=0.0)
        inv_denom = np.clip(inv_denom, 0.0, 1e6)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            features_loss = reconstruction_errors @ inv_denom

        features_loss = np.nan_to_num(features_loss, nan=0.0, posinf=1e6, neginf=0.0)

        nb_reconstructed_variables = np.sum(obf_vars, axis=1, dtype=np.float64)
        nb_reconstructed_variables = np.clip(nb_reconstructed_variables, 1.0, None)

        features_loss = features_loss / nb_reconstructed_variables
        features_loss = np.nan_to_num(features_loss, nan=0.0, posinf=1e6, neginf=0.0)

        loss = float(np.mean(features_loss))
        loss = float(np.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0))
        return loss

    ptm.UnsupervisedLossNumpy = safe_unsupervised_loss_numpy

    def _safe_unsup_numpy_metric_call(self, y_pred, embedded_x, obf_vars):
        return float(safe_unsupervised_loss_numpy(y_pred, embedded_x, obf_vars))

    ptm.UnsupervisedNumpyMetric.__call__ = _safe_unsup_numpy_metric_call


patch_only_numpy_unsup_metric()

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_ROOT, "data", "InSDN")

PRETRAIN_PATH = os.path.join(BASE_DIR, "pretrain_normal_train.csv")
FINETUNE_TRAIN_PATH = os.path.join(BASE_DIR, "finetune_train.csv")
FINETUNE_VAL_PATH = os.path.join(BASE_DIR, "finetune_val.csv")
OPEN_TEST_PATH = os.path.join(BASE_DIR, "open_test.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "InSDN")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

LABEL_COL = "OriginalLabel"
OPENSET_COL = "OpenSetLabel"
REJECTED_CLASS_LABEL = "New Attack"

DROP_COLS = [
    "Label",
    "OriginalLabel",
    "OpenSetLabel"
]

CLIP_LOW_Q = 0.01
CLIP_HIGH_Q = 0.99

PRETRAINING_RATIO = 0.3

N_D = 24
N_A = 24
N_STEPS = 4
GAMMA = 1.3
LR = 0.002

DETECTOR_TYPE = "validation_calibrated_hybrid_class_specific_rejection"

KNOWN_REJECTION_RATE = 0.02

W_SIM = 0.45
W_CONF = 0.45
W_DISAGREE = 0.10

CLASS_REJECT_RATES = {
    "DDoS": 0.005,
    "Normal": 0.010,
    "Probe": 0.010
}

PRETRAIN_MAX_EPOCHS = 35
FINETUNE_MAX_EPOCHS = 70

PATIENCE_PRETRAIN = 8
PATIENCE_FINETUNE = 12

BATCH_SIZE = 1024
VIRTUAL_BATCH_SIZE = 128

OPEN_LABELS_ORDER = ["Normal", "DDoS", "Probe", REJECTED_CLASS_LABEL]


def pct(x: float) -> float:
    return float(x * 100.0)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def sanitize_array(x: np.ndarray, clip_value: float = 1e6) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = sanitize_array(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    x = x / norms
    return sanitize_array(x)


def robust_numeric_frame(
    df: pd.DataFrame,
    cols: List[str],
    fill_values: pd.Series = None
) -> pd.DataFrame:
    out = df[cols].copy()

    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)

    if fill_values is None:
        fill_values = out.median(numeric_only=True).fillna(0.0)

    out = out.fillna(fill_values)
    return out.astype(np.float32)


def assert_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    df_name: str
) -> None:
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def assert_same_feature_layout(
    reference_cols: List[str],
    df: pd.DataFrame,
    df_name: str
) -> None:
    missing = [c for c in reference_cols if c not in df.columns]

    if missing:
        raise ValueError(f"{df_name} is missing feature columns: {missing}")


def find_bad_feature_columns(
    X_ref: np.ndarray,
    feature_cols: List[str],
    std_eps: float = 1e-12
) -> List[str]:
    stds = np.nanstd(X_ref, axis=0)
    bad = []

    for c, s in zip(feature_cols, stds):
        if (not np.isfinite(s)) or (s <= std_eps):
            bad.append(c)

    return bad


def fit_feature_clip_bounds(
    X_ref: np.ndarray,
    low_q: float,
    high_q: float
) -> Tuple[np.ndarray, np.ndarray]:
    low = np.quantile(X_ref, low_q, axis=0)
    high = np.quantile(X_ref, high_q, axis=0)

    same_mask = high <= low
    high[same_mask] = low[same_mask] + 1e-6

    return low.astype(np.float32), high.astype(np.float32)


def apply_feature_clipping(
    X: np.ndarray,
    low: np.ndarray,
    high: np.ndarray
) -> np.ndarray:
    return np.clip(X, low, high).astype(np.float32)


def extract_tabnet_latent(
    model: TabNetClassifier,
    X: np.ndarray,
    batch_size: int = 4096
) -> np.ndarray:
    model.network.eval()
    latents = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch_np = X[start:end]

            batch_tensor = torch.tensor(
                batch_np,
                dtype=torch.float32,
                device=model.device
            )

            net = model.network

            if hasattr(net, "encoder"):
                if hasattr(net, "embedder"):
                    x_for_encoder = net.embedder(batch_tensor)
                else:
                    x_for_encoder = batch_tensor

                steps_out, _ = net.encoder(x_for_encoder)

            elif hasattr(net, "tabnet") and hasattr(net.tabnet, "encoder"):
                core = net.tabnet

                if hasattr(core, "embedder"):
                    x_for_encoder = core.embedder(batch_tensor)
                else:
                    x_for_encoder = batch_tensor

                steps_out, _ = core.encoder(x_for_encoder)

            else:
                raise RuntimeError("Could not locate TabNet encoder inside the trained model.")

            latent_tensor = torch.sum(torch.stack(steps_out, dim=0), dim=0)
            latents.append(latent_tensor.cpu().numpy())

    return sanitize_array(np.vstack(latents))


def build_class_prototypes(
    embeddings_norm: np.ndarray,
    labels_text: np.ndarray,
    class_names: List[str]
) -> Dict[str, np.ndarray]:
    prototypes = {}

    for cls in class_names:
        cls_emb = embeddings_norm[labels_text == cls]

        if len(cls_emb) == 0:
            raise ValueError(f"No samples found to build prototype for class: {cls}")

        proto = cls_emb.mean(axis=0)
        proto = proto / max(np.linalg.norm(proto), 1e-12)
        prototypes[cls] = sanitize_array(proto.reshape(1, -1))[0]

    return prototypes


def prototype_similarity_matrix(
    embeddings_norm: np.ndarray,
    prototypes: Dict[str, np.ndarray],
    class_names: List[str]
) -> np.ndarray:
    proto_matrix = np.vstack([prototypes[c] for c in class_names]).astype(np.float32)

    with np.errstate(all="ignore"):
        sims = embeddings_norm @ proto_matrix.T

    sims = np.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
    sims = np.clip(sims, -1.0, 1.0)
    return sims.astype(np.float32)


def choose_similarity_thresholds(
    val_embeddings_norm: np.ndarray,
    val_labels_text: np.ndarray,
    prototypes: Dict[str, np.ndarray],
    class_names: List[str],
    reject_rate: float
) -> Dict[str, float]:
    thresholds = {}

    for cls in class_names:
        cls_mask = val_labels_text == cls
        cls_emb = val_embeddings_norm[cls_mask]

        if len(cls_emb) == 0:
            raise ValueError(f"No validation samples found for class: {cls}")

        proto = prototypes[cls].reshape(-1, 1)

        with np.errstate(all="ignore"):
            sims = (cls_emb @ proto).reshape(-1)

        sims = np.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        sims = np.clip(sims, -1.0, 1.0)

        thresholds[cls] = float(np.quantile(sims, reject_rate))

    return thresholds


def choose_confidence_thresholds(
    val_proba: np.ndarray,
    y_val_text: np.ndarray,
    label_encoder: LabelEncoder,
    class_names: List[str],
    reject_rate: float
) -> Dict[str, float]:
    thresholds = {}

    for cls in class_names:
        cls_idx = int(label_encoder.transform([cls])[0])
        cls_mask = y_val_text == cls
        cls_conf = val_proba[cls_mask, cls_idx]

        if len(cls_conf) == 0:
            raise ValueError(f"No validation confidence samples found for class: {cls}")

        thresholds[cls] = float(np.quantile(cls_conf, reject_rate))

    return thresholds


def safe_ratio_deficit(
    value: float,
    threshold: float,
    upper: float = 1.0,
    eps: float = 1e-12
) -> float:
    denom = max(upper - threshold, eps)
    deficit = max(0.0, threshold - value) / denom
    return float(np.clip(deficit, 0.0, 5.0))


def compute_hybrid_scores(
    sims: np.ndarray,
    proba: np.ndarray,
    sim_thresholds: Dict[str, float],
    conf_thresholds: Dict[str, float],
    class_names: List[str],
    w_sim: float,
    w_conf: float,
    w_disagree: float
):
    proto_idx = np.argmax(sims, axis=1)
    clf_idx = np.argmax(proba, axis=1)

    proto_cls = np.array([class_names[i] for i in proto_idx]).astype(str)
    clf_cls = np.array([class_names[i] for i in clf_idx]).astype(str)

    max_sim = sims[np.arange(len(sims)), proto_idx]
    proto_conf = proba[np.arange(len(proba)), proto_idx]

    hybrid_scores = []

    for i, cls in enumerate(proto_cls):
        sim_deficit = safe_ratio_deficit(
            value=float(max_sim[i]),
            threshold=float(sim_thresholds[cls]),
            upper=1.0
        )

        conf_deficit = safe_ratio_deficit(
            value=float(proto_conf[i]),
            threshold=float(conf_thresholds[cls]),
            upper=1.0
        )

        disagree = 1.0 if proto_cls[i] != clf_cls[i] else 0.0

        score = (
            w_sim * sim_deficit
            + w_conf * conf_deficit
            + w_disagree * disagree
        )

        hybrid_scores.append(score)

    return (
        proto_cls,
        clf_cls,
        max_sim.astype(np.float32),
        proto_conf.astype(np.float32),
        np.asarray(hybrid_scores, dtype=np.float32)
    )


pretrain_df = pd.read_csv(PRETRAIN_PATH)
finetune_train_df = pd.read_csv(FINETUNE_TRAIN_PATH)
finetune_val_df = pd.read_csv(FINETUNE_VAL_PATH)
open_test_df = pd.read_csv(OPEN_TEST_PATH)

print("=" * 80)
print("LOADED INSDN DATA")
print("=" * 80)
print("Pretrain shape      :", pretrain_df.shape)
print("Finetune train shape:", finetune_train_df.shape)
print("Finetune val shape  :", finetune_val_df.shape)
print("Open test shape     :", open_test_df.shape)

for df_name, df in [
    ("pretrain_df", pretrain_df),
    ("finetune_train_df", finetune_train_df),
    ("finetune_val_df", finetune_val_df),
    ("open_test_df", open_test_df),
]:
    required = [OPENSET_COL] if df_name == "open_test_df" else [LABEL_COL]
    assert_required_columns(df, required, df_name)

raw_feature_cols = [c for c in pretrain_df.columns if c not in DROP_COLS]

assert_same_feature_layout(raw_feature_cols, finetune_train_df, "finetune_train_df")
assert_same_feature_layout(raw_feature_cols, finetune_val_df, "finetune_val_df")
assert_same_feature_layout(raw_feature_cols, open_test_df, "open_test_df")

pretrain_features_df = robust_numeric_frame(
    pretrain_df,
    raw_feature_cols,
    fill_values=None
)

shared_fill_values = pretrain_features_df.median(numeric_only=True).fillna(0.0)

train_features_df = robust_numeric_frame(
    finetune_train_df,
    raw_feature_cols,
    fill_values=shared_fill_values
)

val_features_df = robust_numeric_frame(
    finetune_val_df,
    raw_feature_cols,
    fill_values=shared_fill_values
)

test_features_df = robust_numeric_frame(
    open_test_df,
    raw_feature_cols,
    fill_values=shared_fill_values
)

bad_cols = find_bad_feature_columns(
    pretrain_features_df.values,
    raw_feature_cols,
    std_eps=1e-12
)

feature_cols = [c for c in raw_feature_cols if c not in bad_cols]

pretrain_features_df = pretrain_features_df[feature_cols]
train_features_df = train_features_df[feature_cols]
val_features_df = val_features_df[feature_cols]
test_features_df = test_features_df[feature_cols]

X_pretrain = pretrain_features_df.values.astype(np.float32)

val_normal_mask = finetune_val_df[LABEL_COL].astype(str).values == "Normal"

if np.sum(val_normal_mask) < 2:
    X_pretrain_eval = val_features_df.values.astype(np.float32)
else:
    X_pretrain_eval = val_features_df.loc[val_normal_mask].values.astype(np.float32)

X_train = train_features_df.values.astype(np.float32)
X_val = val_features_df.values.astype(np.float32)
X_test = test_features_df.values.astype(np.float32)

y_train_text = finetune_train_df[LABEL_COL].astype(str).values
y_val_text = finetune_val_df[LABEL_COL].astype(str).values
y_test_open_text_raw = open_test_df[OPENSET_COL].astype(str).values

clip_low, clip_high = fit_feature_clip_bounds(
    X_pretrain,
    CLIP_LOW_Q,
    CLIP_HIGH_Q
)

X_pretrain_clip = apply_feature_clipping(X_pretrain, clip_low, clip_high)
X_pretrain_eval_clip = apply_feature_clipping(X_pretrain_eval, clip_low, clip_high)
X_train_clip = apply_feature_clipping(X_train, clip_low, clip_high)
X_val_clip = apply_feature_clipping(X_val, clip_low, clip_high)
X_test_clip = apply_feature_clipping(X_test, clip_low, clip_high)

scaler = StandardScaler()

X_pretrain_scaled = sanitize_array(scaler.fit_transform(X_pretrain_clip))
X_pretrain_eval_scaled = sanitize_array(scaler.transform(X_pretrain_eval_clip))
X_train_scaled = sanitize_array(scaler.transform(X_train_clip))
X_val_scaled = sanitize_array(scaler.transform(X_val_clip))
X_test_scaled = sanitize_array(scaler.transform(X_test_clip))

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "standard_scaler.pkl"))
joblib.dump({"low": clip_low, "high": clip_high}, os.path.join(OUTPUT_DIR, "clip_bounds.pkl"))

print("\nPreprocessing completed.")

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)
y_val = label_encoder.transform(y_val_text)

known_classes = label_encoder.classes_.tolist()
y_test_open_text = np.where(
    np.isin(y_test_open_text_raw, known_classes),
    y_test_open_text_raw,
    REJECTED_CLASS_LABEL
)

joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

print("\nKnown supervised classes:", known_classes)

print("\n" + "=" * 80)
print("START SELF-SUPERVISED PRETRAINING")
print("=" * 80)

pretrainer = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=LR),
    mask_type="entmax",
    n_d=N_D,
    n_a=N_A,
    n_steps=N_STEPS,
    gamma=GAMMA,
    seed=RANDOM_STATE,
    verbose=10
)

pretrainer.fit(
    X_train=X_pretrain_scaled,
    eval_set=[X_pretrain_eval_scaled],
    max_epochs=PRETRAIN_MAX_EPOCHS,
    patience=PATIENCE_PRETRAIN,
    batch_size=BATCH_SIZE,
    virtual_batch_size=VIRTUAL_BATCH_SIZE,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=PRETRAINING_RATIO
)

print("\n" + "=" * 80)
print("START SUPERVISED FINE-TUNING")
print("=" * 80)

clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=LR),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={
        "step_size": 20,
        "gamma": 0.9
    },
    mask_type="entmax",
    n_d=N_D,
    n_a=N_A,
    n_steps=N_STEPS,
    gamma=GAMMA,
    seed=RANDOM_STATE,
    verbose=10
)

clf.fit(
    X_train=X_train_scaled,
    y_train=y_train,
    eval_set=[(X_val_scaled, y_val)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=FINETUNE_MAX_EPOCHS,
    patience=PATIENCE_FINETUNE,
    batch_size=BATCH_SIZE,
    virtual_batch_size=VIRTUAL_BATCH_SIZE,
    num_workers=0,
    drop_last=False,
    from_unsupervised=pretrainer
)

model_path = os.path.join(OUTPUT_DIR, "tabnet_insdn_model")
clf.save_model(model_path)

val_pred_clf_idx = clf.predict(X_val_scaled)
val_pred_clf_text = label_encoder.inverse_transform(val_pred_clf_idx)
val_proba = clf.predict_proba(X_val_scaled)

val_ca = accuracy_score(y_val_text, val_pred_clf_text)

val_f1_macro = f1_score(
    y_val_text,
    val_pred_clf_text,
    average="macro",
    zero_division=0
)

val_rec_macro = recall_score(
    y_val_text,
    val_pred_clf_text,
    average="macro",
    zero_division=0
)

val_prec_macro = precision_score(
    y_val_text,
    val_pred_clf_text,
    average="macro",
    zero_division=0
)

val_mcc = matthews_corrcoef(y_val_text, val_pred_clf_text)

print("\n" + "=" * 80)
print("CLOSED-SET VALIDATION METRICS (%)")
print("=" * 80)
print(f"CA     : {pct(val_ca):.4f}%")
print(f"F1     : {pct(val_f1_macro):.4f}%")
print(f"Recall : {pct(val_rec_macro):.4f}%")
print(f"Prec   : {pct(val_prec_macro):.4f}%")
print(f"MCC    : {pct(val_mcc):.4f}%")

print("\n" + "=" * 80)
print("EXTRACTING LATENT REPRESENTATIONS")
print("=" * 80)

train_latent = extract_tabnet_latent(
    clf,
    X_train_scaled,
    batch_size=4096
)

val_latent = extract_tabnet_latent(
    clf,
    X_val_scaled,
    batch_size=4096
)

test_latent = extract_tabnet_latent(
    clf,
    X_test_scaled,
    batch_size=4096
)

print("Train latent shape:", train_latent.shape)
print("Val latent shape  :", val_latent.shape)
print("Test latent shape :", test_latent.shape)

train_latent_norm = l2_normalize(train_latent)
val_latent_norm = l2_normalize(val_latent)
test_latent_norm = l2_normalize(test_latent)

prototypes = build_class_prototypes(
    train_latent_norm,
    y_train_text,
    known_classes
)

sim_thresholds = choose_similarity_thresholds(
    val_embeddings_norm=val_latent_norm,
    val_labels_text=y_val_text,
    prototypes=prototypes,
    class_names=known_classes,
    reject_rate=KNOWN_REJECTION_RATE
)

conf_thresholds = choose_confidence_thresholds(
    val_proba=val_proba,
    y_val_text=y_val_text,
    label_encoder=label_encoder,
    class_names=known_classes,
    reject_rate=KNOWN_REJECTION_RATE
)

save_json(
    {k: v.tolist() for k, v in prototypes.items()},
    os.path.join(OUTPUT_DIR, "class_prototypes.json")
)

save_json(
    sim_thresholds,
    os.path.join(OUTPUT_DIR, "similarity_thresholds.json")
)

save_json(
    conf_thresholds,
    os.path.join(OUTPUT_DIR, "confidence_thresholds.json")
)

print("\nSimilarity thresholds:")
for cls in known_classes:
    print(f"{cls}: {sim_thresholds[cls]:.6f}")

print("\nConfidence thresholds:")
for cls in known_classes:
    print(f"{cls}: {conf_thresholds[cls]:.6f}")

val_sims = prototype_similarity_matrix(
    val_latent_norm,
    prototypes,
    known_classes
)

val_proto_cls, val_clf_cls, val_max_sim, val_proto_conf, val_hybrid_scores = compute_hybrid_scores(
    sims=val_sims,
    proba=val_proba,
    sim_thresholds=sim_thresholds,
    conf_thresholds=conf_thresholds,
    class_names=known_classes,
    w_sim=W_SIM,
    w_conf=W_CONF,
    w_disagree=W_DISAGREE
)

hybrid_thresholds = {}

for cls in known_classes:
    cls_scores = val_hybrid_scores[val_proto_cls == cls]

    if len(cls_scores) == 0:
        raise ValueError(f"No hybrid scores found for predicted prototype class: {cls}")

    cls_rr = CLASS_REJECT_RATES.get(cls, KNOWN_REJECTION_RATE)
    hybrid_thresholds[cls] = float(np.quantile(cls_scores, 1.0 - cls_rr))

save_json(
    hybrid_thresholds,
    os.path.join(OUTPUT_DIR, "hybrid_class_thresholds.json")
)

print("\nHybrid class-specific thresholds:")
for cls in known_classes:
    print(f"{cls}: {hybrid_thresholds[cls]:.6f}")

val_pred_open = np.array([
    REJECTED_CLASS_LABEL if val_hybrid_scores[i] > hybrid_thresholds[val_proto_cls[i]]
    else val_proto_cls[i]
    for i in range(len(val_proto_cls))
])

val_reject_ratio = np.mean(val_pred_open == REJECTED_CLASS_LABEL)
val_after_rejection_acc = accuracy_score(y_val_text, val_pred_open)

print(f"\nValidation rejection ratio: {pct(val_reject_ratio):.4f}%")
print(f"Validation accuracy after rejection: {pct(val_after_rejection_acc):.4f}%")

test_sims = prototype_similarity_matrix(
    test_latent_norm,
    prototypes,
    known_classes
)

test_proba = clf.predict_proba(X_test_scaled)

test_proto_cls, test_clf_cls, test_max_sim, test_proto_conf, test_hybrid_scores = compute_hybrid_scores(
    sims=test_sims,
    proba=test_proba,
    sim_thresholds=sim_thresholds,
    conf_thresholds=conf_thresholds,
    class_names=known_classes,
    w_sim=W_SIM,
    w_conf=W_CONF,
    w_disagree=W_DISAGREE
)

test_pred_open = np.array([
    REJECTED_CLASS_LABEL if test_hybrid_scores[i] > hybrid_thresholds[test_proto_cls[i]]
    else test_proto_cls[i]
    for i in range(len(test_proto_cls))
])

open_ca = accuracy_score(y_test_open_text, test_pred_open)

open_f1_macro = f1_score(
    y_test_open_text,
    test_pred_open,
    labels=OPEN_LABELS_ORDER,
    average="macro",
    zero_division=0
)

open_rec_macro = recall_score(
    y_test_open_text,
    test_pred_open,
    labels=OPEN_LABELS_ORDER,
    average="macro",
    zero_division=0
)

open_prec_macro = precision_score(
    y_test_open_text,
    test_pred_open,
    labels=OPEN_LABELS_ORDER,
    average="macro",
    zero_division=0
)

open_mcc = matthews_corrcoef(y_test_open_text, test_pred_open)

print("\n" + "=" * 80)
print("OPEN-SET TEST METRICS (%)")
print("=" * 80)
print(f"CA     : {pct(open_ca):.4f}%")
print(f"F1     : {pct(open_f1_macro):.4f}%")
print(f"Recall : {pct(open_rec_macro):.4f}%")
print(f"Prec   : {pct(open_prec_macro):.4f}%")
print(f"MCC    : {pct(open_mcc):.4f}%")

report_dict = classification_report(
    y_test_open_text,
    test_pred_open,
    labels=OPEN_LABELS_ORDER,
    zero_division=0,
    output_dict=True
)

report_df_full = pd.DataFrame(report_dict).T

report_df = report_df_full.loc[
    OPEN_LABELS_ORDER,
    ["precision", "recall", "f1-score", "support"]
].copy()

for col in ["precision", "recall", "f1-score"]:
    report_df[col] = report_df[col] * 100.0

print("\n" + "=" * 80)
print("PER-CLASS REPORT (%)")
print("=" * 80)
print(report_df.round(4))

cm = confusion_matrix(
    y_test_open_text,
    test_pred_open,
    labels=OPEN_LABELS_ORDER
)

cm_df = pd.DataFrame(
    cm,
    index=[f"True_{c}" for c in OPEN_LABELS_ORDER],
    columns=[f"Pred_{c}" for c in OPEN_LABELS_ORDER]
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
print(cm_df)

predictions_df = pd.DataFrame({
    "true_open_label": y_test_open_text,
    "pred_open_label": test_pred_open,
    "prototype_class": test_proto_cls,
    "classifier_class": test_clf_cls,
    "max_similarity": test_max_sim,
    "prototype_confidence": test_proto_conf,
    "hybrid_score": test_hybrid_scores
})

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "open_set_predictions.csv"),
    index=False
)

report_df.to_csv(
    os.path.join(OUTPUT_DIR, "per_class_report_percent.csv"),
    index=True
)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
    index=True
)

metrics_summary = {
    "dataset": "InSDN",
    "detector_type": DETECTOR_TYPE,
    "known_classes": known_classes,
    "open_labels_order": OPEN_LABELS_ORDER,
    "weights": {
        "W_SIM": W_SIM,
        "W_CONF": W_CONF,
        "W_DISAGREE": W_DISAGREE
    },
    "known_rejection_rate": KNOWN_REJECTION_RATE,
    "class_reject_rates": CLASS_REJECT_RATES,
    "closed_set_validation_metrics_percent": {
        "CA": pct(val_ca),
        "F1": pct(val_f1_macro),
        "Recall": pct(val_rec_macro),
        "Precision": pct(val_prec_macro),
        "MCC": pct(val_mcc)
    },
    "validation_calibration": {
        "validation_rejection_ratio_percent": pct(val_reject_ratio),
        "validation_accuracy_after_rejection_percent": pct(val_after_rejection_acc),
        "similarity_thresholds": sim_thresholds,
        "confidence_thresholds": conf_thresholds,
        "hybrid_thresholds": hybrid_thresholds
    },
    "open_set_test_metrics_percent": {
        "CA": pct(open_ca),
        "F1": pct(open_f1_macro),
        "Recall": pct(open_rec_macro),
        "Precision": pct(open_prec_macro),
        "MCC": pct(open_mcc)
    }
}

save_json(
    metrics_summary,
    os.path.join(OUTPUT_DIR, "metrics_summary.json")
)

with open(os.path.join(OUTPUT_DIR, "final_report_percent.txt"), "w", encoding="utf-8") as f:
    f.write("CLOSED-SET VALIDATION METRICS (%)\n")
    f.write("=" * 80 + "\n")
    f.write(f"CA     : {pct(val_ca):.4f}%\n")
    f.write(f"F1     : {pct(val_f1_macro):.4f}%\n")
    f.write(f"Recall : {pct(val_rec_macro):.4f}%\n")
    f.write(f"Prec   : {pct(val_prec_macro):.4f}%\n")
    f.write(f"MCC    : {pct(val_mcc):.4f}%\n\n")

    f.write("OPEN-SET TEST METRICS (%)\n")
    f.write("=" * 80 + "\n")
    f.write(f"CA     : {pct(open_ca):.4f}%\n")
    f.write(f"F1     : {pct(open_f1_macro):.4f}%\n")
    f.write(f"Recall : {pct(open_rec_macro):.4f}%\n")
    f.write(f"Prec   : {pct(open_prec_macro):.4f}%\n")
    f.write(f"MCC    : {pct(open_mcc):.4f}%\n\n")

    f.write("PER-CLASS REPORT (%)\n")
    f.write("=" * 80 + "\n")
    f.write(report_df.round(4).to_string())
    f.write("\n\n")

    f.write("CONFUSION MATRIX\n")
    f.write("=" * 80 + "\n")
    f.write(cm_df.to_string())
    f.write("\n")

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("Output directory:", OUTPUT_DIR)
