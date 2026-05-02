"""
PlacementIQ — Model Training
• 80% train / 20% test (stratified)
• Model ONLY sees train.csv — test.csv is held out
• Saves train.csv + test.csv for frontend use
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib, json, os

BASE = os.path.dirname(os.path.abspath(__file__))

print("=" * 55)
print("  PlacementIQ — Model Training  (80/20 split)")
print("=" * 55)

raw = pd.read_csv(os.path.join(BASE, "train.csv"))
raw = raw[[c for c in raw.columns if not c.endswith(("_enc","_bin"))]]
print(f"\nRaw dataset: {raw.shape[0]} rows")
print(raw["PlacementStatus"].value_counts().to_string())

le_extra    = LabelEncoder()
le_training = LabelEncoder()
le_status   = LabelEncoder()

raw["ExtracurricularActivities_enc"] = le_extra.fit_transform(raw["ExtracurricularActivities"])
raw["PlacementTraining_enc"]         = le_training.fit_transform(raw["PlacementTraining"])
raw["PlacementStatus_enc"]           = le_status.fit_transform(raw["PlacementStatus"])

FEATURES = [
    "CGPA","Internships","Projects","Workshops_Certifications",
    "AptitudeTestScore","SoftSkillsRating",
    "ExtracurricularActivities_enc","PlacementTraining_enc",
    "SSC_Marks","HSC_Marks",
]

X = raw[FEATURES]
y = raw["PlacementStatus_enc"]

# ─ 80/20 stratified split ─
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

# Save splits (clean — no encoded cols)
drop_cols = ["ExtracurricularActivities_enc","PlacementTraining_enc","PlacementStatus_enc"]
raw.iloc[X_train.index].drop(columns=drop_cols, errors="ignore").to_csv(os.path.join(BASE,"train.csv"), index=False)
raw.iloc[X_test.index].drop(columns=drop_cols,  errors="ignore").to_csv(os.path.join(BASE,"test.csv"),  index=False)
print("Saved train.csv and test.csv")

# ─ Scale: fit ONLY on train ─
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # no leakage

# ─ Train ─
model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    min_samples_split=20, subsample=0.85, random_state=42
)
model.fit(X_train_scaled, y_train)

# ─ Evaluate on held-out test set ─
y_pred      = model.predict(X_test_scaled)
placed_idx  = list(le_status.classes_).index("Placed")
y_pred_prob = model.predict_proba(X_test_scaled)[:, placed_idx]

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average="weighted")
roc_auc  = roc_auc_score(y_test, y_pred_prob)
cm       = confusion_matrix(y_test, y_pred)

print("\n===== TEST SET RESULTS (model never saw this) =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=le_status.classes_))

# ─ CV on train ─
cv_scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    X_train_scaled, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy"
)
print(f"5-Fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

fi = dict(zip(FEATURES, model.feature_importances_.tolist()))

stats = {
    "total_students": len(raw), "train_size": len(X_train), "test_size": len(X_test),
    "placed": int((raw["PlacementStatus"]=="Placed").sum()),
    "not_placed": int((raw["PlacementStatus"]=="NotPlaced").sum()),
    "placement_rate": float((raw["PlacementStatus"]=="Placed").mean()),
    "avg_cgpa": float(raw["CGPA"].mean()),
    "avg_aptitude": float(raw["AptitudeTestScore"].mean()),
    "avg_soft_skills": float(raw["SoftSkillsRating"].mean()),
    "model_accuracy": float(accuracy),
    "model_f1": float(f1),
    "model_roc_auc": float(roc_auc),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "cv_scores": cv_scores.tolist(),
    "confusion_matrix": cm.tolist(),
    "feature_importance": fi,
    "features": FEATURES,
    "classes": list(le_status.classes_),
}

joblib.dump(model,       os.path.join(BASE,"placement_model.pkl"))
joblib.dump(scaler,      os.path.join(BASE,"scaler.pkl"))
joblib.dump(le_extra,    os.path.join(BASE,"le_extra.pkl"))
joblib.dump(le_training, os.path.join(BASE,"le_training.pkl"))
joblib.dump(le_status,   os.path.join(BASE,"le_status.pkl"))

with open(os.path.join(BASE,"model_stats.json"),"w") as f:
    json.dump(stats, f, indent=2)

print("\nAll artifacts saved. Done.")
