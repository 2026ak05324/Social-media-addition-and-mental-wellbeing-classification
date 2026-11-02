"""Train 5 classifiers on the social-media addiction dataset and save them.

Usage:  python train.py social_media_addiction_mental_wellbeing.csv
Target: Addiction_Level (Low / Moderate / High / Severe) - 4-class classification.
Outputs: model/*.joblib, model/meta.json, test_data.csv (held-out 20% split)
"""
import sys, json, warnings, joblib
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
TARGET = "Addiction_Level"
CLASS_ORDER = ["Low", "Moderate", "High", "Severe"]
CATEGORICAL = ["Gender", "Occupation", "Relationship_Status", "Primary_Platform",
               "Late_Night_Usage", "First_Check_Morning", "Tried_To_Cut_Back", "Failed_To_Cut_Back"]


def compute_metrics(model, X, y):
    """Macro-averaged precision/recall/F1, one-vs-rest macro AUC, multiclass MCC."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=model.classes_),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


df = pd.read_csv(sys.argv[1])
df = df.dropna(subset=[TARGET]).drop(columns=["User_ID"])
FEATURES = [c for c in df.columns if c != TARGET]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
X, y = df[FEATURES], df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def pipe(clf):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


models = {
    "Logistic Regression": pipe(LogisticRegression(max_iter=3000, random_state=SEED)),
    "Decision Tree": pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=SEED)),
    "KNN": pipe(KNeighborsClassifier(n_neighbors=15)),
    "Naive Bayes": pipe(GaussianNB()),
    "Support Vector Machine": pipe(SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
}

Path("model").mkdir(exist_ok=True)
rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    joblib.dump(m, f"model/{name.lower().replace(' ', '_')}.joblib")
    met, _ = compute_metrics(m, X_test, y_test)
    rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in met.items()}})

out = X_test.copy(); out[TARGET] = y_test.values
out.to_csv("test_data.csv", index=False)
json.dump({"features": FEATURES, "classes": CLASS_ORDER, "n_train": len(X_train), "n_test": len(X_test)},
          open("model/meta.json", "w"), indent=2)
print(pd.DataFrame(rows).to_string(index=False))
print(f"train={len(X_train)}, test={len(X_test)}, class counts (test)={y_test.value_counts().to_dict()}")
