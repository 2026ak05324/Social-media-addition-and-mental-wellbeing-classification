from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

TARGET = "Addiction_Level"
CLASS_ORDER = ["Low", "Moderate", "High", "Severe"]
FEATURES = ["Age", "Gender", "Occupation", "Relationship_Status", "Primary_Platform", "Daily_Usage_Hours", "Platforms_Used_Count", "Posts_Per_Week", "Late_Night_Usage", "First_Check_Morning", "Notifications_Per_Day", "FOMO_Score", "Social_Comparison_Score", "Validation_Seeking_Score", "Scroll_Without_Purpose", "Tried_To_Cut_Back", "Failed_To_Cut_Back", "Anxiety_Score", "Depression_Score", "Loneliness_Score", "Self_Esteem_Score", "Sleep_Quality_Score", "Sleep_Hours", "Productivity_Loss_Score", "Offline_Relationship_Quality", "Physical_Activity_Hrs_Week", "Screen_Free_Time_Hrs", "Mental_Wellbeing_Score"]
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}
BASE = Path(__file__).parent


@st.cache_resource
def load_model(key):
    return joblib.load(BASE / "model" / f"{key}.joblib")


def compute_metrics(model, X, y):
    """Macro-averaged precision/recall/F1, one-vs-rest macro AUC, multiclass MCC."""
    y_pred = model.predict(X)
    try:
        auc = roc_auc_score(y, model.predict_proba(X), multi_class="ovr", average="macro", labels=model.classes_)
    except ValueError:   # uploaded data does not contain every class
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


st.set_page_config(page_title="Social Media Addiction Classifier", layout="wide")
st.title("Social Media Addiction Level Classifier")
st.caption("Predicts a user's **Addiction_Level** (Low / Moderate / High / Severe) from usage habits and "
           "mental-wellbeing scores. Pre-trained models are loaded from disk - nothing is retrained on uploaded data.")

st.sidebar.header("Controls")
choice = st.sidebar.selectbox("Select model", list(MODELS))
up = st.sidebar.file_uploader("Upload test CSV (features + Addiction_Level)", type="csv")

if up is not None:
    df = pd.read_csv(up)
elif (BASE / "test_data.csv").exists():
    df = pd.read_csv(BASE / "test_data.csv")
    st.info("No file uploaded - using the bundled test_data.csv.")
else:
    st.warning("Upload a test CSV to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

df = df.dropna(subset=[TARGET])
X, y = df[FEATURES], df[TARGET]
st.subheader("Data preview")
st.write(f"{len(df):,} rows - class counts: {y.value_counts().to_dict()}")
st.dataframe(df.head(10), width="stretch")

model = load_model(MODELS[choice])
metrics, y_pred = compute_metrics(model, X, y)

st.subheader(f"Evaluation metrics - {choice}")
st.caption("Precision, Recall and F1 are macro-averaged over the 4 classes; AUC is one-vs-rest macro.")
cols = st.columns(6)
for c, (k, v) in zip(cols, metrics.items()):
    c.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, y_pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

st.subheader("All models on this data")
rows = []
for name, key in MODELS.items():
    m, _ = compute_metrics(load_model(key), X, y)
    rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})
st.dataframe(pd.DataFrame(rows).set_index("ML Model Name"), width="stretch")
