"""
Titanic Survival Predictor
==========================
Backend  : scikit-learn (RandomForestClassifier)
Frontend : Gradio (Python-native UI)
Dataset  : Titanic-Dataset.csv
           https://www.kaggle.com/datasets/arjunsinghgangwar/titanic-survival-prediction-dataset/data
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.  DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

DATASET_PATH = "Titanic-Dataset.csv"

def load_and_preprocess(path: str = DATASET_PATH) -> tuple:
    """Load the Titanic CSV, clean it, and return feature/label arrays."""
    df = pd.read_csv(path)

    # ── Feature engineering ──────────────────────────────────────────────
    # Title extraction from Name
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = df["Title"].value_counts()
    rare_titles = rare_titles[rare_titles < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].map(
        {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    ).fillna(4).astype(int)

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age — median by Pclass × Title
    df["Age"] = df.groupby(["Pclass", "Title"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Fare — fill with median per class
    df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"), inplace=True)

    # Embarked — fill with mode
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Encode categoricals
    le = LabelEncoder()
    df["Sex_enc"]      = le.fit_transform(df["Sex"])          # male=1, female=0
    df["Embarked_enc"] = le.fit_transform(df["Embarked"])

    # Age / Fare bands for richer signal
    df["AgeBand"]  = pd.cut(df["Age"],  bins=5, labels=False)
    df["FareBand"] = pd.qcut(df["Fare"], q=4, labels=False, duplicates="drop")

    FEATURES = [
        "Pclass", "Sex_enc", "Age", "SibSp", "Parch",
        "Fare", "Embarked_enc", "Title",
        "FamilySize", "IsAlone", "AgeBand", "FareBand",
    ]
    X = df[FEATURES].values
    y = df["Survived"].values
    return X, y, df, FEATURES


X, y, df_full, FEATURE_NAMES = load_and_preprocess()

# ---------------------------------------------------------------------------
# 2.  MODEL TRAINING
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
report   = classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"])

print(f"\n✅  Model trained  |  Test Accuracy: {accuracy:.4f}  |  CV Mean: {cv_score:.4f}\n")
print(report)

# ---------------------------------------------------------------------------
# 3.  CHART HELPERS  (return PIL images via matplotlib)
# ---------------------------------------------------------------------------

def _fig_to_path(fig, name: str) -> str:
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "charts"
    )
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"{name}.png")

    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    return path


def chart_survival_overview() -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Titanic — Survival Overview", fontsize=14, fontweight="bold")

    # Survival count
    counts = df_full["Survived"].value_counts()
    axes[0].bar(["Not Survived", "Survived"], counts.values,
                color=["#e74c3c", "#2ecc71"], edgecolor="white")
    axes[0].set_title("Overall Survival Count")
    axes[0].set_ylabel("Passengers")

    # By Sex
    sex_surv = df_full.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
    sex_surv.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#2ecc71"],
                  edgecolor="white", legend=False)
    axes[1].set_title("Survival by Sex")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    # By Pclass
    pc_surv = df_full.groupby(["Pclass", "Survived"]).size().unstack(fill_value=0)
    pc_surv.plot(kind="bar", ax=axes[2], color=["#e74c3c", "#2ecc71"],
                 edgecolor="white", legend=True)
    axes[2].set_title("Survival by Pclass")
    axes[2].set_xticklabels([f"Class {i}" for i in [1, 2, 3]], rotation=0)
    axes[2].legend(["Not Survived", "Survived"], fontsize=8)

    plt.tight_layout()
    return _fig_to_path(fig, "overview")


def chart_age_fare() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Age & Fare Distributions", fontsize=14, fontweight="bold")

    for surv, color, label in [(0, "#e74c3c", "Not Survived"), (1, "#2ecc71", "Survived")]:
        subset = df_full[df_full["Survived"] == surv]
        axes[0].hist(subset["Age"].dropna(), bins=30, alpha=0.6,
                     color=color, label=label, edgecolor="white")
        axes[1].hist(subset["Fare"].dropna(), bins=30, alpha=0.6,
                     color=color, label=label, edgecolor="white")

    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].legend()
    axes[1].set_title("Fare Distribution")
    axes[1].set_xlabel("Fare")
    axes[1].legend()

    plt.tight_layout()
    return _fig_to_path(fig, "age_fare")


def chart_confusion_matrix() -> str:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Survived", "Survived"],
                yticklabels=["Not Survived", "Survived"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    return _fig_to_path(fig, "cm")


def chart_feature_importance() -> str:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(importances)),
           importances[indices],
           color="#3498db", edgecolor="white")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Feature Importances (Random Forest)")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    return _fig_to_path(fig, "fi")


# ---------------------------------------------------------------------------
# 4.  PREDICTION FUNCTION
# ---------------------------------------------------------------------------

def predict_survival(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
):
    """Run inference and return a rich result string."""
    sex_enc      = 1 if sex == "Male" else 0
    embarked_enc = {"C": 0, "Q": 1, "S": 2}[embarked]
    title        = 0 if sex == "Male" else 1   # Mr → 0, Miss/Mrs → 1 approx.

    family_size  = sibsp + parch + 1
    is_alone     = int(family_size == 1)

    # Discretise age / fare to match training bands
    age_band  = min(int(age // (80 / 5)), 4)
    fare_band = 0 if fare < 7.91 else (1 if fare < 14.454 else (2 if fare < 31 else 3))

    row = np.array([[
        pclass, sex_enc, age, sibsp, parch,
        fare, embarked_enc, title,
        family_size, is_alone, age_band, fare_band,
    ]])

    prediction   = model.predict(row)[0]
    probability  = model.predict_proba(row)[0]
    surv_prob    = probability[1] * 100
    not_surv_prob = probability[0] * 100

    verdict = "✅  **SURVIVED**" if prediction == 1 else "❌  **DID NOT SURVIVE**"

    result = f"""
## Prediction Result

### {verdict}

| Metric | Value |
|--------|-------|
| Survival Probability | **{surv_prob:.1f}%** |
| Non-Survival Probability | **{not_surv_prob:.1f}%** |
| Model Confidence | **{"High" if max(surv_prob, not_surv_prob) > 75 else "Medium" if max(surv_prob, not_surv_prob) > 55 else "Low"}** |

---
**Input Summary**
- Passenger Class : {pclass}
- Sex : {sex}
- Age : {age} years
- Siblings / Spouses aboard : {sibsp}
- Parents / Children aboard : {parch}
- Fare paid : £{fare:.2f}
- Port of Embarkation : {embarked}
- Family Size : {family_size} | Travelling Alone : {"Yes" if is_alone else "No"}

---
*Model: Random Forest (200 trees) | Test Accuracy: {accuracy:.2%} | 5-Fold CV: {cv_score:.2%}*
"""
    return result.strip()


# ---------------------------------------------------------------------------
# 5.  GRADIO  UI
# ---------------------------------------------------------------------------

CSS = """
body { font-family: 'Segoe UI', sans-serif; }
.gr-button-primary { background: #2ecc71 !important; }
footer { display: none !important; }
"""

def build_ui():
    with gr.Blocks(title="Titanic Survival Predictor", css=CSS) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:20px 0 10px;">
          <h1 style="margin:0;font-size:2em;">🚢 Titanic Survival Predictor</h1>
          <p style="color:#666;margin:6px 0 0;">
            Enter passenger details below to predict survival chances using a
            trained <strong>Random Forest</strong> model.
          </p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1 : Predictor ─────────────────────────────────────────
            with gr.Tab("🔮 Predict Survival"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Passenger Details")
                        pclass   = gr.Radio([1, 2, 3], label="Passenger Class",
                                            info="1 = First, 2 = Second, 3 = Third", value=3)
                        sex      = gr.Radio(["Male", "Female"], label="Sex", value="Male")
                        age      = gr.Slider(1, 80, value=30, step=1, label="Age (years)")
                        fare     = gr.Slider(0, 520, value=32, step=0.5, label="Fare (£)")
                        embarked = gr.Radio(["S", "C", "Q"],
                                            label="Port of Embarkation",
                                            info="S = Southampton · C = Cherbourg · Q = Queenstown",
                                            value="S")
                        with gr.Row():
                            sibsp = gr.Number(value=0, label="Siblings / Spouses aboard",
                                             minimum=0, maximum=8, precision=0)
                            parch = gr.Number(value=0, label="Parents / Children aboard",
                                             minimum=0, maximum=6, precision=0)

                        predict_btn = gr.Button("⚡  Predict", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Result")
                        result_out = gr.Markdown(value="*Fill in the details and click Predict…*")

                predict_btn.click(
                    fn=predict_survival,
                    inputs=[pclass, sex, age, sibsp, parch, fare, embarked],
                    outputs=result_out,
                )

                # ── Quick examples ────────────────────────────────────────
                gr.Markdown("---\n#### 📋 Quick Examples")
                gr.Examples(
                    examples=[
                        [1, "Female", 38, 1, 0, 71.28, "C"],
                        [3, "Male",   22, 1, 0,  7.25, "S"],
                        [2, "Female", 26, 0, 0, 13.00, "S"],
                        [3, "Male",   35, 0, 0,  8.05, "S"],
                        [1, "Male",   54, 0, 0, 51.86, "S"],
                    ],
                    inputs=[pclass, sex, age, sibsp, parch, fare, embarked],
                    label="Click a row to populate the form",
                )

            # ── Tab 2 : Dataset Analytics ─────────────────────────────────
            with gr.Tab("📊 Dataset Analytics"):
                gr.Markdown("### Survival Overview")
                overview_img = gr.Image(value=chart_survival_overview(),
                                        label="", show_label=False, type="filepath")

                gr.Markdown("### Age & Fare Distributions")
                age_fare_img = gr.Image(value=chart_age_fare(),
                                        label="", show_label=False, type="filepath")

            # ── Tab 3 : Model Performance ──────────────────────────────────
            with gr.Tab("🤖 Model Performance"):
                gr.Markdown(f"""
### Model Metrics
| | Value |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Trees** | 200 |
| **Test Accuracy** | {accuracy:.4f} ({accuracy:.2%}) |
| **5-Fold CV Accuracy** | {cv_score:.4f} ({cv_score:.2%}) |
| **Train / Test Split** | 80% / 20% |
""")
                with gr.Row():
                    cm_img = gr.Image(value=chart_confusion_matrix(),
                                      label="Confusion Matrix", type="filepath")
                    fi_img = gr.Image(value=chart_feature_importance(),
                                      label="Feature Importances", type="filepath")

                gr.Markdown("### Classification Report (Test Set)")
                gr.Textbox(value=report, label="", lines=12, interactive=False)

            # ── Tab 4 : About ──────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
## About This Project

### Overview
This application predicts whether a Titanic passenger would have survived the disaster,
based on personal and ticket attributes.  It uses a **Random Forest Classifier** trained
on the classic Titanic dataset.

### Dataset
- **Source** : [Titanic Survival Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/arjunsinghgangwar/titanic-survival-prediction-dataset/data)
- **File** : `Titanic-Dataset.csv`
- **Rows** : {len(df_full)} passengers
- **Survival Rate** : {df_full['Survived'].mean():.1%}

### Features Used
| Feature | Description |
|---------|-------------|
| Pclass | Ticket class (1st / 2nd / 3rd) |
| Sex | Gender of the passenger |
| Age | Age in years |
| SibSp | # of siblings / spouses aboard |
| Parch | # of parents / children aboard |
| Fare | Passenger fare |
| Embarked | Port of embarkation (C / Q / S) |
| Title | Derived from name (Mr / Miss / Mrs / Master / Rare) |
| FamilySize | SibSp + Parch + 1 |
| IsAlone | 1 if FamilySize == 1 |
| AgeBand | Age discretised into 5 equal-width bands |
| FareBand | Fare discretised into 4 quartile bands |

### Technology Stack
| Layer | Tool |
|-------|------|
| Data | pandas, NumPy |
| ML | scikit-learn (RandomForestClassifier) |
| Visualisation | matplotlib, seaborn |
| Frontend / UI | Gradio |

### Model Accuracy
**{accuracy:.2%}** on held-out test set · **{cv_score:.2%}** 5-fold cross-validation mean
""")

        gr.HTML("""
        <div style="text-align:center;color:#999;font-size:12px;
                    border-top:1px solid #eee;padding:12px 0 4px;margin-top:16px;">
          Titanic Survival Predictor — Built with Python · scikit-learn · Gradio
        </div>
        """)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
