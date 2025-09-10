import streamlit as st
import pandas as pd
import torch, joblib, json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# Config
# =========================
ART_DIR = Path("mlp_llm_targets_artifacts")
TARGET = "llm_mean_score"   # only CoLLM prediction
ICD_FILE = "icd_dict.json"  # ICD-10 dictionary file
LOGO_FILE = "cwru_logo.png" # path to your logo image
UKB_REF = "ukbscores.csv"   # UKB reference dataset with sex, ref_age, llm_mean_score_distilled

# ---------------- Load ICD Dictionary ---------------- #
with open(ICD_FILE, "r") as f:
    icd_dict = json.load(f)

# ---------------- Model ---------------- #
def make_activation(name: str):
    return {"relu": torch.nn.ReLU, "leakyrelu": torch.nn.LeakyReLU}[name.lower()]()

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout, activation="relu"):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [
                torch.nn.Linear(last, int(h)),
                make_activation(activation),
                torch.nn.Dropout(float(dropout)),
            ]
            last = int(h)
        layers += [torch.nn.Linear(last, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model(target, art_dir=ART_DIR):
    weights_path  = art_dir / f"mlp_{target}.pth"
    features_path = art_dir / f"features_{target}.pkl"
    config_path   = art_dir / f"best_config_{target}.json"
    scaler_path   = art_dir / f"scaler_{target}.pkl"

    feature_list = joblib.load(features_path)
    with open(config_path) as f:
        conf = json.load(f)

    hidden     = tuple(conf["hidden"]) if isinstance(conf["hidden"], (list, tuple)) else tuple(eval(str(conf["hidden"])))
    dropout    = float(conf.get("dropout", 0.2))
    activation = str(conf.get("activation", "relu"))
    in_dim     = len(feature_list)

    model = MLP(in_dim, hidden, dropout, activation)
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, feature_list, scaler

# ---------------- Prediction ---------------- #
def predict_collm(age, gender, selected_codes):
    model, feature_list, scaler = load_model(TARGET)
    row = {f: 0 for f in feature_list}

    if "ref_age" in row:
        row["ref_age"] = age
    if "sex" in row:
        row["sex"] = 0 if gender == "Male" else 1

    # ICD features: match prefix
    for code in selected_codes:
        matches = [f for f in feature_list if f.startswith(code)]
        if matches:
            row[matches[0]] = 1

    df = pd.DataFrame([row])
    X_np = df.values.astype("float32")
    if scaler:
        X_np = scaler.transform(X_np)
    X_t = torch.tensor(X_np)

    with torch.no_grad():
        preds = model(X_t).numpy().ravel()
    return preds[0]

# ---------------- Load UKB Reference ---------------- #
@st.cache_data
def load_reference():
    df = pd.read_csv(UKB_REF)
    df["sex"] = df["sex"].map({0: "Male", 1: "Female"})
    bins = [0, 50, 60, 70, 80, 120]
    labels = ["<50", "50–59", "60–69", "70–79", "80+"]
    df["age_group"] = pd.cut(df["ref_age"], bins=bins, labels=labels, right=False)
    return df

df_ref = load_reference()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CoLLM Predictor", page_icon="🧠")

# Sidebar with logo + history
st.sidebar.image(LOGO_FILE, use_container_width=True)
st.sidebar.title("CoLLM App")

# initialize prediction history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# --- Clear All first (fixes double-click issue) ---
if st.sidebar.button("🗑️ Clear All History"):
    st.session_state.history = []
    st.sidebar.success("All history cleared.")

# Show history if available
if st.session_state.history:
    st.sidebar.markdown("### 📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.sidebar.dataframe(hist_df, use_container_width=True)

    # Row deletion UI
    delete_indices = st.sidebar.multiselect(
        "Select rows to delete:",
        options=list(range(len(hist_df))),
        format_func=lambda i: f"{i+1}: Age {hist_df.loc[i,'Age']}, Score {hist_df.loc[i,'CoLLM Score']}"
    )

    if st.sidebar.button("Delete Selected"):
        st.session_state.history = [
            row for i, row in enumerate(st.session_state.history) if i not in delete_indices
        ]
        st.sidebar.success("✅ Selected rows deleted.")

# ---------------- Main Page ----------------
st.title("🔮 CoLLM Comorbidity Score Prediction")

age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.radio("Gender", ["Male", "Female"])
selected = st.multiselect(
    "Select diagnoses:",
    options=list(icd_dict.keys()),
    format_func=lambda x: f"{x} - {icd_dict[x]}"
)

if st.button("Predict CoLLM Score"):
    score = predict_collm(age, gender, selected)

    # ---------------- Percentile + Category ----------------
    overall_pct = (df_ref["llm_mean_score_distilled"] < score).mean() * 100
    bins = [0, 50, 60, 70, 80, 120]
    labels = ["<50", "50–59", "60–69", "70–79", "80+"]
    age_group = pd.cut([age], bins=bins, labels=labels, right=False)[0]
    df_group = df_ref[df_ref["age_group"] == age_group]
    group_pct = (df_group["llm_mean_score_distilled"] < score).mean() * 100 if not df_group.empty else np.nan

    # Risk category (based on age group percentile)
    if group_pct < 25:
        risk = "Low"
        color = "green"
    elif group_pct < 60:
        risk = "Moderate"
        color = "orange"
    else:
        risk = "High"
        color = "red"

    # ---------------- Output ----------------
    st.markdown(f"### ✅ Predicted CoLLM Score: **{score:.3f}**")
    st.markdown(f"<span style='color:{color};font-weight:bold;'>Risk Category: {risk}</span>", unsafe_allow_html=True)

    st.markdown(f"""
    **Percentile Ranks:**
    - Overall UKB: {overall_pct:.1f}th percentile
    - Age group ({age_group}): {group_pct:.1f}th percentile
    """)

    # Plot distribution
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(df_ref["llm_mean_score_distilled"], bins=50, alpha=0.4, color="gray", label="All UKB")
    if not df_group.empty:
        ax.hist(df_group["llm_mean_score_distilled"], bins=30, alpha=0.5, color="blue", label=f"Age group {age_group}")
    ax.axvline(score, color="red", linestyle="--", linewidth=2, label="Your score")
    ax.set_xlabel("CoLLM Distilled Score")
    ax.set_ylabel("Count")
    ax.set_title("Your Score vs UKB Reference")
    ax.legend()
    st.pyplot(fig)

    # Narrative
    st.markdown(f"""
    🩺 **Interpretation:**  
    Your CoLLM score of **{score:.2f}** places you in the **{overall_pct:.1f}th percentile overall**  
    and the **{group_pct:.1f}th percentile within your age group ({age_group})**.  
    This corresponds to a **{risk} comorbidity burden** relative to the UK Biobank population.  
    """)

    # Save to history
    st.session_state.history.append({
        "Age": age,
        "Sex": gender,
        "Diagnoses": ", ".join(selected) if selected else "None",
        "CoLLM Score": round(score, 3),
        "Risk": risk
    })
