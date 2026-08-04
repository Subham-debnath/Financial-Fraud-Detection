import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import glob
from datetime import datetime

try:
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score, classification_report
    )
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================
MODEL_INFO = {
    "algorithm": "Random Forest",
    "accuracy": 0.999507,
    "precision": 0.971831,
    "recall": 0.726316,
    "f1_score": 0.831325,
    "dataset": "Credit Card Dataset (Kaggle)",
    "training_samples": "N/A",
}

REQUIRED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL_COLUMN_CANDIDATES = ["Class", "class", "Label", "label", "Actual", "actual", "is_fraud", "isFraud"]

COLORS = {
    "navy": "#0B1F3A", "navy_light": "#16314F", "accent": "#4F7CFF", "accent2": "#7C5CFF",
    "accent_soft": "#EAF0FF", "fraud": "#E5484D", "fraud_soft": "#FDEAEA",
    "legit": "#1FA97A", "legit_soft": "#E8F8F1", "muted": "#64748B",
}
FRAUD_MAP = {"Legitimate": COLORS["legit"], "Fraud": COLORS["fraud"]}

st.set_page_config(page_title="Financial Fraud Detection", page_icon="💳", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# THEME
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
:root {{ --navy:{COLORS['navy']}; --accent:{COLORS['accent']}; --accent2:{COLORS['accent2']};
  --accent-soft:{COLORS['accent_soft']}; --fraud:{COLORS['fraud']}; --fraud-soft:{COLORS['fraud_soft']};
  --legit:{COLORS['legit']}; --legit-soft:{COLORS['legit_soft']}; --muted:{COLORS['muted']}; }}

.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #EEF3FF 0%, transparent 60%),
  radial-gradient(1000px 500px at 100% 0%, #F3FBF7 0%, transparent 55%), #F7F9FC; }}

section[data-testid="stSidebar"] {{ background: linear-gradient(200deg, var(--navy) 0%, #10233D 55%, #0A1830 100%); }}
section[data-testid="stSidebar"] * {{ color:#E8EEF9 !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12); }}
section[data-testid="stSidebar"] .stRadio label {{ font-weight:600; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
  padding:0.35rem 0.6rem; border-radius:8px; transition:background .15s ease; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background: rgba(255,255,255,0.06); }}

h1,h2,h3 {{ color:var(--navy); font-weight:800 !important; letter-spacing:-0.02em; }}

.hero-banner {{ background: linear-gradient(120deg, var(--navy) 0%, #1E3A5F 55%, var(--accent) 140%);
  padding: 3rem 2.6rem; border-radius: 20px; color:white; margin-bottom: 1.8rem;
  box-shadow: 0 16px 40px rgba(11,31,58,0.28); position:relative; overflow:hidden; }}
.hero-banner::after {{ content:""; position:absolute; inset:0;
  background: radial-gradient(circle at 85% 20%, rgba(124,92,255,0.35), transparent 55%); }}
.hero-banner h1 {{ color:white !important; font-size:2.5rem; margin-bottom:0.6rem; position:relative; }}
.hero-banner p {{ color:#D6E2F5; font-size:1.05rem; max-width:640px; margin:0; position:relative; }}
.hero-badge {{ display:inline-block; background:rgba(255,255,255,0.14); color:#EAF0FF; font-size:0.75rem;
  font-weight:700; letter-spacing:0.06em; text-transform:uppercase; padding:0.3rem 0.8rem;
  border-radius:999px; margin-bottom:1rem; position:relative; border:1px solid rgba(255,255,255,0.2); }}

.feature-card, .glass-card {{ background:white; border:1px solid #EAEFF5; border-radius:16px; padding:1.6rem;
  height:100%; box-shadow:0 4px 16px rgba(15,30,60,0.05); transition:transform .15s ease, box-shadow .15s ease; }}
.feature-card:hover {{ transform:translateY(-4px); box-shadow:0 14px 28px rgba(15,30,60,0.12); border-color:#DCE6F7; }}
.feature-card .icon {{ font-size:1.9rem; margin-bottom:0.5rem;
  filter: drop-shadow(0 4px 8px rgba(79,124,255,0.25)); }}
.feature-card h4 {{ margin:0 0 0.4rem 0; color:var(--navy); font-weight:700; }}
.feature-card p {{ color:var(--muted); font-size:0.92rem; margin:0; }}

.section-pill {{ display:inline-block; background:var(--accent-soft); color:var(--accent); font-weight:700;
  font-size:0.75rem; letter-spacing:0.06em; text-transform:uppercase; padding:0.3rem 0.85rem;
  border-radius:999px; margin-bottom:0.6rem; }}

.metric-card {{ background:white; border:1px solid #EAEFF5; border-radius:16px; padding:1.3rem 1.5rem;
  box-shadow:0 4px 16px rgba(15,30,60,0.05); text-align:left; position:relative; overflow:hidden;
  transition:transform .15s ease; }}
.metric-card:hover {{ transform:translateY(-2px); }}
.metric-card::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); }}
.metric-card.fraud::before {{ background:var(--fraud); }}
.metric-card.legit::before {{ background:var(--legit); }}
.metric-card .label {{ color:var(--muted); font-size:0.8rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.04em; margin-bottom:0.4rem; }}
.metric-card .value {{ font-size:2rem; font-weight:800; color:var(--navy); }}
.metric-card.fraud .value {{ color:var(--fraud); }}
.metric-card.legit .value {{ color:var(--legit); }}
.metric-card.accent .value {{ color:var(--accent); }}

.stButton>button, .stDownloadButton>button {{ background:linear-gradient(120deg,var(--navy),#173a63) !important;
  color:white !important; border-radius:10px !important; border:none !important; font-weight:600 !important;
  padding:0.55rem 1.3rem !important; transition:all .15s ease !important; box-shadow:0 4px 12px rgba(11,31,58,0.18); }}
.stButton>button:hover, .stDownloadButton>button:hover {{
  background:linear-gradient(120deg,var(--accent),var(--accent2)) !important; transform:translateY(-1px);
  box-shadow:0 8px 18px rgba(79,124,255,0.3); }}

[data-testid="stDataFrame"] {{ border-radius:14px; overflow:hidden; border:1px solid #EAEFF5;
  box-shadow:0 2px 10px rgba(15,30,60,0.04); }}
[data-testid="stFileUploaderDropzone"] {{ border-radius:14px; border:2px dashed #C7D3E6 !important; background:#FBFCFE; }}
div[data-testid="stAlert"] {{ border-radius:12px; }}

.result-banner {{ border-radius:16px; padding:1.5rem 1.7rem; font-weight:700; font-size:1.2rem; margin:0.8rem 0;
  box-shadow:0 6px 20px rgba(15,30,60,0.06); }}
.result-banner.fraud {{ background:var(--fraud-soft); color:var(--fraud); border:1px solid rgba(229,72,77,0.25); }}
.result-banner.legit {{ background:var(--legit-soft); color:var(--legit); border:1px solid rgba(31,169,122,0.25); }}
hr {{ margin:1.4rem 0; }}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, variant=""):
    st.markdown(f'<div class="metric-card {variant}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


def style_fig(fig, title=None, legend=True):
    """Apply consistent, professional theming to any Plotly figure."""
    fig.update_layout(
        title=title or fig.layout.title.text,
        font=dict(family="Inter, sans-serif", color=COLORS["navy"]),
        title_font=dict(size=16, family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=55, l=10, r=10, b=10),
        legend=dict(orientation="h", y=-0.18) if legend else dict(),
        showlegend=legend,
    )
    return fig


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)


def discover_models():
    found = sorted(glob.glob("*.pkl"))
    return found if found else ["model.pkl"]


def get_model(path):
    try:
        return load_model(path), None
    except FileNotFoundError:
        return None, f"`{path}` not found. Please add the trained model file to the app directory."
    except Exception as e:
        return None, f"Failed to load model: {e}"


def score_dataframe(model, X, threshold):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int), proba.round(4)
    return model.predict(X), None


# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("history", [])

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style="text-align:center; padding:0.5rem 0 1.2rem 0;">
  <div style="font-size:2.2rem;">💳</div>
  <div style="font-size:1.15rem; font-weight:800; letter-spacing:-0.01em;">Fraud Detection</div>
  <div style="font-size:0.8rem; color:#B7C6E0; margin-top:0.2rem;">ML-Powered Transaction Analysis</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "🏠 Home", "📊 Upload & Predict", "🔎 Single Transaction",
    "🕒 Prediction History", "🧠 Model Info", "ℹ️ About",
], label_visibility="collapsed")

available_models = discover_models()
if len(available_models) > 1:
    st.sidebar.markdown("---")
    st.sidebar.caption("MODEL")
    selected_model_path = st.sidebar.selectbox("Choose a model", available_models, label_visibility="collapsed")
else:
    selected_model_path = available_models[0]

st.sidebar.markdown("---")
st.sidebar.caption("Built with scikit-learn / XGBoost + Streamlit")

# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">Real-Time ML Scoring</div>
        <h1>💳 Financial Fraud Detection</h1>
        <p>An end-to-end system that analyzes transaction data and flags potentially
        fraudulent activity in real time — powered by a trained classification model
        and an interactive dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-pill">How it works</span>', unsafe_allow_html=True)
    features = [
        ("📤", "Upload", "Upload a CSV of transactions and preview your data instantly."),
        ("🔍", "Predict", "Every transaction is scored with an adjustable fraud threshold."),
        ("📈", "Analyze", "Explore results, feature drivers, and evaluation metrics."),
        ("⬇️", "Export", "Download the full results or just the flagged fraud cases."),
    ]
    for col, (icon, title, desc) in zip(st.columns(4), features):
        col.markdown(f'<div class="feature-card"><div class="icon">{icon}</div>'
                      f'<h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👉 Head to **Upload & Predict** from the sidebar to get started, "
            "or try **Single Transaction** to score one transaction by hand.")

# ============================================================
# PAGE: UPLOAD & PREDICT
# ============================================================
elif page == "📊 Upload & Predict":
    st.markdown('<span class="section-pill">Live Prediction</span>', unsafe_allow_html=True)
    st.title("📊 Upload & Predict")
    st.caption("Upload a transaction CSV to generate fraud predictions and explore the results.")

    model, error = get_model(selected_model_path)
    if error:
        st.error(error); st.stop()

    threshold = 0.5
    if hasattr(model, "predict_proba"):
        threshold = st.slider("Fraud probability threshold", 0.0, 1.0, 0.5, 0.01,
            help="Transactions with predicted fraud probability at or above this value are "
                 "flagged as Fraud. Lower it to catch more fraud at the cost of more false "
                 "alarms; raise it to reduce false alarms at the cost of missed fraud.")
    else:
        st.info("This model doesn't expose probability scores, so the threshold slider is "
                "unavailable — predictions use the model's default decision rule.")

    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}"); st.stop()

        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}"); st.stop()
        if len(df) == 0:
            st.warning("The uploaded file has no rows to score."); st.stop()

        X = df[REQUIRED_COLUMNS]
        label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns), None)

        na_rows = X.isnull().any(axis=1)
        if na_rows.any():
            st.warning(f"{int(na_rows.sum())} row(s) contain missing values in feature "
                       "columns and will be dropped before scoring.")
            df = df.loc[~na_rows].reset_index(drop=True)
            X = df[REQUIRED_COLUMNS]
        if len(df) == 0:
            st.warning("No valid rows remain after removing missing values."); st.stop()

        with st.spinner("Scoring transactions..."):
            try:
                preds, proba = score_dataframe(model, X, threshold)
            except Exception as e:
                st.error(f"Prediction failed: {e}"); st.stop()

        df["Prediction"] = pd.Series(preds).map({0: "Legitimate", 1: "Fraud"})
        if proba is not None:
            df["Fraud_Probability"] = proba

        # ---- Metrics ----
        st.markdown("#### Dashboard Metrics")
        total_count = len(df)
        fraud_count = int((df["Prediction"] == "Fraud").sum())
        legit_count = total_count - fraud_count
        fraud_rate = (fraud_count / total_count * 100) if total_count else 0

        for col, (label, val, variant) in zip(st.columns(4), [
            ("Total Transactions", f"{total_count:,}", "accent"),
            ("Fraud Transactions", f"{fraud_count:,}", "fraud"),
            ("Legitimate Transactions", f"{legit_count:,}", "legit"),
            ("Fraud Rate", f"{fraud_rate:.2f}%", "fraud"),
        ]):
            with col:
                metric_card(label, val, variant)

        if fraud_count == 0:
            st.success("No transactions were flagged as fraud at this threshold.")
        if proba is not None and fraud_count > 0:
            avg_conf = df.loc[df["Prediction"] == "Fraud", "Fraud_Probability"].mean()
            st.caption(f"Average model confidence on flagged fraud: **{avg_conf:.2%}**")

        # ---- Charts ----
        st.markdown("#### Visual Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(names=["Legitimate", "Fraud"], values=[legit_count, fraud_count],
                         color=["Legitimate", "Fraud"], color_discrete_map=FRAUD_MAP, hole=0.5)
            st.plotly_chart(style_fig(fig, "Transaction Distribution"), use_container_width=True)
        with c2:
            fig = px.bar(x=["Legitimate", "Fraud"], y=[legit_count, fraud_count],
                         labels={"x": "Class", "y": "Count"}, color=["Legitimate", "Fraud"],
                         color_discrete_map=FRAUD_MAP)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(style_fig(fig, "Transaction Counts", legend=False), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(df, x="Amount", color="Prediction", nbins=40, barmode="overlay",
                               color_discrete_map=FRAUD_MAP, opacity=0.7)
            st.plotly_chart(style_fig(fig, "Transaction Amount by Class"), use_container_width=True)
        with c4:
            fig = px.scatter(df, x="Time", y="Amount", color="Prediction",
                             color_discrete_map=FRAUD_MAP, opacity=0.6)
            st.plotly_chart(style_fig(fig, "Transactions Over Time"), use_container_width=True)

        # ---- Feature importance ----
        if hasattr(model, "feature_importances_"):
            st.markdown("#### What Drives These Predictions")
            st.caption("Global feature importance from the trained model (not specific to any single transaction).")
            top_imp = pd.Series(model.feature_importances_, index=REQUIRED_COLUMNS).sort_values(ascending=False).head(10)
            fig = px.bar(x=top_imp.values, y=top_imp.index, orientation="h",
                        labels={"x": "Importance", "y": "Feature"})
            fig.update_traces(marker_color=COLORS["accent"], marker_line_width=0)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, "Top 10 Most Influential Features", legend=False), use_container_width=True)

        # ---- Evaluation vs ground truth ----
        if label_col and SKLEARN_METRICS_AVAILABLE:
            st.markdown("#### Evaluation Against Ground Truth")
            st.caption(f"Detected a label column (`{label_col}`) — comparing predictions against it.")
            y_true = df[label_col].astype(int)
            y_pred = (df["Prediction"] == "Fraud").astype(int)

            e1, e2 = st.columns(2)
            with e1:
                cm = confusion_matrix(y_true, y_pred)
                fig = px.imshow(cm, text_auto=True, x=["Predicted Legitimate", "Predicted Fraud"],
                                y=["Actual Legitimate", "Actual Fraud"], color_continuous_scale="Blues")
                st.plotly_chart(style_fig(fig, "Confusion Matrix", legend=False), use_container_width=True)
            with e2:
                if proba is not None:
                    fpr, tpr, _ = roc_curve(y_true, proba)
                    roc_auc = auc(fpr, tpr)
                    fig = go.Figure([
                        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})",
                                   line=dict(color=COLORS["accent"])),
                        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                   line=dict(dash="dash", color="#C7D3E6")),
                    ])
                    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    st.plotly_chart(style_fig(fig, "ROC Curve"), use_container_width=True)
                else:
                    st.info("ROC curve requires probability scores, which this model doesn't provide.")

            if proba is not None:
                prec, rec, _ = precision_recall_curve(y_true, proba)
                ap = average_precision_score(y_true, proba)
                fig = go.Figure(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AP = {ap:.3f})",
                                           line=dict(color=COLORS["fraud"])))
                fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(style_fig(fig, "Precision-Recall Curve"), use_container_width=True)

            with st.expander("Classification report"):
                st.code(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))

        # ---- Results table ----
        st.markdown("#### Prediction Results")
        show_fraud_only = st.checkbox("Show fraud transactions only")
        st.dataframe(df[df["Prediction"] == "Fraud"] if show_fraud_only else df, use_container_width=True)

        # ---- Downloads ----
        st.markdown("#### Download Results")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download Full Results", df.to_csv(index=False),
                           file_name="predictions_full.csv", mime="text/csv")
        d2.download_button("⬇️ Download Fraud-Only Results", df[df["Prediction"] == "Fraud"].to_csv(index=False),
                           file_name="predictions_fraud_only.csv", mime="text/csv")

        st.session_state.history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "File": uploaded_file.name,
            "Model": selected_model_path, "Threshold": threshold, "Total": total_count,
            "Fraud": fraud_count, "Legitimate": legit_count, "Fraud Rate": f"{fraud_rate:.2f}%",
        })

# ============================================================
# PAGE: SINGLE TRANSACTION
# ============================================================
elif page == "🔎 Single Transaction":
    st.markdown('<span class="section-pill">Manual Scoring</span>', unsafe_allow_html=True)
    st.title("🔎 Score a Single Transaction")
    st.caption("Enter a transaction's feature values by hand to get an instant prediction — "
               "useful for quick demos without preparing a CSV.")

    model, error = get_model(selected_model_path)
    if error:
        st.error(error); st.stop()

    threshold = 0.5
    if hasattr(model, "predict_proba"):
        threshold = st.slider("Fraud probability threshold", 0.0, 1.0, 0.5, 0.01, key="single_threshold")

    with st.form("single_transaction_form"):
        st.markdown("##### Core Fields")
        c1, c2 = st.columns(2)
        time_val = c1.number_input("Time", value=0.0, step=1.0)
        amount_val = c2.number_input("Amount", value=0.0, step=1.0, min_value=0.0)

        st.markdown("##### PCA Features (V1–V28)")
        v_values = {}
        v_cols = st.columns(4)
        for i in range(1, 29):
            v_values[f"V{i}"] = v_cols[(i - 1) % 4].number_input(f"V{i}", value=0.0, step=0.1, key=f"v_{i}")

        submitted = st.form_submit_button("Predict")

    if submitted:
        X_single = pd.DataFrame([{"Time": time_val, **v_values, "Amount": amount_val}])[REQUIRED_COLUMNS]
        with st.spinner("Scoring transaction..."):
            try:
                pred, proba = score_dataframe(model, X_single, threshold)
            except Exception as e:
                st.error(f"Prediction failed: {e}"); st.stop()

        is_fraud = pred[0] == 1
        label = "Fraud" if is_fraud else "Legitimate"
        variant = "fraud" if is_fraud else "legit"
        icon = "🚨" if is_fraud else "✅"
        conf_text = f" (confidence: {proba[0]:.2%})" if proba is not None else ""
        st.markdown(f'<div class="result-banner {variant}">{icon} Predicted: {label}{conf_text}</div>',
                   unsafe_allow_html=True)

        if SHAP_AVAILABLE and hasattr(model, "predict_proba"):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_single)
                sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                contrib = pd.Series(np.ravel(sv), index=REQUIRED_COLUMNS).sort_values(key=abs, ascending=False).head(10)
                st.markdown("##### Why this prediction?")
                st.caption("Top features pushing this specific transaction toward or away from Fraud.")
                fig = px.bar(x=contrib.values, y=contrib.index, orientation="h",
                            labels={"x": "SHAP value (impact on fraud score)", "y": "Feature"},
                            color=contrib.values, color_continuous_scale=[COLORS["legit"], COLORS["fraud"]])
                fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig, legend=False), use_container_width=True)
            except Exception:
                st.caption("Per-transaction explanation unavailable for this model type.")
        elif hasattr(model, "feature_importances_"):
            st.caption("Install the `shap` package for a per-transaction explanation. Showing overall top features instead.")
            top_imp = pd.Series(model.feature_importances_, index=REQUIRED_COLUMNS).sort_values(ascending=False).head(10)
            st.bar_chart(top_imp)

# ============================================================
# PAGE: PREDICTION HISTORY
# ============================================================
elif page == "🕒 Prediction History":
    st.markdown('<span class="section-pill">Session Log</span>', unsafe_allow_html=True)
    st.title("🕒 Prediction History")
    st.caption("Batch predictions you've run this session. Cleared when the app restarts.")

    if not st.session_state.history:
        st.info("No batch predictions yet — run one from **Upload & Predict** and it'll show up here.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        fig = px.line(hist_df, x="Timestamp", y="Fraud", markers=True)
        fig.update_traces(line_color=COLORS["accent"])
        st.plotly_chart(style_fig(fig, "Fraud Count Across Runs This Session", legend=False), use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ============================================================
# PAGE: MODEL INFO
# ============================================================
elif page == "🧠 Model Info":
    st.markdown('<span class="section-pill">Under the hood</span>', unsafe_allow_html=True)
    st.title("🧠 Model Information")
    st.caption("Details about the machine learning model powering this fraud detection system.")

    i1, i2 = st.columns(2)
    i1.markdown(f"""<div class="feature-card"><h4>📋 Model Details</h4>
        <p><b>Algorithm:</b> {MODEL_INFO['algorithm']}</p>
        <p><b>Training Dataset:</b> {MODEL_INFO['dataset']}</p>
        <p><b>Training Samples:</b> {MODEL_INFO['training_samples']}</p>
        <p><b>Active Model File:</b> {selected_model_path}</p></div>""", unsafe_allow_html=True)
    i2.markdown("""<div class="feature-card"><h4>🎯 Why These Metrics</h4>
        <p>Fraud datasets are highly imbalanced, so accuracy alone can be misleading.
        Precision, recall, and F1-score give a fuller picture of how well the model
        catches fraud without over-flagging legitimate transactions.</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>#### Performance Metrics", unsafe_allow_html=True)
    for col, (label, key, variant) in zip(st.columns(4), [
        ("Accuracy", "accuracy", "accent"), ("Precision", "precision", "legit"),
        ("Recall", "recall", "legit"), ("F1-Score", "f1_score", "accent"),
    ]):
        with col:
            metric_card(label, f"{MODEL_INFO[key] * 100:.2f}%", variant)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = px.bar(x=["Accuracy", "Precision", "Recall", "F1-Score"],
                y=[MODEL_INFO[k] for k in ["accuracy", "precision", "recall", "f1_score"]],
                labels={"x": "Metric", "y": "Score"}, range_y=[0, 1],
                color=["Accuracy", "Precision", "Recall", "F1-Score"],
                color_discrete_sequence=[COLORS["accent"], COLORS["legit"], COLORS["legit"], COLORS["accent"]])
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(style_fig(fig, "Model Performance Overview", legend=False), use_container_width=True)

    st.info("⚠️ These values are placeholders — update `MODEL_INFO` at the top of the script with your "
            "model's real evaluation results. You can also upload a CSV with a ground-truth label column "
            "on the Upload & Predict page to compute live evaluation metrics.")

# ============================================================
# PAGE: ABOUT
# ============================================================
elif page == "ℹ️ About":
    st.markdown('<span class="section-pill">Project details</span>', unsafe_allow_html=True)
    st.title("ℹ️ About This Project")
    st.markdown("""
### Project Overview
This is a machine learning based system designed to detect fraudulent financial transactions.
It analyzes transaction-level features and classifies each transaction as **Legitimate** or
**Fraudulent**, along with a fraud probability score.

### Workflow
1. **Data Collection** — Transaction data sourced from a financial dataset.
2. **Preprocessing** — Cleaning, handling missing values, and addressing class imbalance.
3. **Feature Engineering** — Preparing and selecting relevant features for model training.
4. **Model Training** — Training and comparing algorithms such as Random Forest, XGBoost, and Neural Networks.
5. **Evaluation** — Assessing performance using accuracy, precision, recall, and F1-score.
6. **Deployment** — Serving the trained model through this interactive Streamlit application.

### Why This Matters
Fraudulent transactions are rare but costly. A good fraud detection system needs to catch as
many fraud cases as possible (**recall**) while minimizing false alarms on legitimate transactions
(**precision**). The adjustable threshold on the prediction page lets you explore that trade-off directly.

### Tech Stack
- **Python** — core language · **Streamlit** — web app framework
- **scikit-learn / XGBoost** — model training · **Plotly** — interactive visualizations
- **Pandas / NumPy** — data handling · **SHAP** *(optional)* — per-transaction explainability

### Note on dataset compatibility
This app expects `Time`, `V1`–`V28`, and `Amount` columns (the Kaggle Credit Card Fraud dataset
format). If your trained model instead uses the PaySim dataset's raw columns (`step`, `type`,
`amount`, `oldbalanceOrg`, etc.), update `REQUIRED_COLUMNS` and the single-transaction form accordingly.
""")