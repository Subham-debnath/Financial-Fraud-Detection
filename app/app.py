import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent

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
    "legit": "#1FA97A", "legit_soft": "#E8F8F1", "amber": "#F5A623", "amber_soft": "#FFF4E0",
    "muted": "#64748B",
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
html {{ color-scheme: light; }}
/* Low-specificity reset so plain Streamlit text stays readable even if a
   visitor's browser/OS is set to dark mode. Any element with its own class
   (hero-banner, feature-card, metric-card, sidebar, etc.) already sets its
   own color with higher specificity and overrides this automatically. */
p, li, span, div, label, small, td, th {{ color:{COLORS['navy']}; }}

:root {{ --navy:{COLORS['navy']}; --accent:{COLORS['accent']}; --accent2:{COLORS['accent2']};
  --accent-soft:{COLORS['accent_soft']}; --fraud:{COLORS['fraud']}; --fraud-soft:{COLORS['fraud_soft']};
  --legit:{COLORS['legit']}; --legit-soft:{COLORS['legit_soft']}; --amber:{COLORS['amber']};
  --amber-soft:{COLORS['amber_soft']}; --muted:{COLORS['muted']}; }}

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

/* Hero */
.hero-banner {{ background: linear-gradient(120deg, var(--navy) 0%, #1E3A5F 55%, var(--accent) 140%);
  padding: 3rem 2.6rem; border-radius: 20px; color:white; margin-bottom: 1.8rem;
  box-shadow: 0 16px 40px rgba(11,31,58,0.28); position:relative; overflow:hidden; }}
.hero-banner::after {{ content:""; position:absolute; inset:0;
  background: radial-gradient(circle at 85% 20%, rgba(124,92,255,0.35), transparent 55%); }}
.hero-banner h1 {{ color:white !important; font-size:2.5rem; margin-bottom:0.6rem; position:relative; }}
.hero-banner p {{ color:#D6E2F5; font-size:1.05rem; max-width:640px; margin:0; position:relative; }}
.hero-badge {{ display:inline-flex; align-items:center; gap:0.4rem; background:rgba(255,255,255,0.14);
  color:#EAF0FF; font-size:0.75rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase;
  padding:0.3rem 0.8rem; border-radius:999px; margin-bottom:1rem; position:relative;
  border:1px solid rgba(255,255,255,0.2); }}
.live-dot {{ width:7px; height:7px; border-radius:50%; background:#3DDC97; box-shadow:0 0 0 rgba(61,220,151,0.6);
  animation: pulseDot 1.8s infinite; display:inline-block; }}
@keyframes pulseDot {{
  0% {{ box-shadow:0 0 0 0 rgba(61,220,151,0.55); }}
  70% {{ box-shadow:0 0 0 8px rgba(61,220,151,0); }}
  100% {{ box-shadow:0 0 0 0 rgba(61,220,151,0); }}
}}

/* Cards */
.feature-card, .glass-card {{ background:white; border:1px solid #EAEFF5; border-radius:16px; padding:1.6rem;
  height:100%; box-shadow:0 4px 16px rgba(15,30,60,0.05); transition:transform .15s ease, box-shadow .15s ease; }}
.feature-card:hover {{ transform:translateY(-4px); box-shadow:0 14px 28px rgba(15,30,60,0.12); border-color:#DCE6F7; }}
.feature-card .icon {{ font-size:1.9rem; margin-bottom:0.5rem; filter: drop-shadow(0 4px 8px rgba(79,124,255,0.25)); }}
.feature-card h4 {{ margin:0 0 0.4rem 0; color:var(--navy); font-weight:700; }}
.feature-card p {{ color:var(--muted); font-size:0.92rem; margin:0; }}

/* Nav cards (dashboard home) */
.nav-card {{ background:white; border:1px solid #EAEFF5; border-radius:16px; padding:1.5rem 1.5rem 1.1rem 1.5rem;
  box-shadow:0 4px 16px rgba(15,30,60,0.05); transition:transform .15s ease, box-shadow .15s ease;
  margin-bottom:0.6rem; position:relative; overflow:hidden; }}
.nav-card:hover {{ transform:translateY(-4px); box-shadow:0 16px 30px rgba(15,30,60,0.14); border-color:#DCE6F7; }}
.nav-card .nav-icon {{ font-size:1.7rem; width:48px; height:48px; display:flex; align-items:center; justify-content:center;
  background:var(--accent-soft); border-radius:12px; margin-bottom:0.8rem; }}
.nav-card h4 {{ margin:0 0 0.3rem 0; color:var(--navy); font-weight:800; font-size:1.05rem; }}
.nav-card p {{ color:var(--muted); font-size:0.88rem; margin:0 0 0.6rem 0; min-height:2.6rem; }}
.nav-card .stButton>button {{ width:100%; background:var(--accent-soft) !important; color:var(--accent) !important;
  box-shadow:none !important; font-weight:700 !important; }}
.nav-card .stButton>button:hover {{ background:var(--accent) !important; color:white !important; transform:none; }}

.section-pill {{ display:inline-block; background:var(--accent-soft); color:var(--accent); font-weight:700;
  font-size:0.75rem; letter-spacing:0.06em; text-transform:uppercase; padding:0.3rem 0.85rem;
  border-radius:999px; margin-bottom:0.6rem; }}

/* KPI / metric cards with staggered entrance animation */
@keyframes fadeSlideUp {{
  from {{ opacity:0; transform:translateY(10px); }}
  to {{ opacity:1; transform:translateY(0); }}
}}
.metric-card {{ background:white; border:1px solid #EAEFF5; border-radius:16px; padding:1.3rem 1.5rem;
  box-shadow:0 4px 16px rgba(15,30,60,0.05); text-align:left; position:relative; overflow:hidden;
  transition:transform .15s ease; animation: fadeSlideUp .5s ease both; }}
.metric-card:hover {{ transform:translateY(-2px); }}
.metric-card::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); }}
.metric-card.fraud::before {{ background:var(--fraud); }}
.metric-card.legit::before {{ background:var(--legit); }}
.metric-card.alert {{ animation: fadeSlideUp .5s ease both, cardGlow 2.2s ease-in-out infinite; }}
@keyframes cardGlow {{
  0%, 100% {{ box-shadow:0 4px 16px rgba(15,30,60,0.05); }}
  50% {{ box-shadow:0 4px 22px rgba(229,72,77,0.28); }}
}}
.metric-card .label {{ color:var(--muted); font-size:0.8rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.04em; margin-bottom:0.4rem; display:flex; justify-content:space-between; align-items:center; }}
.metric-card .value {{ font-size:2rem; font-weight:800; color:var(--navy); }}
.metric-card.fraud .value {{ color:var(--fraud); }}
.metric-card.legit .value {{ color:var(--legit); }}
.metric-card.accent .value {{ color:var(--accent); }}
.metric-card .delta {{ font-size:0.78rem; font-weight:600; color:var(--muted); margin-top:0.25rem; }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{ background:linear-gradient(120deg,var(--navy),#173a63) !important;
  color:white !important; border-radius:10px !important; border:none !important; font-weight:600 !important;
  padding:0.55rem 1.3rem !important; transition:all .15s ease !important; box-shadow:0 4px 12px rgba(11,31,58,0.18); }}
.stButton>button:hover, .stDownloadButton>button:hover {{
  background:linear-gradient(120deg,var(--accent),var(--accent2)) !important; transform:translateY(-1px);
  box-shadow:0 8px 18px rgba(79,124,255,0.3); }}

/* Upload area */
.upload-instructions {{ background:white; border:1px solid #EAEFF5; border-radius:14px; padding:1.1rem 1.4rem;
  margin-bottom:0.9rem; box-shadow:0 2px 10px rgba(15,30,60,0.04); }}
.upload-instructions .cols-chip {{ display:inline-block; background:var(--accent-soft); color:var(--accent);
  font-family: 'JetBrains Mono', monospace; font-size:0.72rem; font-weight:700; padding:0.15rem 0.5rem;
  border-radius:6px; margin:0.15rem 0.2rem 0.15rem 0; }}
[data-testid="stFileUploaderDropzone"] {{ border-radius:14px; border:2px dashed #C7D3E6 !important; background:#FBFCFE; }}
.file-chip-row {{ display:flex; gap:0.7rem; flex-wrap:wrap; margin:0.8rem 0 0.2rem 0; }}
.file-chip {{ background:white; border:1px solid #EAEFF5; border-radius:12px; padding:0.6rem 1rem;
  box-shadow:0 2px 8px rgba(15,30,60,0.04); font-size:0.82rem; color:var(--muted); }}
.file-chip b {{ color:var(--navy); display:block; font-size:1rem; }}

[data-testid="stDataFrame"] {{ border-radius:14px; overflow:hidden; border:1px solid #EAEFF5;
  box-shadow:0 2px 10px rgba(15,30,60,0.04); }}
div[data-testid="stAlert"] {{ border-radius:12px; }}

/* Result banner (single transaction) */
.result-banner {{ border-radius:16px; padding:1.5rem 1.7rem; margin:0.8rem 0; box-shadow:0 6px 20px rgba(15,30,60,0.06);
  display:flex; align-items:center; gap:1rem; }}
.result-banner .r-icon {{ font-size:2.2rem; }}
.result-banner .r-title {{ font-weight:800; font-size:1.25rem; margin-bottom:0.15rem; }}
.result-banner .r-sub {{ font-size:0.9rem; font-weight:500; opacity:0.85; }}
.result-banner.fraud {{ background:var(--fraud-soft); color:var(--fraud); border:1px solid rgba(229,72,77,0.25); }}
.result-banner.legit {{ background:var(--legit-soft); color:var(--legit); border:1px solid rgba(31,169,122,0.25); }}

/* Filters panel */
.filter-panel {{ background:white; border:1px solid #EAEFF5; border-radius:14px; padding:1rem 1.3rem 0.3rem 1.3rem;
  margin-bottom:0.8rem; }}

/* Footer */
.app-footer {{ margin-top:2.5rem; padding:1.4rem 0 0.6rem 0; border-top:1px solid #E4E9F2;
  display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem; }}
.app-footer .f-left {{ color:var(--muted); font-size:0.82rem; }}
.app-footer .f-right {{ display:flex; gap:0.6rem; align-items:center; }}
.app-footer .f-badge {{ background:var(--accent-soft); color:var(--accent); font-size:0.72rem; font-weight:700;
  padding:0.25rem 0.6rem; border-radius:999px; }}
hr {{ margin:1.4rem 0; }}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, variant="", delay=0.0, badge=None, alert=False):
    classes = f"metric-card {variant}{' alert' if alert else ''}"
    badge_html = f'<span>{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="{classes}" style="animation-delay:{delay}s">'
        f'<div class="label"><span>{label}</span>{badge_html}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, title=None, legend=True):
    """Apply consistent, professional theming to any Plotly figure."""
    fig.update_layout(
        title=title or (fig.layout.title.text if fig.layout.title else None),
        font=dict(family="Inter, sans-serif", color=COLORS["navy"]),
        title_font=dict(size=16, family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=55, l=10, r=10, b=10),
        legend=dict(orientation="h", y=-0.18) if legend else dict(),
        showlegend=legend,
        hoverlabel=dict(bgcolor="white", font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(gridcolor="#EEF1F6", zerolinecolor="#EEF1F6")
    fig.update_yaxes(gridcolor="#EEF1F6", zerolinecolor="#EEF1F6")
    return fig


def make_gauge(value_pct, threshold_pct=50.0):
    """Fraud-probability gauge with green/amber/red risk zones and a
    threshold marker for the currently selected decision threshold."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"suffix": "%", "font": {"size": 34, "color": COLORS["navy"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["navy"], "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": COLORS["legit_soft"]},
                {"range": [40, 70], "color": COLORS["amber_soft"]},
                {"range": [70, 100], "color": COLORS["fraud_soft"]},
            ],
            "threshold": {
                "line": {"color": COLORS["fraud"], "width": 3},
                "thickness": 0.85,
                "value": threshold_pct,
            },
        },
    ))
    fig.update_layout(
        height=250, margin=dict(t=30, b=10, l=25, r=25),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=COLORS["navy"]),
    )
    return fig


def footer():
    st.markdown(f"""
    <div class="app-footer">
        <div class="f-left">💳 Financial Fraud Detection System · Built with Streamlit, scikit-learn &amp; Plotly ·
        For educational / demo purposes — not a certified compliance tool.</div>
        <div class="f-right">
            <span class="f-badge"><span class="live-dot"></span>&nbsp; Model: {selected_model_path.split('/')[-1]}</span>
            <span class="f-badge">v1.1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def score_dataframe(model, X, threshold=0.5):
    """Run predictions using predict_proba + an adjustable threshold when
    available, falling back to the model's default predict() otherwise."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
        return preds, proba.round(4)
    return model.predict(X), None


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)


def discover_models():
    found = sorted(BASE_DIR.glob("*.pkl"))
    return [str(f) for f in found] if found else [str(BASE_DIR / "model.pkl")]


def get_model(path):
    try:
        return load_model(path), None
    except FileNotFoundError:
        return None, f"{path} not found."
    except Exception as e:
        return None, f"Failed to load model: {e}"


# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("history", [])
st.session_state.setdefault("nav", "🏠 Home")


def go_to(target):
    st.session_state["nav"] = target


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style="text-align:center; padding:0.5rem 0 1.2rem 0;">
  <div style="font-size:2.2rem;">💳</div>
  <div style="font-size:1.15rem; font-weight:800; letter-spacing:-0.01em;">Fraud Detection</div>
  <div style="font-size:0.8rem; color:#B7C6E0; margin-top:0.2rem;">
    <span class="live-dot"></span>&nbsp; ML-Powered Transaction Analysis
  </div>
</div>
""", unsafe_allow_html=True)

NAV_OPTIONS = [
    "🏠 Home", "📊 Upload & Predict", "🔎 Single Transaction",
    "🕒 Prediction History", "🧠 Model Info", "ℹ️ About",
]
page = st.sidebar.radio("Navigate", NAV_OPTIONS, key="nav", label_visibility="collapsed")

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
# PAGE: HOME (Dashboard)
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge"><span class="live-dot"></span> Real-Time ML Scoring</div>
        <h1>💳 Financial Fraud Detection</h1>
        <p>An end-to-end system that analyzes transaction data and flags potentially
        fraudulent activity in real time — powered by a trained classification model
        and an interactive dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Session snapshot (animated KPIs) ----
    hist = st.session_state.history
    st.markdown('<span class="section-pill">Session Snapshot</span>', unsafe_allow_html=True)
    if hist:
        total_scored = sum(h["Total"] for h in hist)
        total_fraud = sum(h["Fraud"] for h in hist)
        runs = len(hist)
        last_run = hist[-1]["Timestamp"]
        overall_rate = (total_fraud / total_scored * 100) if total_scored else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Transactions Scored", f"{total_scored:,}", "accent", delay=0.0)
        with k2:
            metric_card("Fraud Flagged", f"{total_fraud:,}", "fraud", delay=0.08, alert=total_fraud > 0)
        with k3:
            metric_card("Batches Run", f"{runs:,}", "legit", delay=0.16)
        with k4:
            metric_card("Overall Fraud Rate", f"{overall_rate:.2f}%", "accent", delay=0.24, badge=f"since {last_run.split(' ')[0]}")
    else:
        st.info("No predictions run yet this session — scores from **Upload & Predict** will summarize here.")

    # ---- Navigation cards ----
    st.markdown('<span class="section-pill">Jump In</span>', unsafe_allow_html=True)
    nav_cards = [
        ("📊", "Upload & Predict", "Score a batch of transactions from CSV with adjustable threshold and full analytics.", "📊 Upload & Predict"),
        ("🔎", "Single Transaction", "Manually enter one transaction's features for an instant fraud check.", "🔎 Single Transaction"),
        ("🕒", "Prediction History", "Review batches you've already scored this session.", "🕒 Prediction History"),
        ("🧠", "Model Info", "See the algorithm and performance metrics behind the predictions.", "🧠 Model Info"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc, target) in zip(cols, nav_cards):
        with col:
            st.markdown(
                f'<div class="nav-card"><div class="nav-icon">{icon}</div>'
                f'<h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
            st.button("Open →", key=f"navcard_{target}", on_click=go_to, args=(target,), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-pill">How It Works</span>', unsafe_allow_html=True)
    steps = [
        ("📤", "Upload", "Upload a CSV of transactions and preview your data instantly."),
        ("🔍", "Predict", "Every transaction is scored with an adjustable fraud threshold."),
        ("📈", "Analyze", "Explore results, feature drivers, and evaluation metrics."),
        ("⬇️", "Export", "Download filtered, full, or fraud-only results."),
    ]
    for col, (icon, title, desc) in zip(st.columns(4), steps):
        col.markdown(f'<div class="feature-card"><div class="icon">{icon}</div>'
                      f'<h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

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

    st.markdown(
        '<div class="upload-instructions">Required columns: '
        + "".join(f'<span class="cols-chip">{c}</span>' for c in ["Time", "V1…V28", "Amount"])
        + '<div style="margin-top:0.5rem; color:var(--muted); font-size:0.85rem;">'
          'Optional label column (<code>Class</code>, <code>Label</code>, etc.) enables live '
          'evaluation metrics below.</div></div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}"); st.stop()

        size_kb = uploaded_file.size / 1024
        st.markdown(
            '<div class="file-chip-row">'
            f'<div class="file-chip"><b>{uploaded_file.name}</b>File name</div>'
            f'<div class="file-chip"><b>{len(df):,}</b>Rows</div>'
            f'<div class="file-chip"><b>{len(df.columns)}</b>Columns</div>'
            f'<div class="file-chip"><b>{size_kb:,.1f} KB</b>Size</div>'
            '</div>', unsafe_allow_html=True
        )

        with st.expander("Dataset Preview", expanded=False):
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

        kpi_defs = [
            ("Total Transactions", f"{total_count:,}", "accent", False),
            ("Fraud Transactions", f"{fraud_count:,}", "fraud", fraud_count > 0),
            ("Legitimate Transactions", f"{legit_count:,}", "legit", False),
            ("Fraud Rate", f"{fraud_rate:.2f}%", "fraud", fraud_rate > 5),
        ]
        for i, (col, (label, val, variant, alert)) in enumerate(zip(st.columns(4), kpi_defs)):
            with col:
                metric_card(label, val, variant, delay=i * 0.08, alert=alert)

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
                         color=["Legitimate", "Fraud"], color_discrete_map=FRAUD_MAP, hole=0.55)
            fig.update_traces(textinfo="percent+label", textfont_size=13)
            fig.add_annotation(text=f"<b>{fraud_rate:.1f}%</b><br><span style='font-size:11px'>fraud</span>",
                               showarrow=False, font=dict(size=18, color=COLORS["navy"]))
            st.plotly_chart(style_fig(fig, "Transaction Distribution"), use_container_width=True)
        with c2:
            fig = px.bar(x=["Legitimate", "Fraud"], y=[legit_count, fraud_count],
                         labels={"x": "Class", "y": "Count"}, color=["Legitimate", "Fraud"],
                         color_discrete_map=FRAUD_MAP, text=[legit_count, fraud_count])
            fig.update_traces(marker_line_width=0, textposition="outside")
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
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
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

        # ---- Interactive filters + results table ----
        st.markdown("#### Prediction Results")
        with st.expander("🔧 Filters", expanded=False):
            st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            pred_filter = f1.multiselect("Prediction", ["Legitimate", "Fraud"], default=["Legitimate", "Fraud"])
            amt_min, amt_max = float(df["Amount"].min()), float(df["Amount"].max())
            if amt_min == amt_max:
                amount_range = (amt_min, amt_max)
                f2.caption(f"Amount: {amt_min:,.2f} (constant)")
            else:
                amount_range = f2.slider("Amount range", amt_min, amt_max, (amt_min, amt_max))
            if proba is not None:
                prob_range = f3.slider("Fraud probability range", 0.0, 1.0, (0.0, 1.0), 0.01)
            else:
                prob_range = None
            st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = df[df["Prediction"].isin(pred_filter) & df["Amount"].between(*amount_range)]
        if prob_range is not None:
            filtered_df = filtered_df[filtered_df["Fraud_Probability"].between(*prob_range)]

        st.caption(f"Showing {len(filtered_df):,} of {total_count:,} transactions")
        st.dataframe(filtered_df, use_container_width=True)

        # ---- Downloads ----
        st.markdown("#### Download Results")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Full Results", df.to_csv(index=False),
                           file_name="predictions_full.csv", mime="text/csv", use_container_width=True)
        d2.download_button("⬇️ Filtered View", filtered_df.to_csv(index=False),
                           file_name="predictions_filtered.csv", mime="text/csv", use_container_width=True)
        d3.download_button("⬇️ Fraud Only", df[df["Prediction"] == "Fraud"].to_csv(index=False),
                           file_name="predictions_fraud_only.csv", mime="text/csv", use_container_width=True)

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

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        X_single = pd.DataFrame([{"Time": time_val, **v_values, "Amount": amount_val}])[REQUIRED_COLUMNS]
        with st.spinner("Scoring transaction..."):
            try:
                pred, proba = score_dataframe(model, X_single, threshold)
            except Exception as e:
                st.error(f"Prediction failed: {e}"); st.stop()

        is_fraud = pred[0] == 1
        label = "Fraud Detected" if is_fraud else "Legitimate Transaction"
        variant = "fraud" if is_fraud else "legit"
        icon = "🚨" if is_fraud else "✅"
        sub = "Recommend manual review or hold." if is_fraud else "No action needed at the current threshold."
        conf_text = f" · confidence {proba[0]:.2%}" if proba is not None else ""

        if proba is not None:
            r1, r2 = st.columns([2, 1])
        else:
            r1, r2 = st.container(), None

        with r1:
            st.markdown(
                f'<div class="result-banner {variant}"><div class="r-icon">{icon}</div>'
                f'<div><div class="r-title">{label}{conf_text}</div><div class="r-sub">{sub}</div></div></div>',
                unsafe_allow_html=True,
            )
        if proba is not None and r2 is not None:
            with r2:
                st.plotly_chart(make_gauge(proba[0] * 100, threshold * 100), use_container_width=True)

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

    st.markdown("#### Performance Metrics")
    perf_defs = [
        ("Accuracy", "accuracy", "accent"), ("Precision", "precision", "legit"),
        ("Recall", "recall", "legit"), ("F1-Score", "f1_score", "accent"),
    ]
    for i, (col, (label, key, variant)) in enumerate(zip(st.columns(4), perf_defs)):
        with col:
            metric_card(label, f"{MODEL_INFO[key] * 100:.2f}%", variant, delay=i * 0.08)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = px.bar(x=["Accuracy", "Precision", "Recall", "F1-Score"],
                y=[MODEL_INFO[k] for k in ["accuracy", "precision", "recall", "f1_score"]],
                labels={"x": "Metric", "y": "Score"}, range_y=[0, 1],
                color=["Accuracy", "Precision", "Recall", "F1-Score"],
                color_discrete_sequence=[COLORS["accent"], COLORS["legit"], COLORS["legit"], COLORS["accent"]],
                text=[f"{MODEL_INFO[k]*100:.1f}%" for k in ["accuracy", "precision", "recall", "f1_score"]])
    fig.update_traces(marker_line_width=0, textposition="outside")
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

# ============================================================
# FOOTER (all pages)
# ============================================================
footer()