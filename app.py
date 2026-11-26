"""FraudShield — Streamlit interface for the existing fraud classifier."""
from datetime import datetime
from pathlib import Path
from html import escape
import hashlib
import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                                 precision_recall_curve, average_precision_score,
                                 classification_report)
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_INFO = {
    "algorithm": "Random Forest", "accuracy": 0.999507,
    "precision": 0.971831, "recall": 0.726316, "f1_score": 0.831325,
    "dataset": "Credit Card Dataset (Kaggle)", "training_samples": "N/A",
}
REQUIRED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL_COLUMN_CANDIDATES = ["Class", "class", "Label", "label", "Actual", "actual", "is_fraud", "isFraud"]
C = {"ink": "#17263C", "muted": "#53647B", "blue": "#2458D3", "red": "#B42335", "green": "#087F65", "line": "#E1E7F0"}
FRAUD_MAP = {"Legitimate": C["green"], "Fraud": C["red"]}
NAV = ["Overview", "Batch analysis", "Single transaction", "Prediction history", "Model information", "About"]

st.set_page_config(page_title="FraudShield | Financial Fraud Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#17263C; --muted:#53647B; --blue:#2458D3; --line:#E1E7F0; }
.stApp { background:#F5F7FB; }
[data-testid="stHeader"] {background:rgba(245,247,251,.96);}
[data-testid="stMainBlockContainer"], .main .block-container {max-width:1440px; padding:2.1rem 2.6rem 2rem;}
[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid var(--line);}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top:1rem;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:4px;}
[data-testid="stSidebar"] [role="radiogroup"] > label {padding:.65rem .75rem; border-radius:8px; margin:0; width:100%;}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {background:#F1F5FC;}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {background:#EAF0FF;}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {color:#1947B3; font-weight:650;}
.brand {display:flex; align-items:center; gap:12px; padding:5px 0 25px;}
.brand-mark {display:grid; place-items:center; background:#2458D3; color:white; width:42px; height:42px; border-radius:11px; flex:none; font-size:17px; font-weight:800; letter-spacing:-1px;}
.brand-name {font-weight:800; font-size:19px; color:var(--ink); letter-spacing:-.6px;}
.brand-sub {font-size:11px; color:var(--muted); letter-spacing:.5px;}
.nav-label,.eyebrow {font-size:11px; font-weight:750; letter-spacing:1.35px; text-transform:uppercase; color:#53647B;}
.nav-label {padding:0 0 6px;}
.page-head {margin-bottom:1.1rem;}
.page-head h1 {font-size:clamp(1.8rem,3vw,2.3rem); font-weight:750; letter-spacing:-1.1px; color:var(--ink); padding:.35rem 0 .25rem; margin:0; line-height:1.18;}
.page-head p {margin:0; font-size:.98rem; color:var(--muted); max-width:800px; line-height:1.55;}
.hero {background:#142B50; border:1px solid #1D365D; border-radius:16px; padding:2rem 2.3rem; position:relative; overflow:hidden; margin-bottom:.5rem;}
.hero:after {content:""; position:absolute; width:270px; height:270px; border:1px solid #3B5374; border-radius:50%; right:-54px; top:-105px; box-shadow:0 0 0 34px #1C3459,0 0 0 35px #3B5374,0 0 0 68px #1C3459; pointer-events:none;}
.hero > * {position:relative; z-index:1;}
.hero .eyebrow {color:#B7CBF6;}
.hero h1 {color:#FFFFFF; font-weight:750; font-size:clamp(1.9rem,3.4vw,2.75rem); letter-spacing:-1.2px; line-height:1.15; margin:.7rem 0; padding:0; max-width:700px;}
.hero p {color:#D5E1F5; margin:0; max-width:620px; font-size:1rem; line-height:1.65;}
.kpi {background:white; border:1px solid var(--line); border-radius:12px; padding:1.1rem 1.15rem; min-height:122px; box-shadow:0 2px 4px #17263C05;}
.kpi-label {font-size:.8rem; font-weight:650; color:var(--muted); line-height:1.3;}
.kpi-value {font-weight:780; font-size:clamp(1.55rem,2.5vw,2rem); letter-spacing:-.9px; margin:.5rem 0 .25rem; color:var(--ink); font-variant-numeric:tabular-nums; overflow-wrap:anywhere;}
.kpi-note {font-size:.74rem; color:var(--muted); line-height:1.4;}
.kpi.red .kpi-value {color:#B42335;} .kpi.green .kpi-value {color:#087F65;} .kpi.blue .kpi-value {color:#2458D3;}
.section-head {font-size:1.12rem; font-weight:750; color:var(--ink); letter-spacing:-.25px; margin:1.1rem 0 .15rem;}
.section-sub {font-size:.85rem; color:var(--muted); margin:0 0 .7rem;}
.module {min-height:114px;}
.module-index {color:#2458D3; font-size:11px; font-weight:750; letter-spacing:1px; margin-bottom:.6rem;}
.module-title {font-size:1.05rem; color:var(--ink); font-weight:750; margin:0 0 .35rem;}
.module-desc {font-size:.86rem; line-height:1.55; color:var(--muted);}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"] {border:1px solid var(--line); border-radius:12px; background:#FFFFFF; box-shadow:0 2px 4px #17263C05;}
/* Native controls retain the Streamlit theme, focus rings, and accessibility. */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"], [data-testid="stFormSubmitButton"] button {border-radius:8px; min-height:42px; font-weight:650; transition:background-color .15s, border-color .15s;}
[data-testid="stBaseButton-primary"] {background:#2458D3; border-color:#2458D3; color:white;}
[data-testid="stBaseButton-primary"]:hover {background:#1946B4; border-color:#1946B4; color:white;}
[data-testid="stBaseButton-secondary"] {background:white; border-color:#CCD7E6; color:#17263C;}
[data-testid="stBaseButton-secondary"]:hover {background:#EAF0FF; color:#1947B3; border-color:#2458D3;}
[data-testid="stFileUploaderDropzone"] {border:1.5px dashed #B9C9E1; background:#F7FAFF; border-radius:10px; padding:1.15rem;}
[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:10px; overflow:hidden;}
[data-testid="stExpander"] details {border-color:var(--line); border-radius:10px; background:white;}
[data-testid="stTabs"] [role="tablist"] {gap:1.2rem;}
[data-testid="stTabs"] [role="tab"] {padding:.65rem 0; font-weight:650;}
[data-testid="stAlert"] {border-radius:9px;}
.result {border:1px solid #C5E7DC; border-left:4px solid #087F65; background:#F0FAF6; border-radius:11px; padding:1.25rem;}
.result.red {border-color:#F2C9D0; border-left-color:#B42335; background:#FFF4F5;}
.result-title {font-size:1.25rem; font-weight:750; color:#087F65; margin:.25rem 0;}
.result.red .result-title {color:#B42335;}
.result-note {font-size:.88rem; line-height:1.55; color:#415168;}
.footer {border-top:1px solid var(--line); margin-top:2.4rem; padding:1rem 0; display:flex; justify-content:space-between; flex-wrap:wrap; gap:.65rem; color:var(--muted); font-size:.75rem;}
@media(max-width:768px) {
 [data-testid="stMainBlockContainer"], .main .block-container {padding:1.2rem 1rem 1.5rem;}
 .hero {padding:1.5rem;} .hero:after {opacity:.35;}
 .kpi {min-height:104px; padding:.85rem;} .module {min-height:0;}
 [data-testid="stTabs"] [role="tablist"] {gap:.65rem;}
}
@media(prefers-reduced-motion:reduce) { * {transition:none !important;} }
</style>
""", unsafe_allow_html=True)


def safe(value):
    return escape(str(value), quote=True)


def page_head(title, description, category="WORKSPACE"):
    st.markdown(f'<div class="page-head"><div class="eyebrow">{safe(category)}</div><h1>{safe(title)}</h1><p>{safe(description)}</p></div>', unsafe_allow_html=True)


def section(title, description=""):
    st.markdown(f'<div class="section-head">{safe(title)}</div>' + (f'<div class="section-sub">{safe(description)}</div>' if description else ""), unsafe_allow_html=True)


def kpi(label, value, note="", tone=""):
    st.markdown(f'<div class="kpi {safe(tone)}"><div class="kpi-label">{safe(label)}</div><div class="kpi-value">{safe(value)}</div><div class="kpi-note">{safe(note)}</div></div>', unsafe_allow_html=True)


def kpi_row(items):
    for col, item in zip(st.columns(len(items)), items):
        with col:
            kpi(*item)


def chart(fig, title=None, legend=True, height=330):
    fig.update_layout(template="plotly_white", height=height, title=None,
                      font=dict(family="Arial, sans-serif", size=12, color=C["ink"]),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      margin=dict(l=12, r=18, t=18, b=12),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
                      showlegend=legend, hoverlabel=dict(bgcolor="white", font_color=C["ink"]))
    fig.update_xaxes(gridcolor="#EDF1F6", zerolinecolor="#DCE3EE", automargin=True)
    fig.update_yaxes(gridcolor="#EDF1F6", zerolinecolor="#DCE3EE", automargin=True)
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        st.plotly_chart(fig, use_container_width=True, theme=None,
                        config={"displaylogo": False, "displayModeBar": False})


@st.cache_resource
def load_model(path, modified_ns):
    # Only load trusted local model artifacts; pickle/joblib is executable.
    return joblib.load(path)


def discover_models():
    return sorted(str(p) for p in BASE_DIR.glob("*.pkl"))


def get_model(path):
    if not path:
        return None, "No .pkl model found. Place your trained model in the same directory as app.py."
    try:
        return load_model(path, Path(path).stat().st_mtime_ns), None
    except Exception as exc:
        return None, f"Could not load {Path(path).name}: {exc}"


def score_dataframe(model, X, threshold=0.5):
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        idx = np.flatnonzero(classes == 1)
        if set(classes.tolist()) != {0, 1} or len(idx) != 1:
            raise ValueError("The model must use binary classes 0 (legitimate) and 1 (fraud).")
        proba = probabilities[:, idx[0]]
        if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
            raise ValueError("The model returned invalid probabilities.")
        return (proba >= threshold).astype(int), proba
    preds = np.asarray(model.predict(X))
    if not np.isin(preds, [0, 1]).all():
        raise ValueError("The model must predict 0 or 1.")
    return preds.astype(int), None


def clean_transactions(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    X = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    cleaned = df.loc[valid].copy().reset_index(drop=True)
    cleaned[REQUIRED_COLUMNS] = X.loc[valid].reset_index(drop=True)
    return cleaned, int((~valid).sum())


def feature_importance(model):
    values = getattr(model, "feature_importances_", None)
    if values is None or len(values) != len(REQUIRED_COLUMNS):
        return None
    return pd.Series(values, index=REQUIRED_COLUMNS).nlargest(10).sort_values()


def show_importance(model):
    imp = feature_importance(model)
    if imp is None:
        st.info("Global feature importance is not available for this model.")
        return
    st.caption("Global model importance, not the explanation for an individual prediction.")
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": "Feature"})
    fig.update_traces(marker_color=C["blue"])
    chart(fig, "Top 10 influential features", legend=False, height=370)


def navigate(target):
    st.session_state.nav = target


def threshold_control(model, key):
    if not hasattr(model, "predict_proba"):
        st.info("This model uses its default decision rule; probability scores are unavailable.")
        return 0.5
    return st.slider("Fraud decision threshold", 0.0, 1.0, 0.5, 0.01, key=key, format="%.2f",
                     help="Lower values may catch more fraud but increase false positives. Higher values may miss more fraud.")


def file_name(path):
    return Path(path).name if path else "No model selected"


st.session_state.setdefault("nav", NAV[0])
st.session_state.setdefault("history", [])
st.session_state.setdefault("batch_result", None)
st.session_state.setdefault("single_result", None)
st.session_state.setdefault("run_number", 0)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">FS</div><div><div class="brand-name">FraudShield</div><div class="brand-sub">TRANSACTION INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", NAV, key="nav", label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="nav-label">Model configuration</div>', unsafe_allow_html=True)
    models = discover_models()
    if models:
        selected_model_path = st.selectbox("Active model", models, format_func=file_name)
        st.caption("Local model file found. Loading is checked when scoring.")
    else:
        selected_model_path = None
        st.warning("No model file found")
    st.divider()
    st.caption("Educational ML project · v2.0")


if page == "Overview":
    st.markdown('<div class="hero"><div class="eyebrow">Financial risk analytics</div><h1>Financial fraud detection</h1><p>Analyze transaction data, identify potentially fraudulent activity, and investigate model predictions from a single workspace.</p></div>', unsafe_allow_html=True)
    a, b, _ = st.columns([1, 1, 1.5])
    a.button("Analyze a CSV", type="primary", use_container_width=True, on_click=navigate, args=("Batch analysis",))
    b.button("Score a transaction", use_container_width=True, on_click=navigate, args=("Single transaction",))
    section("Session overview", "Batch activity in this browser session. Counts reset with session history.")
    hist = st.session_state.history
    total = sum(h["Total"] for h in hist)
    fraud = sum(h["Fraud"] for h in hist)
    kpi_row([("Transactions scored", f"{total:,}", "Across completed batches", "blue"),
             ("Flagged for review", f"{fraud:,}", "Predicted fraud", "red"),
             ("Not flagged", f"{total-fraud:,}", "Predicted legitimate", "green"),
             ("Flag rate", f"{fraud/total:.2%}" if total else "—", f"{len(hist)} completed batches", "")])
    section("Explore the workspace", "Choose a tool to start your analysis.")
    modules = [("01 / ANALYZE", "Batch analysis", "Upload a CSV, choose a decision threshold and investigate flagged transactions.", "Batch analysis"),
               ("02 / INVESTIGATE", "Single transaction", "Enter transaction features and inspect its predicted risk and available explanations.", "Single transaction"),
               ("03 / REVIEW", "Prediction history", "Review previous batch runs, compare fraud counts and export the session log.", "Prediction history"),
               ("04 / UNDERSTAND", "Model information", "Inspect reference performance metrics and the active model's details.", "Model information")]
    for start in (0, 2):
        for col, (idx, title, desc, target) in zip(st.columns(2), modules[start:start+2]):
            with col, st.container(border=True):
                st.markdown(f'<div class="module"><div class="module-index">{safe(idx)}</div><div class="module-title">{safe(title)}</div><div class="module-desc">{safe(desc)}</div></div>', unsafe_allow_html=True)
                st.button(f"Open {title.lower()}", key=f"open_{target}", use_container_width=True, on_click=navigate, args=(target,))
    section("How it works")
    st.markdown("**01 Upload**  ·  Provide transaction features    →    **02 Score**  ·  Run the classifier    →    **03 Review**  ·  Inspect flagged records    →    **04 Export**  ·  Download results")

elif page == "Batch analysis":
    page_head("Batch transaction analysis", "Upload a transaction dataset, configure your decision threshold, and review the results.", "Transaction monitoring")
    model, error = get_model(selected_model_path)
    if error:
        st.error(error)
        st.stop()
    with st.container(border=True):
        left, right = st.columns([1.6, 1])
        with left:
            st.markdown("**01 · Upload dataset**")
            uploaded = st.file_uploader("Transaction CSV", type=["csv"], help="Required: Time, V1–V28, Amount. Optional: Class or another recognized binary label column.")
            st.caption("Required: Time, V1–V28, Amount. Optional: Class (0 or 1) for evaluation.")
            template = pd.DataFrame(columns=REQUIRED_COLUMNS + ["Class"]).to_csv(index=False)
            st.download_button("Download CSV template", template, "transaction_template.csv", "text/csv")
        with right:
            st.markdown("**02 · Configure scoring**")
            threshold = threshold_control(model, "batch_threshold")
            st.caption("The selected threshold is applied only when you run analysis.")
            run = st.button("Run analysis", type="primary", use_container_width=True, disabled=uploaded is None)
    if uploaded is None:
        st.info("Upload a compatible CSV to begin. The template contains column headers only.")
    else:
        raw = uploaded.getvalue()
        signature = (selected_model_path, Path(selected_model_path).stat().st_mtime_ns, hashlib.sha256(raw).hexdigest(), uploaded.name, threshold)
        try:
            original = pd.read_csv(io.BytesIO(raw))
            clean, dropped = clean_transactions(original)
            if clean.empty:
                raise ValueError("No valid transaction rows remain.")
        except Exception as exc:
            st.error(f"Dataset could not be validated: {exc}")
            st.stop()
        with st.expander(f"Dataset preview · {len(original):,} rows · {len(original.columns)} columns · {uploaded.size/1024:,.1f} KB"):
            st.dataframe(original.head(10), use_container_width=True, hide_index=True)
        if dropped:
            st.warning(f"{dropped:,} rows contain missing, nonnumeric or infinite feature values and will be excluded.")
        if run:
            try:
                with st.spinner("Analyzing transactions…"):
                    df = clean.copy()
                    label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in df), None)
                    preds, proba = score_dataframe(model, df[REQUIRED_COLUMNS], threshold)
                    df["Prediction"] = np.where(preds == 1, "Fraud", "Legitimate")
                    # Remove stale scores if an input CSV is itself a prior export.
                    if proba is not None:
                        df["Fraud_Probability"] = proba
                    elif "Fraud_Probability" in df:
                        df = df.drop(columns="Fraud_Probability")
                    flagged = int((preds == 1).sum())
                    st.session_state.run_number += 1
                    st.session_state.batch_result = {"signature": signature, "df": df, "label": label_col, "id": st.session_state.run_number}
                    st.session_state.history.append({"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "File": uploaded.name,
                        "Model": file_name(selected_model_path), "Threshold": threshold if proba is not None else None,
                        "Total": len(df), "Fraud": flagged, "Legitimate": len(df)-flagged, "Fraud Rate": f"{flagged/len(df):.2%}"})
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()
        result = st.session_state.batch_result
        if not result or result["signature"] != signature:
            st.info("Dataset is ready. Select **Run analysis** to generate results for the current settings.")
        else:
            df = result["df"]
            label_col = result["label"]
            key = str(result["id"])
            has_proba = "Fraud_Probability" in df
            total = len(df)
            fraud = int(df["Prediction"].eq("Fraud").sum())
            legit = total - fraud
            section("Analysis results", f"{file_name(selected_model_path)} · Threshold {threshold:.0%}" if has_proba else f"{file_name(selected_model_path)} · Default decision rule")
            kpi_row([("Transactions analyzed", f"{total:,}", "Valid rows", "blue"), ("Flagged for review", f"{fraud:,}", "Predicted fraud", "red"),
                     ("Not flagged", f"{legit:,}", "Predicted legitimate", "green"), ("Flag rate", f"{fraud/total:.2%}", "Not confirmed fraud", "")])
            if fraud:
                st.warning(f"{fraud:,} transactions met the model's fraud decision rule. Review these records before taking action.")
            else:
                st.success("No transactions were flagged at the current settings. This does not guarantee that all transactions are safe.")
            overview, transactions, evaluation, explain = st.tabs(["Overview", "Transactions & export", "Model evaluation", "Feature importance"])
            with overview:
                if has_proba and fraud:
                    st.caption(f"Average predicted fraud probability among flagged rows: {df.loc[df.Prediction.eq('Fraud'), 'Fraud_Probability'].mean():.2%}")
                l, r = st.columns(2)
                with l:
                    fig = px.pie(names=["Legitimate", "Fraud"], values=[legit, fraud], color=["Legitimate", "Fraud"], color_discrete_map=FRAUD_MAP, hole=.72)
                    fig.update_traces(textinfo="none", sort=False, marker=dict(line=dict(color="white", width=2)))
                    fig.add_annotation(text=f"<b>{fraud/total:.1%}</b><br>flagged", showarrow=False, font=dict(size=17, color=C["ink"]))
                    chart(fig, "Transaction distribution")
                with r:
                    fig = px.bar(x=["Legitimate", "Fraud"], y=[legit, fraud], color=["Legitimate", "Fraud"], color_discrete_map=FRAUD_MAP, labels={"x":"Prediction", "y":"Transactions"}, text=[legit, fraud])
                    fig.update_traces(textposition="auto", marker_line_width=0)
                    chart(fig, "Transaction counts", legend=False)
                l, r = st.columns(2)
                with l:
                    fig = px.histogram(df, x="Amount", color="Prediction", nbins=40, barmode="overlay", opacity=.7, color_discrete_map=FRAUD_MAP)
                    chart(fig, "Transaction amount distribution")
                with r:
                    # A deterministic sample keeps interactive plots responsive on large uploads.
                    plot_df = df if total <= 12000 else df.sample(12000, random_state=42)
                    fig = px.scatter(plot_df, x="Time", y="Amount", color="Prediction", opacity=.65, color_discrete_map=FRAUD_MAP)
                    chart(fig, "Transactions over time")
                    if len(plot_df) < total:
                        st.caption(f"Scatter plot shows {len(plot_df):,} sampled rows; metrics and exports use all rows.")
            with transactions:
                with st.container(border=True):
                    st.markdown("**Filter transactions**")
                    f1, f2, f3 = st.columns(3)
                    classes = f1.multiselect("Prediction", ["Legitimate", "Fraud"], default=["Legitimate", "Fraud"], key="classes_"+key)
                    lo, hi = float(df.Amount.min()), float(df.Amount.max())
                    amounts = f2.slider("Amount range", lo, hi, (lo, hi), key="amounts_"+key) if lo < hi else (lo, hi)
                    if lo == hi:
                        f2.caption(f"Fixed amount: {lo:,.2f}")
                    probs = f3.slider("Fraud probability", 0.0, 1.0, (0.0, 1.0), .01, key="probs_"+key) if has_proba else None
                    filtered = df[df.Prediction.isin(classes) & df.Amount.between(*amounts)]
                    if probs is not None:
                        filtered = filtered[filtered.Fraud_Probability.between(*probs)]
                st.caption(f"Showing {len(filtered):,} of {total:,} transactions")
                first = ["Prediction"] + (["Fraud_Probability"] if has_proba else []) + ["Amount", "Time"]
                ordered = first + [c for c in df.columns if c not in first]
                config = {"Fraud_Probability": st.column_config.ProgressColumn("Fraud probability", min_value=0, max_value=1, format="%.4f"), "Amount": st.column_config.NumberColumn("Amount", format="%.2f")}
                st.dataframe(filtered[ordered], use_container_width=True, hide_index=True, height=420, column_config=config)
                section("Export results")
                for col, text, data, filename in zip(st.columns(3), ["All transactions", "Filtered transactions", "Flagged transactions"], [df, filtered, df[df.Prediction.eq("Fraud")]], ["predictions_full.csv", "predictions_filtered.csv", "predictions_fraud_only.csv"]):
                    col.download_button(text, data.to_csv(index=False), filename, "text/csv", use_container_width=True)
            with evaluation:
                if not SKLEARN_METRICS_AVAILABLE:
                    st.info("Install scikit-learn to view evaluation metrics.")
                elif label_col is None:
                    st.info("Include a recognized ground-truth column such as Class (0 = legitimate, 1 = fraud) to evaluate predictions.")
                else:
                    actual = pd.to_numeric(df[label_col], errors="coerce")
                    mask = actual.isin([0, 1])
                    if not mask.any():
                        st.warning("No valid binary ground-truth labels were found.")
                    else:
                        if not mask.all():
                            st.warning(f"Excluded {int((~mask).sum()):,} invalid or missing labels from evaluation only.")
                        y_true = actual[mask].astype(int)
                        y_pred = df.loc[mask, "Prediction"].eq("Fraud").astype(int)
                        report = classification_report(y_true, y_pred, labels=[0,1], target_names=["Legitimate","Fraud"], output_dict=True, zero_division=0)
                        metrics = report["Fraud"]
                        kpi_row([("Accuracy", f"{(y_true == y_pred).mean():.2%}", "All labeled rows", "blue"), ("Precision", f"{metrics['precision']:.2%}", "Fraud class", ""), ("Recall", f"{metrics['recall']:.2%}", "Fraud class", ""), ("F1-score", f"{metrics['f1-score']:.2%}", "Fraud class", "")])
                        l, r = st.columns(2)
                        with l:
                            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                            fig = go.Figure(go.Heatmap(z=cm, x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"], colorscale="Blues", showscale=False, hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>"))
                            for i, row in enumerate(cm):
                                for j, value in enumerate(row):
                                    fig.add_annotation(x=["Legitimate", "Fraud"][j], y=["Legitimate", "Fraud"][i], text=f"{value:,}", showarrow=False, font=dict(color="white" if value > cm.max()/2 else C["ink"], size=17))
                            fig.update_layout(xaxis_title="Predicted class", yaxis_title="Actual class", yaxis_autorange="reversed")
                            chart(fig, "Confusion matrix", legend=False)
                        if has_proba and y_true.nunique() == 2:
                            scores = df.loc[mask, "Fraud_Probability"]
                            fpr, tpr, _ = roc_curve(y_true, scores)
                            with r:
                                fig = go.Figure([go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC {auc(fpr,tpr):.3f}", line=dict(color=C["blue"], width=3)), go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(color="#A5B3C8", dash="dash"))])
                                fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
                                chart(fig, "ROC curve")
                            precision, recall, _ = precision_recall_curve(y_true, scores)
                            fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line=dict(color=C["blue"], width=3), name=f"AP {average_precision_score(y_true,scores):.3f}"))
                            fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
                            chart(fig, "Precision–recall curve")
                        else:
                            r.info("ROC and precision–recall curves require probability scores and both ground-truth classes.")
                        with st.expander("Full classification report"):
                            st.dataframe(pd.DataFrame({k:report[k] for k in ["Legitimate", "Fraud", "macro avg", "weighted avg"]}).T, use_container_width=True)
            with explain:
                show_importance(model)

elif page == "Single transaction":
    page_head("Single transaction scoring", "Enter the exact feature values expected by the trained model to inspect an individual transaction.", "Risk investigation")
    model, error = get_model(selected_model_path)
    if error:
        st.error(error)
        st.stop()
    with st.container(border=True):
        threshold = threshold_control(model, "single_threshold")
    with st.form("single_transaction_form"):
        st.markdown("**01 · Transaction details**")
        l, r = st.columns(2)
        time = l.number_input("Time", value=0.0, step=1.0, help="Use the same time representation as your training dataset.")
        amount = r.number_input("Amount", min_value=0.0, value=0.0, step=1.0)
        st.markdown("**02 · PCA features**")
        st.caption("V1–V28 must be the PCA-transformed values from a compatible dataset, not raw card details.")
        values = {}
        for start in range(1, 29, 4):
            for col, i in zip(st.columns(4), range(start, min(start+4, 29))):
                values[f"V{i}"] = col.number_input(f"V{i}", value=0.0, step=.1, format="%.6f", key=f"v_{i}")
        submitted = st.form_submit_button("Score transaction", type="primary", use_container_width=True)
    signature = (selected_model_path, Path(selected_model_path).stat().st_mtime_ns, threshold)
    if submitted:
        try:
            X = pd.DataFrame([{"Time":time, **values, "Amount":amount}])[REQUIRED_COLUMNS]
            with st.spinner("Scoring transaction…"):
                pred, proba = score_dataframe(model, X, threshold)
            st.session_state.single_result = {"signature":signature, "X":X, "flag":bool(pred[0] == 1), "prob":float(proba[0]) if proba is not None else None}
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()
    result = st.session_state.single_result
    if result and result["signature"] == signature:
        section("Scoring result", "Based on the last submitted transaction and the current decision rule.")
        flagged, prob = result["flag"], result["prob"]
        title = "Flagged for manual review" if flagged else "Not flagged by the model"
        note = "The model predicts potential fraud. Investigate before taking action." if flagged else "The model predicts a legitimate transaction at these settings. This is not a guarantee of safety."
        label = "red" if flagged else ""
        st.markdown(f'<div class="result {label}"><div class="eyebrow">Model decision</div><div class="result-title">{safe(title)}</div><div class="result-note">{safe(note)}</div></div>', unsafe_allow_html=True)
        if prob is not None:
            l, r = st.columns([1, 2])
            with l:
                kpi("Predicted fraud probability", f"{prob:.2%}", "Not a calibrated guarantee of risk", "red" if flagged else "green")
                kpi("Decision threshold", f"{threshold:.0%}", "Flag when score meets or exceeds threshold", "blue")
            with r:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100,
                    number=dict(suffix="%", font=dict(size=34, color=C["ink"])),
                    gauge=dict(axis=dict(range=[0,100]), bar=dict(color=C["red"] if flagged else C["green"], thickness=.35), bgcolor="white", borderwidth=0,
                               steps=[dict(range=[0,threshold*100], color="#E4F3EE"),dict(range=[threshold*100,100],color="#FCE9EC")],
                               threshold=dict(line=dict(color=C["ink"],width=3),thickness=.85,value=threshold*100))))
                chart(fig, "Fraud score vs. threshold", legend=False, height=235)
        with st.expander("Explain this prediction", expanded=False):
            # SHAP is optional; import only when a user explicitly requests the explanation.
            explain_now = st.button("Generate feature explanation", key="generate_shap")
            if explain_now:
                try:
                    import shap
                    with st.spinner("Generating explanation…"):
                        sv = shap.TreeExplainer(model).shap_values(result["X"])
                        classes = np.asarray(getattr(model,"classes_",[0,1]))
                        fraud_idx = int(np.flatnonzero(classes == 1)[0])
                        if isinstance(sv,list):
                            vector = np.asarray(sv[fraud_idx])[0]
                        else:
                            arr = np.asarray(sv)
                            if arr.ndim == 3:
                                vector = arr[0,:,fraud_idx]
                            elif arr.ndim == 2:
                                vector = arr[0]
                            else:
                                raise ValueError("Unsupported explanation shape")
                        vector = np.ravel(vector)
                        if len(vector) != len(REQUIRED_COLUMNS):
                            raise ValueError("Explanation does not match model features")
                        impact = pd.Series(vector,index=REQUIRED_COLUMNS)
                        impact = impact.loc[impact.abs().nlargest(10).index].sort_values()
                    st.caption("Positive SHAP values increase the explained model output; negative values decrease it. Units depend on the model explainer.")
                    fig = go.Figure(go.Bar(x=impact.values,y=impact.index,orientation="h",marker_color=[C["red"] if v>0 else C["green"] for v in impact]))
                    fig.update_layout(xaxis_title="SHAP contribution",yaxis_title="Feature")
                    chart(fig,"Top transaction-specific contributions",legend=False,height=370)
                except Exception:
                    st.info("A transaction-specific SHAP explanation is unavailable. Showing global importance if supported.")
                    show_importance(model)
    elif result:
        st.info("Model settings have changed. Submit the transaction again to update its result.")

elif page == "Prediction history":
    page_head("Prediction history", "Review and export completed batch analyses from your current session.", "Activity log")
    hist = st.session_state.history
    if not hist:
        with st.container(border=True):
            st.markdown("**No completed batch analyses**")
            st.caption("Your activity will appear here after you run your first batch.")
            st.button("Start batch analysis", type="primary", on_click=navigate, args=("Batch analysis",))
    else:
        df = pd.DataFrame(hist)
        total, flagged = int(df.Total.sum()), int(df.Fraud.sum())
        kpi_row([("Completed batches", f"{len(df):,}", "This session", "blue"), ("Transactions scored", f"{total:,}", "Across all runs", ""), ("Flagged for review", f"{flagged:,}", "Predicted fraud", "red")])
        section("Run activity")
        fig = px.line(df.assign(Run=range(1,len(df)+1)), x="Run", y="Fraud", markers=True, hover_data=["Timestamp","File"], labels={"Run":"Batch run", "Fraud":"Flagged transactions"})
        fig.update_traces(line_color=C["blue"], line_width=3, marker_size=8)
        chart(fig, "Flagged transactions by batch", legend=False)
        section("Session log")
        st.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True)
        l,r = st.columns([1,1])
        l.download_button("Export session history", df.to_csv(index=False), "prediction_history.csv", "text/csv", use_container_width=True)
        with r:
            with st.expander("Clear session history"):
                st.caption("This removes the activity log and current batch result from this session.")
                if st.button("Confirm clear history", use_container_width=True):
                    st.session_state.history = []
                    st.session_state.batch_result = None
                    st.rerun()

elif page == "Model information":
    page_head("Model information", "Explore the classifier, reference evaluation metrics and feature importance.", "Model intelligence")
    model, error = get_model(selected_model_path)
    l,r = st.columns(2)
    with l, st.container(border=True):
        st.markdown("**Active model**")
        st.write("Model file:", file_name(selected_model_path))
        if error:
            st.warning(error)
        else:
            st.write("Estimator:", type(model).__name__)
            st.write("Probability scoring:", "Available" if hasattr(model,"predict_proba") else "Unavailable")
    with r, st.container(border=True):
        st.markdown("**Reference training information**")
        st.write("Algorithm:", MODEL_INFO["algorithm"])
        st.write("Dataset:", MODEL_INFO["dataset"])
        st.write("Training samples:", MODEL_INFO["training_samples"])
    section("Reference performance", "Values supplied in the original source code; not independently verified or linked to the selected model file.")
    perf = [("Accuracy","accuracy","All correct predictions"),("Precision","precision","Correct fraud flags"),("Recall","recall","Actual fraud identified"),("F1-score","f1_score","Precision–recall balance")]
    kpi_row([(label,f"{MODEL_INFO[key]:.2%}",desc,"blue" if key=="accuracy" else "") for label,key,desc in perf])
    fig = px.bar(x=[x[0] for x in perf], y=[MODEL_INFO[x[1]] for x in perf], text=[f"{MODEL_INFO[x[1]]:.2%}" for x in perf], labels={"x":"Metric","y":"Score"})
    fig.update_traces(marker_color=C["blue"],textposition="outside",cliponaxis=False)
    fig.update_yaxes(range=[0,1.12],tickformat=".0%")
    chart(fig,"Reference performance comparison",legend=False)
    st.info("Accuracy can be misleading on imbalanced fraud datasets. Use labeled, representative test data to evaluate precision, recall and F1-score. Update MODEL_INFO only with verified evaluation results.")
    if model is not None:
        section("Feature importance")
        show_importance(model)

elif page == "About":
    page_head("About FraudShield", "A machine learning project for financial transaction classification and interactive risk analysis.", "Project documentation")
    with st.container(border=True):
        st.markdown("""### Project overview
The application classifies transaction features as legitimate or potentially fraudulent using a trained binary classifier. It supports probability-based decision thresholds when the model exposes them.

### Machine learning workflow
Data collection → preprocessing and class-imbalance handling → feature engineering → model training → evaluation → interactive deployment.

### Dataset compatibility
The application expects **Time**, **V1–V28**, and **Amount**, following the Kaggle Credit Card Fraud dataset schema. Optional binary labels enable live evaluation. Models trained on other schemas, such as PaySim, need corresponding input changes and preprocessing.

### Technology stack
Python · Streamlit · pandas · NumPy · scikit-learn · Plotly · joblib · SHAP (optional).

### Important limitations
Model outputs are predictions, not confirmed fraud decisions. This educational application is not a certified fraud prevention, compliance, or payment-blocking system. Only load trusted local model files and use authorized transaction datasets.
""")
    st.button("Return to overview", on_click=navigate, args=("Overview",))

st.markdown(f'<div class="footer"><span>FraudShield · Educational financial risk analytics</span><span>Model: {safe(file_name(selected_model_path))} · v2.0</span></div>', unsafe_allow_html=True)
