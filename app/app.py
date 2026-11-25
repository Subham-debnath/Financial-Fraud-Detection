import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
from pathlib import Path
"""Optional model diagnostics for the fraud investigation application."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

LABELS = ["Class", "class", "Label", "label", "Actual", "actual", "is_fraud", "isFraud"]


def render_importance(model, features):
    """Display global importance only when it matches the expected features."""
    values = getattr(model, "feature_importances_", None)
    if values is None:
        st.info("Global feature importance is unavailable for this model.")
        return
    if len(values) != len(features):
        st.info("Feature importance does not match the expected input columns.")
        return
    data = pd.DataFrame({"Feature": features, "Importance": values}).nlargest(10, "Importance").sort_values("Importance")
    fig = px.bar(data, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#4868e8"])
    fig.update_layout(height=360, margin=dict(l=5, r=10, t=10, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Global model importance describes the model overall, not the reason for an individual prediction.")


def render_explanation(model, X):
    """Try a SHAP tree explanation for a single transaction."""
    try:
        import shap
    except ImportError:
        st.info("Install the optional shap package to see an explanation for this transaction.")
        return
    try:
        if len(X) != 1:
            raise ValueError("Select exactly one transaction.")
        raw = shap.TreeExplainer(model).shap_values(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        fraud_index = classes.index(1) if 1 in classes else None
        if isinstance(raw, list):
            if fraud_index is None:
                raise ValueError("Fraud class 1 not found.")
            values = np.asarray(raw[fraud_index])[0]
        else:
            array = np.asarray(raw)
            if array.ndim == 3:
                if fraud_index is None:
                    raise ValueError("Fraud class 1 not found.")
                if array.shape[1] == X.shape[1]:
                    values = array[0, :, fraud_index]
                elif array.shape[2] == X.shape[1]:
                    values = array[fraud_index, 0, :]
                else:
                    raise ValueError("Unsupported explanation shape.")
            elif array.ndim == 2:
                values = array[0]
            else:
                raise ValueError("Unsupported explanation shape.")
        if len(values) != X.shape[1] or not np.isfinite(values).all():
            raise ValueError("Explanation does not match the input features.")
        data = pd.DataFrame({"Feature": X.columns, "Contribution": values})
        data = data.iloc[np.argsort(np.abs(data["Contribution"].to_numpy()))[-10:]]
        data["Direction"] = np.where(data["Contribution"] >= 0, "Increases score", "Decreases score")
        fig = px.bar(data, x="Contribution", y="Feature", color="Direction", orientation="h", color_discrete_map={"Increases score": "#e45b61", "Decreases score": "#16a085"})
        fig.update_layout(height=360, margin=dict(l=5, r=10, t=10, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("SHAP contributions describe the model output relative to its baseline; transformed V features are not directly interpretable as transaction attributes.")
    except Exception as exc:
        st.info(f"A per-transaction explanation is unavailable for this model: {exc}")


def render_evaluation(df):
    """Evaluate predictions only when a valid 0/1 ground-truth column exists."""
    label = next((name for name in LABELS if name in df.columns), None)
    if label is None:
        st.info("Add a 0/1 ground-truth label column to the CSV to evaluate predictions.")
        return
    try:
        from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
        actual = pd.to_numeric(df[label], errors="coerce")
        if actual.isna().any() or not actual.isin([0, 1]).all():
            st.warning(f"Column {label!r} must contain only 0 and 1, with no missing values.")
            return
        if not df["Prediction"].isin(["Fraud", "Legitimate"]).all():
            st.warning("Predictions must be labelled Fraud or Legitimate for evaluation.")
            return
        y = actual.astype(int).to_numpy()
        predicted = (df["Prediction"] == "Fraud").astype(int).to_numpy()
        matrix = confusion_matrix(y, predicted, labels=[0, 1])
        st.caption(f"Ground truth: {label}. Metrics describe this uploaded dataset at the selected threshold.")
        fig = go.Figure(data=go.Heatmap(z=matrix, x=["Predicted legitimate", "Predicted fraud"], y=["Actual legitimate", "Actual fraud"], colorscale="Blues", showscale=False, text=matrix, hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>"))
        for i in range(2):
            for j in range(2):
                fig.add_annotation(x=j, y=i, text=str(matrix[i, j]), showarrow=False, font=dict(color="#15243b", size=18))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.text(classification_report(y, predicted, labels=[0, 1], target_names=["Legitimate", "Fraud"], zero_division=0))
        if len(np.unique(y)) < 2:
            st.info("ROC and precision-recall curves require both classes in the uploaded labels.")
            return
        if "Fraud_Probability" not in df:
            st.info("Probability scores are unavailable; ROC and precision-recall curves cannot be computed.")
            return
        scores = pd.to_numeric(df["Fraud_Probability"], errors="coerce").to_numpy()
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            st.warning("Fraud scores must be finite values between 0 and 1.")
            return
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC {auc(fpr,tpr):.3f}"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="#aab5c6")))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", height=320)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", name=f"AP {average_precision_score(y,scores):.3f}"))
            fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", height=320, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install scikit-learn to calculate evaluation metrics.")
    except Exception as exc:
        st.warning(f"Could not evaluate this dataset: {exc}")


BASE_DIR = Path(__file__).parent
REQUIRED_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

st.set_page_config(page_title='Fraud Operations', page_icon='◈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="st-"]{font-family:'DM Sans',sans-serif} .stApp{background:#f3f6fa;color:#12243a}
[data-testid="stHeader"]{background:transparent} .block-container{max-width:1450px;padding-top:2.2rem;padding-bottom:3rem}
h1,h2,h3{font-family:Manrope,sans-serif;color:#10233a;letter-spacing:-.035em} h1{font-size:2rem!important}
section[data-testid="stSidebar"]{background:#101f33} section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] span{color:#d3dfec}
section[data-testid="stSidebar"] [role="radiogroup"] label{border-radius:9px;padding:8px 10px;margin:2px 0}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#213953}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e1e8f0;border-radius:14px;padding:18px 20px;box-shadow:0 3px 14px #14284008}
div[data-testid="stMetricLabel"]{color:#67778b;font-size:.85rem} div[data-testid="stMetricValue"]{color:#10233a;font-family:Manrope,sans-serif;font-weight:800}
div[data-testid="stDataFrame"]{border:1px solid #e0e7ef;border-radius:12px;overflow:hidden}
div[data-testid="stFileUploaderDropzone"]{background:#fff;border:1px dashed #aebfd1;border-radius:14px}
div.stButton>button,div.stDownloadButton>button{border-radius:9px;font-weight:700;min-height:42px}
div.stButton>button[kind="primary"]{background:#145bdb;border-color:#145bdb}
.brand{padding:8px 4px 22px;border-bottom:1px solid #34445a;margin-bottom:20px}.brand-logo{display:inline-block;color:#8cd8d2;background:#263f51;border-radius:10px;padding:8px 12px;font-weight:800;font-size:20px;margin-right:9px}.brand-name{color:#f4f8fc;font-family:Manrope;font-size:17px;font-weight:800}.brand-sub{font-size:11px;color:#9cafc4;letter-spacing:.12em;margin-top:8px;text-transform:uppercase}
.eyebrow{color:#547394;font-size:11px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px}
.heading-note{color:#64758a;font-size:15px;margin-top:-8px;margin-bottom:20px}
.hero{background:linear-gradient(110deg,#12273f,#1b3b58);border-radius:16px;padding:25px 28px;margin-bottom:20px;color:#fff}.hero .eyebrow{color:#84d2c9}.hero h2{color:#fff;font-size:24px;margin:3px 0 8px}.hero p{color:#bdccda;margin:0;max-width:670px;font-size:14px;line-height:1.6}
.section-label{color:#667b92;font-size:11px;letter-spacing:.13em;font-weight:800;text-transform:uppercase;margin:22px 0 8px}
.empty{background:#fff;border:1px solid #e1e8f0;border-radius:14px;padding:30px;color:#607389;text-align:center}.empty strong{display:block;color:#172d45;font-size:17px;margin-bottom:6px}
.detail{background:#fff;border:1px solid #e1e8f0;border-radius:14px;padding:20px 22px;margin:8px 0}.detail .title{font-family:Manrope;font-weight:800;color:#172c43;font-size:18px}.detail .sub{font-size:13px;color:#67798c;margin-top:6px}
.flag{display:inline-block;background:#fff0ee;color:#ac3a38;padding:5px 10px;border-radius:6px;font-size:11px;font-weight:800;letter-spacing:.04em}.clear{background:#e6f6f1;color:#08775d}
@media(max-width:700px){.block-container{padding-top:1rem}.hero{padding:20px}h1{font-size:1.6rem!important}}
</style>''', unsafe_allow_html=True)

for k,v in {'history':[], 'batch':None, 'reviews':{}, 'single':None, 'nav':'Overview'}.items():
    if k not in st.session_state: st.session_state[k]=v

@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)

def predict(model, x, threshold):
    if hasattr(model, 'predict_proba'):
        classes = list(model.classes_)
        if 1 not in classes: raise ValueError('Model classes must include 1 for fraud.')
        probs = np.asarray(model.predict_proba(x))[:, classes.index(1)]
        if not np.isfinite(probs).all(): raise ValueError('Model returned invalid probabilities.')
        return (probs >= threshold).astype(int), probs
    preds = np.asarray(model.predict(x))
    if not np.isin(preds, [0,1]).all(): raise ValueError('Model predictions must use 0 and 1.')
    return preds, None

def prepare(df):
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    if df.empty: raise ValueError('The CSV contains no rows.')
    reserved=[c for c in ['Prediction','Fraud_Probability','Review_Status','Case_ID'] if c in df.columns]
    if reserved: raise ValueError('Rename these reserved output columns before scoring: '+', '.join(reserved))
    x=df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors='coerce')
    invalid=x.isna() | ~np.isfinite(x)
    if invalid.any().any(): raise ValueError('Missing or non-numeric values in: '+', '.join(invalid.columns[invalid.any()].tolist()))
    return x

def batch_view():
    result=st.session_state.batch['data'].copy()
    result['Review_Status']=result['Case_ID'].map(st.session_state.reviews).fillna('New')
    return result

def heading(kicker,title,subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div>',unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="heading-note">{subtitle}</div>',unsafe_allow_html=True)

def navigate(target):
    st.session_state.nav = target

def summary(df):
    n=len(df); f=int((df['Prediction']=='Fraud').sum())
    a,b,c,d=st.columns(4)
    a.metric('Transactions',f'{n:,}'); b.metric('Flagged for review',f'{f:,}'); c.metric('Not flagged',f'{n-f:,}'); d.metric('Flagged rate',f'{f/n:.2%}' if n else '—')

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-logo">◈</span><span class="brand-name">SENTINEL</span><div class="brand-sub">Fraud intelligence workspace</div></div>',unsafe_allow_html=True)
    st.caption('WORKSPACE')
    page=st.radio('Navigation',['Overview','Batch analysis','Investigation queue','Single transaction','Run history','Model information'],key='nav',label_visibility='collapsed')
    st.divider()
    paths=sorted(BASE_DIR.glob('*.pkl'))
    if not paths: paths=[BASE_DIR/'model.pkl']
    selected=st.selectbox('Active model',paths,format_func=lambda p:p.name) if len(paths)>1 else paths[0]
    if len(paths)==1: st.caption('MODEL  ·  '+selected.name)
    try:
        model=load_model(str(selected),selected.stat().st_mtime_ns)
        st.caption('●  Model loaded')
    except Exception as exc:
        model=None
        st.warning(f'Model unavailable: {exc}')
    st.divider(); st.caption('Session workspace · Reviews and history are temporary')

if page=='Overview':
    heading('Operations / Overview','Risk at a glance','A clear view of the latest scoring activity and transactions requiring attention.')
    st.markdown('<div class="hero"><div class="eyebrow">Your investigation workspace</div><h2>Find the signal. Review with confidence.</h2><p>Bring transaction scoring, risk analysis, and case review together in one focused workspace.</p></div>',unsafe_allow_html=True)
    if st.session_state.batch is None:
        a,b,c,d=st.columns(4)
        for col,label in zip([a,b,c,d],['Transactions','Flagged for review','Not flagged','Flagged rate']): col.metric(label,'—')
        st.markdown('<div class="section-label">Getting started</div><div class="empty"><strong>Your workspace is ready</strong>Upload a transaction CSV to generate your first risk overview.</div>',unsafe_allow_html=True)
        st.button('Start batch analysis →',type='primary',on_click=navigate,args=('Batch analysis',))
    else:
        df=batch_view(); summary(df)
        st.markdown('<div class="section-label">Latest batch · priority queue</div>',unsafe_allow_html=True)
        q=df[df.Prediction=='Fraud']
        if 'Fraud_Probability' in q: q=q.sort_values('Fraud_Probability',ascending=False)
        if q.empty: st.info('No transactions were flagged in this batch at the selected threshold.')
        else:
            st.dataframe(q[[c for c in ['Case_ID','Time','Amount','Fraud_Probability','Review_Status'] if c in q]].head(10),hide_index=True,use_container_width=True)
            st.button('Open investigation queue →',on_click=navigate,args=('Investigation queue',))
        st.caption(f"Latest batch: {st.session_state.batch['filename']} · {st.session_state.batch['timestamp']} · Session data")

elif page=='Batch analysis':
    heading('Scoring / Batch analysis','Analyze transactions','Upload a CSV, check its structure, and run the selected model.')
    left,right=st.columns([1.6,1])
    with left:
        file=st.file_uploader('Transaction CSV',type=['csv'],help='Required columns: Time, V1–V28, Amount.')
    with right:
        st.markdown('<div class="detail"><div class="title">Before you begin</div><div class="sub">Required fields: Time, V1–V28 and Amount. Additional columns are preserved. A 0/1 label column enables evaluation.</div></div>',unsafe_allow_html=True)
    threshold=st.slider('Fraud decision threshold',0.0,1.0,0.5,0.01,disabled=model is not None and not hasattr(model,'predict_proba'),help='Scores at or above this threshold are flagged. Lower thresholds can increase alerts.')
    if file:
        try:
            raw=pd.read_csv(file); x=prepare(raw)
            with st.expander(f'Preview input · {len(raw):,} transactions',expanded=True): st.dataframe(raw.head(5),hide_index=True,use_container_width=True)
            if st.button('Run fraud analysis',type='primary',disabled=model is None):
                with st.spinner('Analyzing transactions...'): pred,prob=predict(model,x,threshold)
                out=raw.reset_index(drop=True).copy(); out[REQUIRED_COLUMNS]=x.reset_index(drop=True); out.insert(0,'Case_ID',np.arange(1,len(out)+1)); out['Prediction']=np.where(pred==1,'Fraud','Legitimate')
                if prob is not None: out['Fraud_Probability']=prob
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.batch={'data':out,'filename':file.name,'timestamp':timestamp,'model':selected.name,'threshold':threshold if prob is not None else None}
                st.session_state.reviews={}
                st.session_state.history.append({'Timestamp':timestamp,'File':file.name,'Model':selected.name,'Threshold':threshold if prob is not None else 'Model default','Total':len(out),'Flagged':int((pred==1).sum())})
                st.success('Analysis complete. Results are ready below and in the investigation queue.')
        except Exception as exc: st.error(f'Unable to analyze this file: {exc}')
    if st.session_state.batch is not None:
        df=batch_view(); st.markdown('<div class="section-label">Latest batch results</div>',unsafe_allow_html=True); summary(df)
        t1,t2,t3=st.tabs(['Risk overview','All transactions','Model evaluation'])
        with t1:
            a,b=st.columns(2)
            with a:
                counts=df.Prediction.value_counts().reindex(['Legitimate','Fraud'],fill_value=0)
                fig=px.pie(names=counts.index,values=counts.values,hole=.68,color=counts.index,color_discrete_map={'Legitimate':'#15a58a','Fraud':'#dc5a55'})
                fig.update_layout(title='Prediction distribution',height=320,margin=dict(t=45,b=10,l=10,r=10),legend=dict(orientation='h',y=-.1)); st.plotly_chart(fig,use_container_width=True)
            with b:
                fig=px.histogram(df,x='Amount',color='Prediction',nbins=35,color_discrete_map={'Legitimate':'#15a58a','Fraud':'#dc5a55'},barmode='overlay',opacity=.75)
                fig.update_layout(title='Transaction amounts',height=320,margin=dict(t=45,b=10,l=10,r=10),legend=dict(orientation='h',y=-.2)); st.plotly_chart(fig,use_container_width=True)
        with t2:
            st.dataframe(df,hide_index=True,use_container_width=True)
            st.download_button('Download full results',df.to_csv(index=False).encode('utf-8'),'predictions_full.csv','text/csv')
        with t3: render_evaluation(df)

elif page=='Investigation queue':
    heading('Operations / Investigations','Investigation queue','Prioritize flagged transactions and record a review decision for this session.')
    if st.session_state.batch is None: st.info('Run a batch analysis to populate the queue.')
    else:
        df=batch_view(); a,b,c=st.columns([1,1,1])
        choice=a.selectbox('Prediction',['Fraud','All','Legitimate'],format_func=lambda s:{'Fraud':'Flagged','All':'All transactions','Legitimate':'Not flagged'}[s])
        statuses=b.multiselect('Review status',['New','Under review','Confirmed fraud','Cleared'],default=['New','Under review','Confirmed fraud','Cleared'])
        minimum=c.number_input('Minimum amount',value=float(df.Amount.min()),max_value=float(df.Amount.max()))
        view=df[(df.Review_Status.isin(statuses)) & (df.Amount>=minimum)]
        if choice!='All': view=view[view.Prediction==choice]
        if 'Fraud_Probability' in view: view=view.sort_values('Fraud_Probability',ascending=False)
        st.caption(f'{len(view):,} matching transactions · IDs identify rows in the latest batch')
        st.dataframe(view[[c for c in ['Case_ID','Time','Amount','Prediction','Fraud_Probability','Review_Status'] if c in view]],hide_index=True,use_container_width=True)
        if not view.empty:
            case=st.selectbox('Select a case to inspect',view.Case_ID.tolist(),format_func=lambda i:f'Case #{i:05d}')
            row=df.loc[df.Case_ID==case].iloc[0]
            st.markdown(f'<div class="detail"><span class="flag {"" if row.Prediction=="Fraud" else "clear"}">{"FLAGGED FOR REVIEW" if row.Prediction=="Fraud" else "NOT FLAGGED"}</span><div class="title" style="margin-top:12px">Case #{case:05d}</div><div class="sub">Review the model signal alongside the transaction information.</div></div>',unsafe_allow_html=True)
            a,b,c=st.columns(3); a.metric('Amount',f"{row.Amount:,.2f}"); b.metric('Model fraud score',f"{row.Fraud_Probability:.2%}" if 'Fraud_Probability' in row else 'Unavailable'); c.metric('Current status',row.Review_Status)
            options=['New','Under review','Confirmed fraud','Cleared']
            status=st.selectbox('Set review status',options,index=options.index(row.Review_Status),key=f'review_{st.session_state.batch["timestamp"]}_{case}')
            if st.button('Save review',type='primary'): st.session_state.reviews[int(case)]=status; st.rerun()
            with st.expander('Transaction details'): st.dataframe(row.to_frame('Value'),use_container_width=True)
        st.download_button('Download filtered results',view.to_csv(index=False).encode('utf-8'),'predictions_filtered.csv','text/csv')
        st.download_button('Download flagged transactions',df[df.Prediction=='Fraud'].to_csv(index=False).encode('utf-8'),'predictions_fraud_only.csv','text/csv')

elif page=='Single transaction':
    heading('Scoring / Manual entry','Single transaction','Inspect a transaction using the model’s required input features.')
    threshold=st.slider('Fraud decision threshold',0.0,1.0,0.5,0.01,key='single_threshold',disabled=model is not None and not hasattr(model,'predict_proba'))
    with st.form('single_form'):
        a,b=st.columns(2); values={'Time':a.number_input('Time',value=0.0),'Amount':b.number_input('Amount',min_value=0.0,value=0.0)}
        st.caption('V1–V28 are transformed dataset features. Enter the values supplied to the model.')
        with st.expander('Enter V1–V28',expanded=True):
            cols=st.columns(4)
            for i in range(1,29): values[f'V{i}']=cols[(i-1)%4].number_input(f'V{i}',value=0.0,format='%.6f')
        submitted=st.form_submit_button('Analyze transaction',type='primary',disabled=model is None)
    if submitted:
        try:
            x=pd.DataFrame([values])[REQUIRED_COLUMNS]; pred,prob=predict(model,x,threshold)
            st.session_state.single={'x':x,'pred':int(pred[0]),'prob':None if prob is None else float(prob[0]),'model':str(selected),'threshold':threshold}
        except Exception as exc: st.error(f'Prediction failed: {exc}')
    result=st.session_state.single
    if result:
        st.markdown('<div class="section-label">Analysis result</div>',unsafe_allow_html=True)
        if result['pred']: st.error('Flagged for review')
        else: st.success('Not flagged at the selected threshold')
        a,b=st.columns(2); a.metric('Model fraud score',f"{result['prob']:.2%}" if result['prob'] is not None else 'Unavailable'); b.metric('Decision threshold',f"{result['threshold']:.0%}" if result['prob'] is not None else 'Model default')
        st.caption('The model score is a prediction, not a confirmed fraud outcome.')
        if model is not None and result['model']==str(selected): render_explanation(model,result['x'])

elif page=='Run history':
    heading('Workspace / Activity','Run history','Review batch scoring activity from this browser session.')
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history),hide_index=True,use_container_width=True)
        if st.button('Clear history'): st.session_state.history=[]; st.rerun()
    else: st.markdown('<div class="empty"><strong>No scoring runs yet</strong>Completed batch analyses will appear here.</div>',unsafe_allow_html=True)
else:
    heading('System / Model','Model information','Understand the selected estimator and the signals it uses.')
    if model is None: st.info('Place your model file next to this app to load it.')
    else:
        a,b,c=st.columns(3); a.metric('Estimator',type(model).__name__); b.metric('Model file',selected.name); c.metric('Probability scoring','Available' if hasattr(model,'predict_proba') else 'Unavailable')
        st.caption('Evaluation metrics are calculated from a batch containing verified 0/1 labels; no placeholder metrics are presented as model performance.')
        render_importance(model,REQUIRED_COLUMNS)
