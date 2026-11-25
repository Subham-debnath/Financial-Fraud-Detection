import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Fraud Operations", page_icon="🛡️", layout="wide")
BASE = Path(__file__).parent
FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
LABELS = ['Class', 'class', 'Label', 'label', 'Actual', 'actual', 'is_fraud', 'isFraud']

st.markdown('''<style>
.stApp{background:#f4f6fa;color:#15243b} section[data-testid="stSidebar"]{background:#101d32}
section[data-testid="stSidebar"] *{color:#e6edf8!important} h1,h2,h3{color:#15243b;letter-spacing:-.03em}
div[data-testid="stMetric"]{background:white;border:1px solid #e3e9f1;border-radius:12px;padding:20px}
div[data-testid="stMetricLabel"]{color:#63738b} .stButton button,.stDownloadButton button{border-radius:8px}
.intro{background:#142741;color:white;border-radius:14px;padding:24px 28px;margin-bottom:20px}
.intro h2{color:white;margin:0}.intro p{color:#b9c9dd;margin:8px 0 0}
.tag{color:#66758b;font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
</style>''', unsafe_allow_html=True)

for key, default in {'runs': [], 'results': None, 'reviews': {}, 'run_id': 0}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)

def score(model, data, threshold):
    if hasattr(model, 'predict_proba'):
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError('The model must have a fraud class labelled 1.')
        probabilities = model.predict_proba(data)[:, classes.index(1)]
        return (probabilities >= threshold).astype(int), probabilities
    return np.asarray(model.predict(data)), None

def validate(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    x = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    if x.isna().any().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('All required features must contain finite numeric values, with no missing values.')
    return x

def status_frame():
    df = st.session_state.results.copy()
    df['Review status'] = [st.session_state.reviews.get(int(i), 'New') for i in df['_row_id']]
    return df

models = sorted(BASE.glob('*.pkl'))
with st.sidebar:
    st.markdown('### ◈ FRAUD OPERATIONS')
    st.caption('Transaction risk workspace')
    page = st.radio('Navigation', ['Overview', 'Batch scoring', 'Investigation queue', 'Single transaction', 'Run history', 'Model information'], label_visibility='collapsed')
    st.divider()
    if models:
        selected = st.selectbox('Model file', models, format_func=lambda p: p.name)
        try:
            model = load_model(str(selected), selected.stat().st_mtime_ns)
            st.success('Model loaded')
        except Exception as exc:
            model = None
            st.error(f'Model could not be loaded: {exc}')
    else:
        model = None
        st.warning('Place your trusted .pkl model in the same folder as this app.')
    st.caption('Review data is held in this browser session.')

if page == 'Overview':
    st.markdown('<div class="intro"><div class="tag" style="color:#8fa9ca">Risk intelligence</div><h2>Fraud operations overview</h2><p>Score transactions, prioritize alerts, and review model results.</p></div>', unsafe_allow_html=True)
    runs = st.session_state.runs
    total = sum(r['Transactions'] for r in runs)
    flagged = sum(r['Flagged'] for r in runs)
    a,b,c,d = st.columns(4)
    a.metric('Transactions scored', f'{total:,}')
    b.metric('Flagged for review', f'{flagged:,}')
    c.metric('Flagged rate', f'{flagged / total:.2%}' if total else '—')
    d.metric('Scoring runs', len(runs))
    st.subheader('Current investigation queue')
    if st.session_state.results is None:
        st.info('No batch has been scored yet. Open Batch scoring to upload a CSV.')
    else:
        q = status_frame()
        q = q[q['Prediction'] == 'Flagged']
        if q.empty: st.success('No transactions were flagged in the latest batch at the selected threshold.')
        else:
            cols = [c for c in ['_row_id', 'Time', 'Amount', 'Fraud score', 'Review status'] if c in q]
            st.dataframe(q[cols].head(10), hide_index=True, use_container_width=True)
            st.caption('Open Investigation queue to review and filter all flagged transactions.')

elif page == 'Batch scoring':
    st.title('Batch scoring')
    st.caption('Upload a CSV containing Time, V1–V28 and Amount. Additional columns are retained in the output.')
    file = st.file_uploader('Transaction data', type=['csv'])
    threshold = st.slider('Decision threshold', 0.0, 1.0, 0.5, 0.01, help='Transactions at or above this model score are flagged. Changing the threshold changes the balance between missed fraud and alerts.')
    if file:
        try:
            df = pd.read_csv(file)
            x = validate(df)
            st.caption(f'{len(df):,} rows ready to score')
            st.dataframe(df.head(5), hide_index=True, use_container_width=True)
            if st.button('Score transactions', type='primary', disabled=model is None):
                if df.empty: raise ValueError('The CSV has no transaction rows.')
                with st.spinner('Scoring transactions...'):
                    pred, probability = score(model, x, threshold)
                result = df.copy().reset_index(drop=True)
                result.insert(0, '_row_id', np.arange(1, len(result)+1))
                result['Prediction'] = np.where(pred == 1, 'Flagged', 'Not flagged')
                if probability is not None: result['Fraud score'] = probability
                st.session_state.results = result
                st.session_state.reviews = {}
                st.session_state.run_id += 1
                st.session_state.runs.append({'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'File': file.name, 'Model': selected.name, 'Threshold': threshold if probability is not None else 'Model default', 'Transactions': len(result), 'Flagged': int((pred == 1).sum())})
                st.success('Scoring complete. Open the investigation queue to review results.')
        except Exception as exc: st.error(f'Unable to process file: {exc}')
    if st.session_state.results is not None:
        result = status_frame()
        a,b,c = st.columns(3)
        a.metric('Transactions', len(result))
        b.metric('Flagged', int((result['Prediction'] == 'Flagged').sum()))
        c.metric('Flagged rate', f"{(result['Prediction'] == 'Flagged').mean():.2%}")
        st.download_button('Download results CSV', result.drop(columns=['_row_id']).to_csv(index=False).encode('utf-8'), 'scored_transactions.csv', 'text/csv')

elif page == 'Investigation queue':
    st.title('Investigation queue')
    st.caption('Review alerts from the latest batch. Review decisions are stored for this session only.')
    if st.session_state.results is None: st.info('Score a batch to populate the queue.')
    else:
        df = status_frame()
        a,b = st.columns(2)
        prediction = a.selectbox('Prediction', ['Flagged', 'All', 'Not flagged'])
        status = b.multiselect('Review status', ['New', 'Under review', 'Confirmed fraud', 'Cleared'], default=['New', 'Under review', 'Confirmed fraud', 'Cleared'])
        view = df[df['Review status'].isin(status)]
        if prediction != 'All': view = view[view['Prediction'] == prediction]
        if 'Fraud score' in view: view = view.sort_values('Fraud score', ascending=False)
        st.caption(f'{len(view):,} transactions')
        cols = [c for c in ['_row_id', 'Time', 'Amount', 'Prediction', 'Fraud score', 'Review status'] if c in view]
        st.dataframe(view[cols], hide_index=True, use_container_width=True)
        if not view.empty:
            row_id = st.selectbox('Select transaction ID to review', view['_row_id'].tolist())
            row = df.loc[df['_row_id'] == row_id].iloc[0]
            st.subheader(f'Transaction #{row_id}')
            a,b,c = st.columns(3)
            a.metric('Decision', row['Prediction'])
            b.metric('Amount', f"{row['Amount']:,.2f}")
            c.metric('Model fraud score', f"{row['Fraud score']:.1%}" if 'Fraud score' in row else 'Unavailable')
            new_status = st.selectbox('Review decision', ['New', 'Under review', 'Confirmed fraud', 'Cleared'], index=['New', 'Under review', 'Confirmed fraud', 'Cleared'].index(row['Review status']))
            if st.button('Save review decision'):
                st.session_state.reviews[int(row_id)] = new_status
                st.rerun()
            with st.expander('All transaction fields'): st.dataframe(row.to_frame('Value'), use_container_width=True)
        st.download_button('Download filtered queue', view.drop(columns=['_row_id']).to_csv(index=False).encode('utf-8'), 'investigation_queue.csv', 'text/csv')

elif page == 'Single transaction':
    st.title('Single transaction')
    st.caption('Enter the model features to score one transaction. V1–V28 are transformed dataset features, not ordinary card details.')
    threshold = st.slider('Decision threshold', 0.0, 1.0, 0.5, 0.01, key='single_threshold')
    with st.form('single'):
        a,b = st.columns(2)
        values = {'Time': a.number_input('Time', value=0.0), 'Amount': b.number_input('Amount', min_value=0.0, value=0.0)}
        with st.expander('PCA features V1–V28', expanded=True):
            cols = st.columns(4)
            for i in range(1,29): values[f'V{i}'] = cols[(i-1)%4].number_input(f'V{i}', value=0.0, format='%.6f')
        submitted = st.form_submit_button('Score transaction', type='primary', disabled=model is None)
    if submitted:
        try:
            pred, probability = score(model, pd.DataFrame([values])[FEATURES], threshold)
            if pred[0] == 1: st.error('Flagged for review')
            else: st.success('Not flagged at the selected threshold')
            if probability is not None: st.metric('Model fraud score', f'{probability[0]:.1%}')
            st.caption('A model flag is a prediction; a reviewer determines the final outcome.')
        except Exception as exc: st.error(f'Prediction failed: {exc}')

elif page == 'Run history':
    st.title('Run history')
    st.caption('Batch scoring runs in this session.')
    if st.session_state.runs:
        st.dataframe(pd.DataFrame(st.session_state.runs), hide_index=True, use_container_width=True)
        if st.button('Clear run history'): st.session_state.runs = []; st.rerun()
    else: st.info('No scoring runs yet.')

else:
    st.title('Model information')
    if model is None: st.info('Load a model to see its details.')
    else:
        st.write('**Active model:**', selected.name)
        st.write('**Estimator:**', type(model).__name__)
        st.write('**Probability scores:**', 'Available' if hasattr(model, 'predict_proba') else 'Unavailable; model default prediction is used.')
        st.caption('Performance metrics depend on the model and evaluation data. Supply verified evaluation results before displaying them here.')
        if hasattr(model, 'feature_importances_') and len(model.feature_importances_) == len(FEATURES):
            imp = pd.DataFrame({'Feature': FEATURES, 'Importance': model.feature_importances_}).nlargest(10, 'Importance').sort_values('Importance')
            st.plotly_chart(px.bar(imp, x='Importance', y='Feature', orientation='h', title='Global feature importance'), use_container_width=True)
            st.caption('Global importance describes the model overall; it does not explain an individual transaction.')
