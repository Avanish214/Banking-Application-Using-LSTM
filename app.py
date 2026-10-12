from pathlib import Path

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from fraud_detection import prepare_sequences, train_lstm


st.set_page_config(page_title="Bank Fraud Monitor", page_icon="🏦", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink: #172522; --muted: #64736f; --line: #dce5e1; --green: #16745b; --red: #c84343; }
    .stApp { background: #f5f8f6; color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    .eyebrow { color: var(--green); font-size: .76rem; font-weight: 700; text-transform: uppercase; }
    .subtle { color: var(--muted); }
    div[data-testid="stMetric"] { background: white; border: 1px solid var(--line); padding: 1rem; border-radius: 6px; }
    div[data-testid="stMetric"] label { color: var(--muted); }
    div.stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); }
    .risk { border-left: 4px solid var(--red); background: #fff; padding: 1rem 1.2rem; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Risk intelligence / model workspace</div>', unsafe_allow_html=True)
st.title("Transaction fraud monitor")
st.markdown(
    '<p class="subtle">Train a sequence model on historical transactions, then inspect its held-out risk signals.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Training data")
    uploaded = st.file_uploader("Kaggle transaction CSV", type="csv")
    st.caption("Expected: numeric transaction features and a binary Class column.")
    sequence_length = st.slider("Transactions per sequence", min_value=2, max_value=20, value=5)
    epochs = st.slider("Maximum epochs", min_value=1, max_value=30, value=10)
    threshold = st.slider("Fraud alert threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    train_clicked = st.button("Train LSTM", type="primary", use_container_width=True)

local_csv = Path("data/creditcard.csv")
if uploaded is not None:
    try:
        transactions = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
elif local_csv.exists():
    transactions = pd.read_csv(local_csv)
else:
    transactions = None

if transactions is None:
    st.info("Upload the Kaggle `creditcard.csv` in the sidebar to begin. The dataset is not bundled with this project.")
    st.markdown("#### Model design")
    st.code("Rolling transaction window → LSTM(64) → Dropout → LSTM(32) → Dense(16) → Fraud probability", language="text")
    st.caption("The public Kaggle file has no customer ID. This project forms chronological dataset-level windows, not customer histories.")
    st.stop()

if "Class" not in transactions.columns:
    st.error("The selected CSV must contain a `Class` column with 0 = genuine and 1 = fraud.")
    st.stop()

transactions["Class"] = pd.to_numeric(transactions["Class"], errors="coerce")
if transactions["Class"].isna().any():
    st.error("The `Class` column contains empty or non-numeric values.")
    st.stop()

fraud_count = int((transactions["Class"] == 1).sum())
genuine_count = int((transactions["Class"] == 0).sum())
if fraud_count + genuine_count != len(transactions):
    st.error("The `Class` column must contain only 0 and 1.")
    st.stop()

summary_cols = st.columns(4)
summary_cols[0].metric("Transactions", f"{len(transactions):,}")
summary_cols[1].metric("Fraud cases", f"{fraud_count:,}")
summary_cols[2].metric("Genuine", f"{genuine_count:,}")
summary_cols[3].metric("Fraud prevalence", f"{fraud_count / len(transactions):.3%}")

st.markdown("#### Class distribution")
distribution = pd.DataFrame({"Class": ["Genuine", "Fraud"], "Transactions": [genuine_count, fraud_count]})
st.bar_chart(distribution, x="Class", y="Transactions", color="#16745b", height=190)

if train_clicked:
    try:
        with st.spinner("Preparing chronological splits and fitting the LSTM..."):
            prepared = prepare_sequences(transactions, sequence_length)
            model, history, probabilities = train_lstm(prepared, epochs=epochs)
        st.session_state["training_result"] = (prepared, model, history, probabilities)
        st.success("Training complete. Metrics below use the final chronological holdout.")
    except ImportError:
        st.error("TensorFlow is not installed. Install project dependencies with `pip install -r requirements.txt`.")
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Training failed: {error}")

if "training_result" not in st.session_state:
    st.warning("Data loaded. Choose the sequence length and train the LSTM to see its held-out results.")
    st.stop()

prepared, model, history, probabilities = st.session_state["training_result"]
predicted = (probabilities >= threshold).astype(np.int32)
actual = prepared.test_y.astype(np.int32)
precision = precision_score(actual, predicted, zero_division=0)
recall = recall_score(actual, predicted, zero_division=0)
f1 = f1_score(actual, predicted, zero_division=0)
auc = roc_auc_score(actual, probabilities) if np.unique(actual).size == 2 else float("nan")

st.markdown("#### Held-out performance")
metric_cols = st.columns(4)
metric_cols[0].metric("Precision", f"{precision:.3f}")
metric_cols[1].metric("Recall", f"{recall:.3f}")
metric_cols[2].metric("F1 score", f"{f1:.3f}")
metric_cols[3].metric("ROC-AUC", f"{auc:.3f}" if np.isfinite(auc) else "Unavailable")
st.caption(f"Classification threshold: {threshold:.2f}. Adjust it in the sidebar to explore the precision/recall trade-off.")

result_cols = st.columns([1, 1.25])
matrix = confusion_matrix(actual, predicted, labels=[0, 1])
figure = ff.create_annotated_heatmap(
    z=matrix,
    x=["Predicted genuine", "Predicted fraud"],
    y=["Actual genuine", "Actual fraud"],
    colorscale=[[0, "#edf3ef"], [1, "#16745b"]],
    showscale=True,
)
figure.update_layout(height=310, margin=dict(l=10, r=10, t=20, b=10), font_color="#172522")
figure.update_yaxes(autorange="reversed")
with result_cols[0]:
    st.markdown("#### Confusion matrix")
    st.plotly_chart(figure, use_container_width=True)

with result_cols[1]:
    st.markdown("#### Highest-risk held-out transaction")
    risk_index = int(np.argmax(probabilities))
    risk_probability = float(probabilities[risk_index])
    status = "FRAUD ALERT" if risk_probability >= threshold else "BELOW ALERT THRESHOLD"
    st.markdown(
        f'<div class="risk"><strong>{status}</strong><br>Fraud probability: {risk_probability:.1%}<br>'
        f'Observed label: {"Fraud" if actual[risk_index] else "Genuine"}</div>',
        unsafe_allow_html=True,
    )
    amount = prepared.test_amounts[risk_index]
    if np.isfinite(amount):
        st.metric("Transaction amount", f"{amount:,.2f}")
    if prepared.test_times is not None:
        st.metric("Dataset time", f"{prepared.test_times[risk_index]:,.0f} seconds")
    if np.isfinite(amount):
        history_amounts = transactions.sort_values("Time", kind="stable")["Amount"].to_numpy() if "Time" in transactions else transactions["Amount"].to_numpy()
        test_start = int(len(transactions) * 0.85) + sequence_length - 1
        prior_start = test_start + risk_index - sequence_length + 1
        prior = history_amounts[max(0, prior_start):prior_start + sequence_length - 1]
        st.caption("Prior rows in this dataset-level window: " + " · ".join(f"{value:,.2f}" for value in prior))

with st.expander("Training history"):
    history_frame = pd.DataFrame(history.history)
    if "loss" in history_frame and "val_loss" in history_frame:
        st.line_chart(history_frame[["loss", "val_loss"]], height=220)

st.caption("Research and training demonstration only. The Kaggle dataset contains no customer identifier; windows do not represent verified customer histories.")