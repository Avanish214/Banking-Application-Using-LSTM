# Bank Fraud Detection with LSTM

A small end-to-end training dashboard for the Kaggle Credit Card Fraud Detection dataset. Upload `creditcard.csv`, train a sequence classifier, inspect held-out metrics, and review the highest-scored held-out transaction.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Download `creditcard.csv` from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv` for automatic loading, or upload it in the dashboard. The dataset is not included in this repository. The CSV must contain numeric features and a binary `Class` column (`0` genuine, `1` fraud); the standard Kaggle file is supported as-is.

## What the pipeline does

- Sorts rows by `Time` when available, then creates fixed-length rolling windows. Each window predicts the label of its final transaction.
- Splits chronologically into 70% training, 15% validation, and 15% test data before scaling, preventing future transactions from leaking into training.
- Fits a `StandardScaler` on training rows only and uses class weights to account for the rare fraud class.
- Trains an LSTM(64) → Dropout → LSTM(32) → Dense(16) → Sigmoid model. The dashboard reports precision, recall, F1, ROC-AUC, and a threshold-adjustable confusion matrix on held-out test data.

## Important dataset limitation

The public Kaggle dataset has `Time`, anonymized `V1`–`V28` features, `Amount`, and `Class`, but no customer or card identifier. Consequently, these windows are consecutive rows in the overall time-sorted dataset, **not one customer's transaction history**. This project demonstrates sequence modeling but cannot substantiate customer-specific behavior or production-grade fraud decisions. A true per-customer model needs transaction data with a stable customer/card key and should be evaluated with customer-aware and time-aware splits.

The dashboard's highest-risk example is selected from the held-out test set. Its previous-transaction amounts are the preceding rows in the dataset-level window, not verified transactions from the same customer. Threshold selection is an educational control; blocking or challenging a live transaction requires separate validation, calibration, and bank risk controls.