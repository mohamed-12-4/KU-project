import streamlit as st
import pandas as pd
import numpy as np
import joblib
from blackscholes import BlackScholesPut


# ---- Load pretrained TabPFN model ----
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return joblib.load(f)

# ----- Load the Scaler ----
@st.cache_resource
def load_scaler(path: str):
    with open(path, "rb") as f:
        return joblib.load(f)
MODEL_PATH = "./tabpfn_model_1.pkl"
SCALER_PATH = "./scaler_1.pkl"
tabpfn_model = load_model(MODEL_PATH)

st.title("🏗️ TabPFN Prediction Dashboard")
st.markdown("Enter your feature values below and get a prediction from TabPFN.")

# ---- User inputs ----
st.sidebar.header("Input Features")

def calculate_bsm(S, K, T, r, sigma):
    return BlackScholesPut(S, K, T, r, sigma).price()

def user_input_features():
    # Replace with your actual feature names
    feature_names = ["asset_price", "maturity", "rate", "div", "ivol"]
    data = {feat: st.sidebar.number_input(feat, value=0.0, format="%.6f") for feat in feature_names}
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("📄 Upload CSV File (optional)")
uploaded_file = st.file_uploader("Upload CSV with same features", type=["csv"])

if uploaded_file is not None:
    try:
        df_csv = pd.read_csv(uploaded_file)
        st.write("Uploaded data:", df_csv.head())
        #check if contains target column to skip it
        if 'american_op' in df_csv.columns:
            df_csv = df_csv.drop(columns=['american_op'])
        input_df = df_csv  # override the manual input
    except Exception as e:
        st.error(f"Error reading CSV: {e}")


if st.button("Predict"):
# ---- Feature engineering & scaling  ----
    with st.spinner("Applying feature engineering and scaling..."):

        df = input_df.copy()
        df["log_moneyness"] = np.log(df["asset_price"] / (df["asset_price"]))
        df["maturity_sqrt"] = np.sqrt(df["maturity"])  # from BSM model
        df["rate_minus_div"] = df["rate"] - df["div"]
        df["vega_like"] = df["ivol"] * df["asset_price"] * np.sqrt(df["maturity"])
        df["discount_factor"] = np.exp(-df["rate"] * df["maturity"])  # PV effect
        df['european_op'] = df.apply(lambda row: calculate_bsm(row['asset_price'], 100, row['maturity'], row['rate'], row['ivol']), axis=1)


        X_scaled = load_scaler(SCALER_PATH).transform(df)

    # ---- Prediction ----
    with st.spinner("Predicting..."):
        if len(X_scaled) > 10_000:
            preds = []
            st.warning("Large input detected, prediction may take a while.")
    
            for i in range(0, len(X_scaled), 10_000):
                batch = X_scaled[i:i + 10_000]
                preds.extend(tabpfn_model.predict(batch))
        
            
            st.write("Predictions:")
            st.dataframe(pd.DataFrame(np.array(preds) + df['european_op'].values, columns=["Prediction"]))
            
            
        
        preds = tabpfn_model.predict(X_scaled)
        st.subheader("Prediction Result")
        if len(preds) == 1:
            st.write(f"Estimated target: **{preds[0]:.6f}**")
        else:
            st.write("Predictions:")
            st.dataframe(pd.DataFrame(np.array(preds) + df['european_op'].values, columns=["Prediction"]))

        st.caption("Model: TabPFNRegressor")
