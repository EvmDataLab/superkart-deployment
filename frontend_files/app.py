import os
import pandas as pd
import requests
import streamlit as st

# Backend URL: the Docker network resolves the container name "superkart-backend"
BACKEND_URL = os.getenv("BACKEND_URL", "http://superkart-backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecaster", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Forecaster")
st.caption("Predict the sales revenue of a product in a given store.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ---------------------------------------------------------------- single
with tab_single:
    st.subheader("Enter product and store details")
    with st.form("single_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Product details**")
            product_weight = st.number_input("Product weight", 0.0, 50.0, 12.66, 0.01)
            sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
            area = st.number_input("Allocated display area (ratio)", 0.0, 1.0, 0.027, 0.001, format="%.3f")
            mrp = st.number_input("Product MRP", 0.0, 1000.0, 117.08, 0.01)
            id_char = st.selectbox("Product class (Id prefix)", ["FD", "NC", "DR"],
                                   help="FD = Food, NC = Non-consumable, DR = Drinks")
            category = st.selectbox("Product category", ["Perishables", "Non Perishables"])
        with col2:
            st.markdown("**Store details**")
            store_size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
            city = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
            store_type = st.selectbox("Store type", ["Departmental Store", "Supermarket Type1",
                                                     "Supermarket Type2", "Food Mart"], index=2)
            store_age = st.number_input("Store age (years)", 0, 100, 16, 1)
        submitted = st.form_submit_button("Predict sales")

    if submitted:
        payload = {
            "Product_Weight": product_weight, "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area, "Product_MRP": mrp, "Store_Size": store_size,
            "Store_Location_City_Type": city, "Store_Type": store_type,
            "Product_Id_char": id_char, "Store_Age_Years": int(store_age),
            "Product_Type_Category": category,
        }
        try:
            resp = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if resp.status_code == 200:
                value = resp.json()["Predicted_Product_Store_Sales_Total"]
                st.success("Prediction successful")
                st.metric("Predicted sales revenue", f"{value:,.2f}")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")

# ----------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Upload a CSV for batch prediction")
    st.write("Required columns: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_MRP, "
             "Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, Store_Age_Years, "
             "Product_Type_Category")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.dataframe(batch_df.head())
        if st.button("Predict for all rows"):
            try:
                resp = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                     files={"file": batch_df.to_csv(index=False).encode("utf-8")}, timeout=120)
                if resp.status_code == 200:
                    preds = pd.Series(resp.json(), name="Predicted_Sales")
                    preds.index = preds.index.astype(int)
                    result = batch_df.join(preds)
                    st.success(f"Predicted {len(result)} rows")
                    st.dataframe(result)
                    st.metric("Total predicted sales", f"{result['Predicted_Sales'].sum():,.2f}")
                    st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"),
                                       "superkart_predictions.csv", "text/csv")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")
