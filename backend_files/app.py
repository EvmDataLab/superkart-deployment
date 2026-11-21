import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialise the Flask application
superkart_api = Flask("SuperKart Sales Predictor")

# Load the serialized model (preprocessing + regressor in one pipeline)
model = joblib.load("superkart_model.joblib")

FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category",
]


def prepare(df):
    """Select the model features in the right order and normalise known label issues."""
    df = df.copy()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
    return df[FEATURES]


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


# Online (single record) inference
@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        payload = request.get_json(force=True)
        missing = [f for f in FEATURES if f not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        input_df = prepare(pd.DataFrame([payload]))
        prediction = float(model.predict(input_df)[0])
        return jsonify({"Predicted_Product_Store_Sales_Total": round(max(prediction, 0.0), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Batch inference from an uploaded CSV file
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "Upload a CSV file using the 'file' form field."}), 400
        input_df = pd.read_csv(file)
        missing = [f for f in FEATURES if f not in input_df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400
        predictions = model.predict(prepare(input_df))
        return jsonify({str(i): round(max(float(p), 0.0), 2)
                        for i, p in zip(input_df.index, predictions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
