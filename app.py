from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
import traceback
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("autism_model.pkl")

with open("columns.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, list):
    COLUMNS = raw
elif isinstance(raw, dict):
    COLUMNS = (
        raw.get("columns")
        or raw.get("data_columns")
        or raw.get("feature_names")
        or raw.get("features")
        or []
    )
else:
    COLUMNS = []

if not COLUMNS:
    raise ValueError("Could not read feature columns from columns.json")


def to_number(value):
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["yes", "true", "1", "male", "m"]:
            return 1
        if v in ["no", "false", "0", "female", "f"]:
            return 0
        try:
            return float(v)
        except ValueError:
            return 0

    return 0


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "feature_count": len(COLUMNS),
        "sample_columns": COLUMNS[:5]
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}

        if isinstance(data.get("features"), dict):
            features = data["features"]
        else:
            features = data

        row = {col: to_number(features.get(col, 0)) for col in COLUMNS}
        X = pd.DataFrame([row], columns=COLUMNS)

        pred = int(model.predict(X)[0])

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])

        return jsonify({
            "prediction": pred,
            "probability": proba
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
