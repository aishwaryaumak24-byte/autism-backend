import json
import os
import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.json")

app = Flask(__name__)
CORS(app)

MODEL = joblib.load(MODEL_PATH)


def load_columns():
    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        columns = raw
    elif isinstance(raw, dict):
        columns = (
            raw.get("columns")
            or raw.get("data_columns")
            or raw.get("feature_names")
        )
    else:
        columns = None

    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "columns.json must be a list or contain one of: columns, data_columns, feature_names"
        )

    return columns


COLUMNS = load_columns()


def to_number(value):
    if value is None or value == "":
        return 0.0

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        v = value.strip().lower()
        mapping = {
            "yes": 1.0,
            "y": 1.0,
            "true": 1.0,
            "male": 1.0,
            "m": 1.0,
            "no": 0.0,
            "n": 0.0,
            "false": 0.0,
            "female": 0.0,
            "f": 0.0,
        }
        if v in mapping:
            return mapping[v]
        try:
            return float(v)
        except ValueError:
            return 0.0

    return 0.0


@app.get("/health")
def health():
    return jsonify({"ok": True, "expected_features": len(COLUMNS)})


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True) or {}
        payload = data.get("features") if isinstance(data.get("features"), dict) else data

        row = [to_number(payload.get(column, 0)) for column in COLUMNS]
        X = np.array([row], dtype=float)

        prediction = int(MODEL.predict(X)[0])

        probability = None
        if hasattr(MODEL, "predict_proba"):
            try:
                probs = MODEL.predict_proba(X)[0]
                probability = float(probs[1] if len(probs) > 1 else probs[0])
            except Exception:
                probability = None

        return jsonify(
            {
                "prediction": prediction,
                "probability": probability,
                "expected_features": len(COLUMNS),
                "received_features": len(payload),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "expected_features": len(COLUMNS)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
