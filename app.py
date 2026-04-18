from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("autism_model.pkl")
with open("columns.json") as f:
    COLUMNS = json.load(f)

@app.route("/")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", data)
    row = [features.get(col, 0) for col in COLUMNS]
    X = np.array([row])

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

    return jsonify({"prediction": pred, "probability": proba})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
