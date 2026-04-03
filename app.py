import os
import json
import pickle
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MDL_DIR   = os.path.join(BASE_DIR, "models")

# ── Load models and data ──────────────────────────────────────
print("Loading scalers...")
with open(os.path.join(MDL_DIR, "all_scalers.pkl"), "rb") as f:
    all_scalers = pickle.load(f)

print("Loading sensor metadata...")
with open(os.path.join(MDL_DIR, "sensor_metadata.json"), "r") as f:
    sensor_metadata = json.load(f)

print("Loading Random Forest model...")
rf_model = joblib.load(os.path.join(MDL_DIR, "random_forest_best.pkl"))

print("Loading LSTM model...")
import tensorflow as tf
keras_path = os.path.join(MDL_DIR, "lstm_best.keras")
h5_path    = os.path.join(MDL_DIR, "lstm_best.h5")
if os.path.exists(keras_path):
    lstm_model = tf.keras.models.load_model(keras_path)
    print("  Loaded: lstm_best.keras")
else:
    lstm_model = tf.keras.models.load_model(h5_path)
    print("  Loaded: lstm_best.h5")

sensor_ids = list(sensor_metadata.keys())
print(f"Ready — {len(sensor_ids)} sensors loaded")

# ── Prediction log ────────────────────────────────────────────
prediction_log = []

# ── Helper: condition badge ───────────────────────────────────
def get_condition(speed_mph):
    if speed_mph >= 50:
        return "Free Flow"
    elif speed_mph >= 30:
        return "Moderate"
    else:
        return "Congested"

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sensors")
def api_sensors():
    return jsonify(sensor_metadata)

@app.route("/api/stats")
def api_stats():
    counts = {"free_flow": 0, "moderate": 0, "congested": 0}
    for sid in sensor_ids:
        scaler = all_scalers.get(sid)
        if scaler is None:
            continue
        dummy = np.zeros((1, 6))
        dummy[0, 0] = 0.7
        speed_mph = float(scaler.inverse_transform(dummy)[0, 0])
        cond = get_condition(speed_mph)
        if cond == "Free Flow":
            counts["free_flow"] += 1
        elif cond == "Moderate":
            counts["moderate"] += 1
        else:
            counts["congested"] += 1
    return jsonify(counts)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data      = request.get_json()
        sensor_id = str(data["sensor_id"])
        model_type = data.get("model", "lstm")
        speeds    = data["speeds"]          # list of 12 floats (raw mph)

        if len(speeds) != 12:
            return jsonify({"error": "Exactly 12 speed values required"}), 400

        scaler = all_scalers.get(sensor_id)
        if scaler is None:
            return jsonify({"error": f"Sensor {sensor_id} not found"}), 404

        now = datetime.utcnow()
        sin_time   = float(np.sin(2 * np.pi * (now.hour * 60 + now.minute) / 5 / 288))
        cos_time   = float(np.cos(2 * np.pi * (now.hour * 60 + now.minute) / 5 / 288))
        sin_day    = float(np.sin(2 * np.pi * now.weekday() / 7))
        cos_day    = float(np.cos(2 * np.pi * now.weekday() / 7))
        is_weekend = float(now.weekday() >= 5)

        if model_type == "lstm":
            features = np.array([[s, sin_time, cos_time, sin_day, cos_day, is_weekend]
                                  for s in speeds], dtype=np.float32)
            scaled   = scaler.transform(features)
            X        = scaled.reshape(1, 12, 6)
            pred_scaled = float(lstm_model.predict(X, verbose=0)[0][0])
            dummy    = np.zeros((1, 6))
            dummy[0, 0] = pred_scaled
            pred_mph = float(scaler.inverse_transform(dummy)[0, 0])

        else:
            features = np.array([[s, sin_time, cos_time, sin_day, cos_day, is_weekend]
                                  for s in speeds], dtype=np.float32)
            scaled   = scaler.transform(features)
            X_flat   = np.hstack([scaled[:, 0], scaled[-1, 1:]])
            pred_scaled = float(rf_model.predict(X_flat.reshape(1, -1))[0])
            dummy    = np.zeros((1, 6))
            dummy[0, 0] = pred_scaled
            pred_mph = float(scaler.inverse_transform(dummy)[0, 0])

        pred_mph  = round(max(0.0, min(pred_mph, 80.0)), 2)
        condition = get_condition(pred_mph)

        log_entry = {
            "timestamp" : datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_id" : sensor_id,
            "model"     : model_type.upper(),
            "speed_mph" : pred_mph,
            "condition" : condition
        }
        prediction_log.append(log_entry)

        return jsonify({
            "sensor_id" : sensor_id,
            "model"     : model_type.upper(),
            "speed_mph" : pred_mph,
            "condition" : condition
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload/csv", methods=["POST"])
def api_upload_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file    = request.files["file"]
        content = file.read().decode("utf-8")
        lines   = [l.strip() for l in content.strip().split("\n") if l.strip()]
        header  = [h.strip() for h in lines[0].split(",")]

        results = []
        errors  = []

        for i, line in enumerate(lines[1:], start=2):
            try:
                row       = dict(zip(header, [v.strip() for v in line.split(",")]))
                sensor_id = str(row.get("sensor_id", "")).strip()
                model_type = str(row.get("model", "lstm")).strip().lower()
                speed_cols = [k for k in header if k.startswith("speed_")]
                speeds    = [float(row[k]) for k in speed_cols]

                if len(speeds) != 12:
                    errors.append(f"Row {i}: need 12 speed columns")
                    continue

                scaler = all_scalers.get(sensor_id)
                if scaler is None:
                    errors.append(f"Row {i}: sensor {sensor_id} not found")
                    continue

                now = datetime.utcnow()
                sin_time   = float(np.sin(2 * np.pi * (now.hour * 60 + now.minute) / 5 / 288))
                cos_time   = float(np.cos(2 * np.pi * (now.hour * 60 + now.minute) / 5 / 288))
                sin_day    = float(np.sin(2 * np.pi * now.weekday() / 7))
                cos_day    = float(np.cos(2 * np.pi * now.weekday() / 7))
                is_weekend = float(now.weekday() >= 5)

                features = np.array([[s, sin_time, cos_time, sin_day, cos_day, is_weekend]
                                      for s in speeds], dtype=np.float32)
                scaled   = scaler.transform(features)

                if model_type == "lstm":
                    X = scaled.reshape(1, 12, 6)
                    pred_scaled = float(lstm_model.predict(X, verbose=0)[0][0])
                else:
                    X_flat = np.hstack([scaled[:, 0], scaled[-1, 1:]])
                    pred_scaled = float(rf_model.predict(X_flat.reshape(1, -1))[0])

                dummy = np.zeros((1, 6))
                dummy[0, 0] = pred_scaled
                pred_mph  = round(float(max(0.0, min(scaler.inverse_transform(dummy)[0, 0], 80.0))), 2)
                condition = get_condition(pred_mph)

                entry = {
                    "timestamp" : now.strftime("%Y-%m-%d %H:%M:%S"),
                    "sensor_id" : sensor_id,
                    "model"     : model_type.upper(),
                    "speed_mph" : pred_mph,
                    "condition" : condition
                }
                prediction_log.append(entry)
                results.append(entry)

            except Exception as ex:
                errors.append(f"Row {i}: {str(ex)}")

        return jsonify({"predictions": results, "errors": errors,
                        "total": len(results)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/log")
def api_log():
    return jsonify(prediction_log[-100:])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
