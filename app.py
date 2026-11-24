"""
HomeValue - Housing Price Prediction System
Flask Production Web Application & REST API
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load Serialized Models and Scaler
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    linear_model = joblib.load(os.path.join(MODELS_DIR, "linear_model.joblib"))
    ridge_model = joblib.load(os.path.join(MODELS_DIR, "ridge_model.joblib"))
    lasso_model = joblib.load(os.path.join(MODELS_DIR, "lasso_model.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))
except Exception as e:
    print(f"Warning: Model artifacts not found or could not be loaded: {e}")
    scaler = None
    linear_model = None
    ridge_model = None
    lasso_model = None
    feature_names = [
        "living_area_sqft",
        "total_interior_sqft",
        "footprint_sqft",
        "garage_capacity_cars",
        "lot_size_sqft",
        "room_count_above_ground"
    ]

# Load Analytics Summary Data
ANALYTICS_PATH = os.path.join(DATA_DIR, "analytics_summary.json")
analytics_data = {}
if os.path.exists(ANALYTICS_PATH):
    with open(ANALYTICS_PATH, "r", encoding="utf-8") as f:
        analytics_data = json.load(f)

# Feature metadata and validation constraints
FEATURE_CONSTRAINTS = {
    "living_area_sqft": {"min": 500.0, "max": 6000.0, "default": 2400.0, "label": "Living Area (sq. ft.)"},
    "total_interior_sqft": {"min": 500.0, "max": 6500.0, "default": 2420.0, "label": "Total Interior Area (sq. ft.)"},
    "footprint_sqft": {"min": 300.0, "max": 4500.0, "default": 1800.0, "label": "Footprint Area (sq. ft.)"},
    "garage_capacity_cars": {"min": 0, "max": 5, "default": 2, "label": "Garage Capacity (cars)"},
    "lot_size_sqft": {"min": 1000.0, "max": 15000.0, "default": 5500.0, "label": "Lot Size (sq. ft.)"},
    "room_count_above_ground": {"min": 1, "max": 12, "default": 5, "label": "Rooms Above Ground"}
}

# --- PAGE ROUTES ---

@app.route("/")
def index():
    """Homepage: Executive Overview, Problem Statement, and Product Highlights."""
    meta = analytics_data.get("metadata", {})
    evals = analytics_data.get("test_evaluations", {})
    return render_template("index.html", metadata=meta, evaluations=evals, current_page="home")

@app.route("/predict")
def predict_page():
    """Interactive Property Valuation Engine."""
    return render_template(
        "predict.html",
        constraints=FEATURE_CONSTRAINTS,
        current_page="predict"
    )

@app.route("/analytics")
def analytics_page():
    """Comprehensive Model & Data Analytics Dashboard."""
    return render_template(
        "analytics.html",
        analytics=analytics_data,
        current_page="analytics"
    )

@app.route("/methodology")
def methodology_page():
    """Technical and Mathematical ML Foundations."""
    return render_template(
        "methodology.html",
        analytics=analytics_data,
        current_page="methodology"
    )

@app.route("/about")
def about_page():
    """Project Overview, Architecture, and Synthetic Data Transparency."""
    meta = analytics_data.get("metadata", {})
    return render_template("about.html", metadata=meta, current_page="about")

# --- JSON REST APIS ---

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict property price across Linear, Ridge, and Lasso regression models.
    Validates input payloads, scales features using training StandardScaler,
    and returns multi-model valuations with comparative variance.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON payload provided."}), 400

        selected_model_name = data.get("model", "Ridge Regression")
        
        # Validate inputs against bounds
        raw_values = []
        validation_errors = []
        
        for feat in feature_names:
            val = data.get(feat)
            if val is None or val == "":
                validation_errors.append(f"Missing required field: {FEATURE_CONSTRAINTS[feat]['label']}")
                continue
            
            try:
                val_float = float(val)
                c = FEATURE_CONSTRAINTS[feat]
                if val_float < c["min"] or val_float > c["max"]:
                    validation_errors.append(
                        f"{c['label']} must be between {c['min']} and {c['max']}. Received: {val_float}"
                    )
                raw_values.append(val_float)
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid numeric value for {FEATURE_CONSTRAINTS[feat]['label']}")

        if validation_errors:
            return jsonify({
                "status": "error",
                "message": "Input validation failed.",
                "errors": validation_errors
            }), 422

        if scaler is None or linear_model is None or ridge_model is None or lasso_model is None:
            return jsonify({
                "status": "error",
                "message": "Model artifacts are not loaded on server."
            }), 500

        # Transform inputs using the fitted training scaler
        input_array = np.array(raw_values).reshape(1, -1)
        scaled_features = scaler.transform(input_array)

        # Run predictions across all three models
        pred_linear = float(linear_model.predict(scaled_features)[0])
        pred_ridge = float(ridge_model.predict(scaled_features)[0])
        pred_lasso = float(lasso_model.predict(scaled_features)[0])

        all_predictions = {
            "Linear Regression": round(pred_linear, 2),
            "Ridge Regression": round(pred_ridge, 2),
            "Lasso Regression": round(pred_lasso, 2)
        }

        # Select target prediction
        if selected_model_name in all_predictions:
            active_prediction = all_predictions[selected_model_name]
        else:
            selected_model_name = "Ridge Regression"
            active_prediction = all_predictions[selected_model_name]

        # Calculate variance / spread among models
        pred_list = list(all_predictions.values())
        spread = round(max(pred_list) - min(pred_list), 2)
        mean_val = round(float(np.mean(pred_list)), 2)

        return jsonify({
            "status": "success",
            "selected_model": selected_model_name,
            "valuation": active_prediction,
            "mean_valuation": mean_val,
            "model_spread": spread,
            "comparisons": all_predictions,
            "inputs": {
                feat: {
                    "label": FEATURE_CONSTRAINTS[feat]["label"],
                    "value": raw_values[i]
                }
                for i, feat in enumerate(feature_names)
            }
        })

    except Exception as ex:
        return jsonify({
            "status": "error",
            "message": f"Server encountered an error processing prediction: {str(ex)}"
        }), 500

@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """Return all computed ground-truth ML metrics and analytics summary."""
    if not analytics_data:
        return jsonify({"status": "error", "message": "Analytics metadata not available."}), 404
    return jsonify({"status": "success", "data": analytics_data})

@app.route("/api/features", methods=["GET"])
def api_features():
    """Return feature definitions, boundaries, and default values."""
    return jsonify({"status": "success", "features": FEATURE_CONSTRAINTS})

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint for platform monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "HomeValue Housing Price Prediction System",
        "version": "1.0.0",
        "models_loaded": scaler is not None
    })

if __name__ == "__main__":
    # Local development server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
