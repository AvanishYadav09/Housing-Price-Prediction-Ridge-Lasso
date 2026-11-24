"""
HomeValue - Housing Price Prediction System
Model Training and Analytics Generation Script

This script trains Linear, Ridge, and Lasso regression models on the exact
dataset and parameters established in the project notebook. It generates and
serializes the trained models and exports ground-truth analytical metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

def calculate_vif(X, feature_names):
    """Calculate Variance Inflation Factor (VIF) for each feature."""
    n, p = X.shape
    vif_dict = {}
    
    for j in range(p):
        y_feature = X[:, j]
        X_other = np.delete(X, j, axis=1)
        X_bias = np.column_stack([np.ones(n), X_other])
        
        beta = np.linalg.lstsq(X_bias, y_feature, rcond=None)[0]
        y_pred = X_bias @ beta
        
        ss_res = np.sum((y_feature - y_pred) ** 2)
        ss_tot = np.sum((y_feature - y_feature.mean()) ** 2)
        
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if (1.0 - r2) > 1e-10 else 999999.0
        vif_dict[feature_names[j]] = round(float(vif), 2)
        
    return vif_dict

def cross_validate_models(X, y, folds=5, seed=42):
    """5-Fold Cross Validation replicating the notebook methodology."""
    indices = np.random.default_rng(seed).permutation(len(X))
    fold_indices = np.array_split(indices, folds)
    
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=10),
        "lasso": Lasso(alpha=1000, max_iter=10000)
    }
    
    cv_results = {
        "Linear Regression": {"r2": [], "rmse": [], "mae": []},
        "Ridge Regression": {"r2": [], "rmse": [], "mae": []},
        "Lasso Regression": {"r2": [], "rmse": [], "mae": []}
    }
    
    for i in range(folds):
        test_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(folds) if j != i])
        
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        for key, name in [("linear", "Linear Regression"), ("ridge", "Ridge Regression"), ("lasso", "Lasso Regression")]:
            clf = models[key]
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_te_s)
            
            r2 = r2_score(y_te, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            mae = float(mean_absolute_error(y_te, y_pred))
            
            cv_results[name]["r2"].append(r2)
            cv_results[name]["rmse"].append(rmse)
            cv_results[name]["mae"].append(mae)
            
    summary = {}
    for name in cv_results:
        summary[name] = {
            "mean_r2": round(float(np.mean(cv_results[name]["r2"])), 4),
            "std_r2": round(float(np.std(cv_results[name]["r2"])), 4),
            "mean_rmse": round(float(np.mean(cv_results[name]["rmse"])), 2),
            "mean_mae": round(float(np.mean(cv_results[name]["mae"])), 2),
            "fold_r2": [round(float(v), 4) for v in cv_results[name]["r2"]],
            "fold_rmse": [round(float(v), 2) for v in cv_results[name]["rmse"]]
        }
    return summary

def main():
    print("=" * 60)
    print("HomeValue - Training Pipeline")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "multicollinear_housing_features.csv")
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    
    features = [
        "living_area_sqft",
        "total_interior_sqft",
        "footprint_sqft",
        "garage_capacity_cars",
        "lot_size_sqft",
        "room_count_above_ground"
    ]
    target = "sale_price"
    
    feature_display_names = {
        "living_area_sqft": "Living Area (sq. ft.)",
        "total_interior_sqft": "Total Interior Area (sq. ft.)",
        "footprint_sqft": "Footprint Area (sq. ft.)",
        "garage_capacity_cars": "Garage Capacity (cars)",
        "lot_size_sqft": "Lot Size (sq. ft.)",
        "room_count_above_ground": "Room Count (above ground)"
    }
    
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    
    # Train / Test Split as per Notebook (80/20, seed 42)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit Models with Notebook Parameters
    linear_model = LinearRegression()
    ridge_model = Ridge(alpha=10)
    lasso_model = Lasso(alpha=1000, max_iter=10000, random_state=42)
    
    linear_model.fit(X_train_scaled, y_train)
    ridge_model.fit(X_train_scaled, y_train)
    lasso_model.fit(X_train_scaled, y_train)
    
    # Evaluate Models on Test Set
    models_dict = {
        "Linear Regression": linear_model,
        "Ridge Regression": ridge_model,
        "Lasso Regression": lasso_model
    }
    
    test_evaluations = {}
    predictions_map = {}
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test_scaled)
        predictions_map[name] = y_pred
        
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        
        test_evaluations[name] = {
            "r2": round(float(r2), 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2)
        }
        print(f"{name:20s} | Test R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        
    # Cross Validation
    cv_summary = cross_validate_models(X, y, folds=5, seed=42)
    print("\n5-Fold Cross Validation Results:")
    for name, s in cv_summary.items():
        print(f"{name:20s} | Mean R2: {s['mean_r2']} | Mean RMSE: {s['mean_rmse']}")
        
    # VIF Calculation
    vif_results = calculate_vif(X_train_scaled, features)
    print("\nVariance Inflation Factors (Multicollinearity Check):")
    for feat, v in vif_results.items():
        print(f"{feat:25s}: {v}")
        
    # Coefficients comparison table
    coefficients = {
        "features": features,
        "feature_labels": [feature_display_names[f] for f in features],
        "Linear": [round(float(c), 4) for c in linear_model.coef_],
        "Ridge": [round(float(c), 4) for c in ridge_model.coef_],
        "Lasso": [round(float(c), 4) for c in lasso_model.coef_],
        "intercepts": {
            "Linear": round(float(linear_model.intercept_), 2),
            "Ridge": round(float(ridge_model.intercept_), 2),
            "Lasso": round(float(lasso_model.intercept_), 2)
        }
    }
    
    # Correlation Matrix
    corr_matrix = df[features + [target]].corr().round(4).to_dict()
    
    # Dataset statistics
    stats = {}
    for col in features + [target]:
        series = df[col]
        stats[col] = {
            "name": feature_display_names.get(col, "Sale Price"),
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "q25": round(float(series.quantile(0.25)), 2),
            "median": round(float(series.median()), 2),
            "q75": round(float(series.quantile(0.75)), 2),
            "max": round(float(series.max()), 2)
        }
        
    # Scatter plot sample points (Actual vs Predicted for test set, sampled 120 points for clean rendering)
    scatter_indices = np.linspace(0, len(y_test) - 1, 120, dtype=int)
    scatter_data = []
    for idx in scatter_indices:
        scatter_data.append({
            "actual": round(float(y_test[idx]), 2),
            "pred_linear": round(float(predictions_map["Linear Regression"][idx]), 2),
            "pred_ridge": round(float(predictions_map["Ridge Regression"][idx]), 2),
            "pred_lasso": round(float(predictions_map["Lasso Regression"][idx]), 2)
        })
        
    # Residual Distribution (Histogram bins for Ridge and Lasso)
    residuals_ridge = y_test - predictions_map["Ridge Regression"]
    hist, bin_edges = np.histogram(residuals_ridge, bins=20)
    residual_hist = {
        "bin_labels": [f"{round(bin_edges[i]/1000):+0.0f}k" for i in range(len(hist))],
        "counts": [int(c) for c in hist]
    }
    
    # Save Model Artifacts
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(linear_model, os.path.join(models_dir, "linear_model.joblib"))
    joblib.dump(ridge_model, os.path.join(models_dir, "ridge_model.joblib"))
    joblib.dump(lasso_model, os.path.join(models_dir, "lasso_model.joblib"))
    joblib.dump(features, os.path.join(models_dir, "features.joblib"))
    
    # Save Analytics Summary JSON
    analytics_payload = {
        "metadata": {
            "total_records": len(df),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "features_count": len(features),
            "model_count": 3,
            "ridge_alpha": 10,
            "lasso_alpha": 1000,
            "best_cv_model": "Lasso Regression",
            "best_test_r2": max([e["r2"] for e in test_evaluations.values()]),
            "synthetic_disclaimer": "The housing records used are synthetically modeled with controlled multicollinearity to benchmark OLS against L1 and L2 regularization."
        },
        "features": features,
        "feature_display_names": feature_display_names,
        "test_evaluations": test_evaluations,
        "cv_summary": cv_summary,
        "vif_results": vif_results,
        "coefficients": coefficients,
        "correlation_matrix": corr_matrix,
        "feature_statistics": stats,
        "scatter_sample": scatter_data,
        "residual_distribution": residual_hist
    }
    
    analytics_path = os.path.join(data_dir, "analytics_summary.json")
    with open(analytics_path, "w", encoding="utf-8") as f:
        json.dump(analytics_payload, f, indent=2)
        
    print(f"\nSaved models to: {models_dir}")
    print(f"Saved analytics summary to: {analytics_path}")
    print("=" * 60)
    print("Training pipeline completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
