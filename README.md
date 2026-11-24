# HomeValue – Housing Price Prediction System

An enterprise-grade, production-structured Machine Learning web application and econometrics benchmarking platform built with **Python**, **Flask**, **Scikit-learn**, and **Vanilla JavaScript** with **Chart.js**.

This project investigates and compares **Ordinary Least Squares (OLS)**, **Ridge Regression (L2)**, and **Lasso Regression (L1)** under conditions of acute structural multicollinearity in housing features.

---

## 🌟 Key Features

1. **Precision Valuation Engine**
   - Interactive real-time property valuation based on 6 architectural and land specifications.
   - Evaluates Linear (OLS), Ridge (L2, &alpha;=10), and Lasso (L1, &alpha;=1000) models simultaneously.
   - Live model consensus spread and dynamic comparative valuation chart.

2. **Executive Analytics Dashboard**
   - Ground-truth evaluation metrics computed across 2,000 observations (1,600 train / 400 test).
   - Side-by-side $R^2$, $RMSE$, and $MAE$ benchmark comparisons.
   - Variance Inflation Factor (VIF) diagnostic bar chart on logarithmic scale.
   - Actual vs. Predicted price scatter visualization with 45-degree identity reference line.
   - Residual distribution histogram illustrating normal zero-centered error characteristics.
   - Standardized coefficient shrinkage table showing feature elimination by Lasso.

3. **In-Depth Econometric Methodology**
   - Mathematical formulation of OLS normal equations, Tikhonov regularization, and L1 coordinate descent with soft thresholding.
   - Explanation of ill-conditioned Gram matrices $(X^T X)$ and why OLS yields negative coefficient artifacts on correlated features.
   - 5-Fold cross-validation stability analysis ($R^2 = 0.9719$ to $0.9721$).

4. **Strict Enterprise Design**
   - Designed to mimic leading PropTech and financial analytics SaaS interfaces.
   - Minimalist corporate palette: Deep Navy (`#172B45`), Slate Navy (`#233D5B`), Accent Teal (`#26877D`), and Clean Background (`#F7F8F6`).
   - Clean Inter typography, restrained card borders, and zero AI-gimmick styling.

---

## 📐 Machine Learning Specifications

| Characteristic | Specification |
|---|---|
| **Dataset Size** | 2,000 property records (80% Train, 20% Holdout Test) |
| **Target Variable** | `sale_price` |
| **Input Features** | `living_area_sqft`, `total_interior_sqft`, `footprint_sqft`, `garage_capacity_cars`, `lot_size_sqft`, `room_count_above_ground` |
| **Feature Scaling** | `StandardScaler` (Fitted strictly on training partition) |
| **Collinear Features** | `living_area_sqft` (VIF 166.5), `total_interior_sqft` (VIF 88.4), `footprint_sqft` (VIF 78.6) |
| **Ridge Hyperparameter** | $\alpha = 10$ (Stabilized Gram matrix) |
| **Lasso Hyperparameter** | $\alpha = 1000$ (Drives `footprint_sqft` coefficient to $0.0000$) |
| **Model Test $R^2$** | **0.9716** (Linear), **0.9713** (Ridge), **0.9716** (Lasso) |
| **5-Fold CV Mean $R^2$** | **0.9721** (Linear), **0.9713** (Ridge), **0.9719** (Lasso) |

---

## 📂 Project Structure

```
├── app.py                     # Production Flask web server and JSON REST APIs
├── train.py                   # Model training and analytical metrics pipeline
├── requirements.txt           # Python dependencies (Flask, Scikit-learn, etc.)
├── Procfile                   # Process configuration for Render deployment
├── runtime.txt                # Python runtime specification (python-3.12.3)
├── models/                    # Serialized joblib models and fitted scaler
│   ├── scaler.joblib
│   ├── linear_model.joblib
│   ├── ridge_model.joblib
│   ├── lasso_model.joblib
│   └── features.joblib
├── data/
│   ├── multicollinear_housing_features.csv
│   └── analytics_summary.json # Ground-truth computed metrics and chart data
├── static/
│   ├── css/
│   │   └── style.css          # PropTech corporate stylesheet
│   └── js/
│       ├── main.js            # Shared client helpers and formatters
│       ├── predict.js         # Interactive valuation engine client logic
│       └── analytics.js       # Chart.js visualization renderer
└── templates/
    ├── base.html              # Core navigation layout and footer
    ├── index.html             # Homepage and executive overview
    ├── predict.html           # Property valuation prediction engine
    ├── analytics.html         # Executive model and data analytics dashboard
    ├── methodology.html       # Mathematical foundations and VIF analysis
    └── about.html             # Project architecture and dataset disclosure
```

---

## 🚀 Local Installation & Execution

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- `pip` or `uv`

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AvanishYadav09/Housing-Price-Prediction-Ridge-Lasso.git
   cd Housing-Price-Prediction-Ridge-Lasso
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Re-train models and generate analytics:**
   ```bash
   python train.py
   ```

5. **Start the Flask web server:**
   ```bash
   python app.py
   ```
   Open your browser at `http://127.0.0.1:5000`.

---

## ☁️ Deployment on Render

This repository is configured for one-click deployment on [Render](https://render.com):

1. Log in to **Render** and click **New +** &rarr; **Web Service**.
2. Connect your GitHub repository: `Housing-Price-Prediction-Ridge-Lasso`.
3. Configure the service settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python train.py`
   - **Start Command**: `gunicorn app:app`
4. Click **Create Web Service**. Your application will be live with free SSL.

---

## 📄 License & Transparency

This project is open-source under the MIT License. The dataset consists of 2,000 synthetically modeled residential records created with controlled collinearity for educational and benchmarking purposes.
