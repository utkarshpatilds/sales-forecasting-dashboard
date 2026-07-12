# Intelligent Sales Forecasting & Inventory Optimization System

An end-to-end machine learning dashboard that predicts retail demand, detects sales anomalies, and segments products into actionable demand profiles.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-live-success)

[Live Demo](https://sales-forecasting-dashboard-utkarshpatil.streamlit.app/) | [Executive Summary](summary.pdf) | [Analysis Notebook](analysis.ipynb)

---

## The Business Problem

Retailers lose billions annually to two forecasting failures:

- Overstocking: wasted capital, dead inventory, markdowns.
- Understocking: lost sales, empty shelves, customer churn.

This project tackles both with a production-grade interactive dashboard that combines time series forecasting, unsupervised anomaly detection, and demand segmentation, packaged for non-technical business users.

Built on 4 years of historical Superstore sales data (around 10,000 orders, $2.3M revenue, 17 sub-categories, 4 regions).

---

## Live Dashboard

The deployed app features 4 interactive pages accessible from the sidebar:

| Page | What it does | Key technique |
|---|---|---|
| Sales Overview | YoY revenue, monthly trend, region/category drill-downs | Plotly aggregations + KPI filters |
| Forecast Explorer | 1-6 month demand projection per category or region | Holt-Winters Exponential Smoothing |
| Anomaly Report | Weekly sales timeline with outlier detection | Isolation Forest (5% contamination) |
| Product Demand Segments | 2D PCA visualization of sub-category clusters | K-Means + StandardScaler + PCA |

---

## Key Findings

- Champion model: Holt-Winters with additive trend and 12-month seasonality achieved about 12% MAPE on held-out monthly sales, outperforming naive baselines by 35%.
- Seasonality signal: Strong, predictable Q4 revenue spikes (Nov/Dec), confirming holiday-driven demand and validating the seasonal model choice.
- Anomalies: About 10 weeks flagged as outliers, corresponding to known retail events (Black Friday surges, bulk B2B orders). These can be excluded from baseline forecasting.
- Segmentation: 3 distinct sub-category profiles emerged: High-Volume Stable, High-Value Volatile, and Niche Low-Volume, each calling for a different inventory strategy (Continuous Replenishment vs. Just-In-Time vs. Drop-Ship).

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| App Framework | Streamlit (multi-page, cached, Plotly rendering) |
| Data Wrangling | Pandas, NumPy |
| Time Series | Statsmodels (Holt-Winters Exponential Smoothing) |
| Machine Learning | scikit-learn: IsolationForest, KMeans, StandardScaler, PCA |
| Visualization | Plotly Express, Plotly Graph Objects |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
sales-forecasting-dashboard/
|-- app.py                 # Streamlit dashboard (4 pages)
|-- requirements.txt       # Pinned Python dependencies
|-- analysis.ipynb         # Full EDA + model comparison notebook
|-- train.csv              # Superstore sales dataset (4 years)
|-- vgsales.csv            # Secondary dataset (video game sales)
|-- summary.pdf            # 2-page executive business report
|-- images/                # Exported static charts from the notebook
|-- README.md              # You are here
```

---

## Run Locally

Prerequisites: Python 3.11 or higher, and `train.csv` in the project root (already included).

```bash
# 1. Clone the repository
git clone https://github.com/utkarshpatilds/sales-forecasting-dashboard.git
cd sales-forecasting-dashboard

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
# On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Methodology Notes

### Forecasting
- Holt-Winters Exponential Smoothing with additive trend and additive 12-month seasonality.
- Chosen over Prophet for zero-dependency install and faster training on small datasets.
- Confidence band approximated from residual standard deviation (plus or minus 1.96 sigma).
- Requires at least 24 months of history per segment for stable seasonal component estimation.

### Anomaly Detection
- Weekly sales aggregation for a smoother signal than daily.
- Isolation Forest with 5% contamination, unsupervised, no labels needed.
- Outputs a binary anomaly flag plus a Plotly scatter overlay on the timeline.

### Segmentation
- Features per sub-category: Total Sales, Average Order Value, Volatility (monthly std), YoY Growth (mean of pct_change).
- StandardScaler normalization, then K-Means (k=2 to 6, user-selectable), then 2-component PCA for visualization.
- Default k=3 reveals: stable high-volume, volatile high-value, and niche low-volume clusters.

---

## Future Work

- Add Prophet as an optional secondary forecaster with a UI toggle.
- Implement MAPE and RMSE computed live on a rolling holdout window.
- Add drill-down from PCA clusters to individual order lists.
- Containerize with Docker for on-prem enterprise deployment.
- Integrate live data ingestion from a SQL warehouse.

---

## Executive Report

For a jargon-free, business-friendly breakdown aimed at stakeholders, see `summary.pdf`. It includes the 3-month financial forecast, risk caveats, and concrete supply-chain recommendations.

---

## Connect

**Utkarsh Patil** - Data Scientist

- LinkedIn: https://www.linkedin.com/in/utkarshpatil-ds/
- Email: utkarshpatil432@gmail.com
- GitHub: https://github.com/utkarshpatilds

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
