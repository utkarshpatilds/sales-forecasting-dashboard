# 📊 Intelligent Sales Forecasting & Inventory Optimization System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![Prophet](https://img.shields.io/badge/Forecasting-Facebook%20Prophet-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **An end-to-end machine learning and time-series forecasting system built to predict retail product demand, detect sales anomalies, and optimize inventory stocking strategies for supply chain managers.**

---

## 📑 Executive Overview
In the retail and e-commerce sector, inaccurate demand forecasting leads to millions in lost revenue due to overstocking (wasted capital) or understocking (lost sales). 

This project tackles this real-world industry challenge by analyzing 4 years of historical Superstore sales data. It combines **Time Series Analysis**, **Machine Learning**, **Anomaly Detection**, and **Unsupervised Clustering** to build a predictive engine. The entire system is packaged into an interactive **Streamlit Dashboard** for non-technical business stakeholders to make data-driven stocking decisions.

🔗 **[🌐 Click Here to View the Live Interactive Dashboard](#)** *(Replace # with your Streamlit Community Cloud link)*

---

## 💼 Business Value & Key Findings
* **Champion Model Selection:** Evaluated SARIMA, Facebook Prophet, and XGBoost. **Facebook Prophet** emerged as the production-ready champion, achieving the lowest error rate (**12.06% MAPE**) by effectively capturing complex holiday seasonality.
* **Seasonality & Trends:** Identified massive, predictable Q4 revenue spikes (Nov/Dec) and identified the **West Region** as the most consistent growth market.
* **Anomaly Detection:** Utilized Isolation Forest to flag 11 major historical outliers (e.g., Black Friday surges, bulk B2B orders), allowing the business to separate true anomalies from baseline seasonal trends.
* **Inventory Optimization:** Applied K-Means clustering to segment products into 3 distinct demand profiles (e.g., *"High-Value Volatile Stars"* vs. *"Stable Everyday Items"*), recommending tailored Just-In-Time (JIT) vs. Continuous Replenishment stocking strategies.

---

## 🛠️ Tech Stack
| Category | Technologies |
| :--- | :--- |
| **Core Languages** | Python 3.x |
| **Data Manipulation** | Pandas, NumPy |
| **Time Series & ML** | Statsmodels (SARIMA), Facebook Prophet, XGBoost, Scikit-Learn |
| **Data Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Streamlit (Interactive Web Dashboard) |
| **Environment** | Jupyter Notebook, Google Colab, GitHub |

---

## 📂 Project Structure
```text
SalesForecasting_[YourName]/
│
├── analysis.ipynb         # Complete Jupyter Notebook (EDA, Modeling, Clustering)
├── app.py                 # Streamlit Dashboard source code
├── requirements.txt       # Python dependencies for the dashboard
├── summary.pdf            # 2-Page Executive Business Report for Stakeholders
├── train.csv              # Primary Dataset (Superstore Sales - 4 Years)
├── vgsales.csv            # Secondary Dataset (Video Game Sales - for merging practice)
│
└── images/                # Exported visualizations (Decomposition, Forecasts, PCA)
    ├── decomposition.png
    ├── model_comparison.png
    ├── anomaly_detection.png
    └── demand_segments.png


4. Explore the Analysis
Open analysis.ipynb in Jupyter Notebook or Google Colab to see the step-by-step data science pipeline, model training, and mathematical evaluations.
📊 Dashboard Features
The deployed Streamlit app features 4 interactive pages:
📊 Sales Overview: High-level KPIs, YoY bar charts, and interactive regional/category filters.
🔮 Forecast Explorer: Dynamic 1-to-6 month ahead forecasting using the Prophet model for any selected Category or Region.
🚨 Anomaly Report: Visual timeline of weekly sales with Isolation Forest outliers highlighted.
📦 Product Demand Segments: 2D PCA visualization of product sub-categories clustered by volatility, volume, and growth rate.
📄 Executive Report
For business leaders who prefer high-level insights over code, please refer to summary.pdf. It contains a jargon-free breakdown of the 3-month financial forecast, risk limitations, and concrete supply chain recommendations.
🤝 Connect with Me
If you have any questions about this project or want to discuss supply chain analytics, feel free to reach out!
Name: [Utkarsh Patil]
LinkedIn: [https://www.linkedin.com/in/utkarshpatil-ds/]
Email: [utkarshpatil432@gmail.com]

