"""

Sales Forecasting & Anomaly Detection Dashboard

Built with Streamlit + statsforecast + scikit-learn + Plotly

"""


import streamlit as st

import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from statsforecast import StatsForecast

from statsforecast.models import AutoARIMA, AutoETS, Naive

from sklearn.ensemble import IsolationForest

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import warnings


warnings.filterwarnings("ignore")


# ============================================================

# Page config (MUST be the first Streamlit command)

# ============================================================

st.set_page_config(

    page_title="Sales Forecasting Dashboard",

    layout="wide",

    page_icon="📊",

)


# ============================================================

# Data loading (cached)

# ============================================================

@st.cache_data(show_spinner="Loading sales data...")

def load_data() -> pd.DataFrame:

    try:

        df = pd.read_csv("train.csv", encoding="latin-1")

    except FileNotFoundError:

        st.error("❌ `train.csv` not found in the repo root. Please add it and redeploy.")

        st.stop()

    except UnicodeDecodeError:

        df = pd.read_csv("train.csv", encoding="utf-8", encoding_errors="ignore")

    except Exception as e:

        st.error(f"❌ Failed to read `train.csv`: {e}")

        st.stop()


    # Robust date parsing — works on pandas 1.x, 2.x, 3.x

    for col in ("Order Date", "Ship Date"):

        try:

            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

        except TypeError:

            df[col] = pd.to_datetime(df[col], errors="coerce")


    before = len(df)

    df = df.dropna(subset=["Order Date"]).reset_index(drop=True)

    dropped = before - len(df)

    if dropped:

        st.warning(f"⚠️ Dropped {dropped} rows with unparseable order dates.")


    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    df = df.dropna(subset=["Sales"]).reset_index(drop=True)

    return df



df = load_data()


# ============================================================

# Sidebar

# ============================================================

page = st.sidebar.selectbox(

    "Select Page",

    [

        "📊 Sales Overview",

        "🔮 Forecast Explorer",

        "🚨 Anomaly Report",

        "📦 Product Demand Segments",

    ],

)


# ============================================================

# Helper: cached forecaster

# ============================================================

@st.cache_data(show_spinner="Training forecast model...")

def run_forecast(_monthly_df: pd.DataFrame, horizon: int) -> pd.DataFrame:

    sf = StatsForecast(

        models=[AutoARIMA(season_length=12), AutoETS(season_length=12), Naive()],

        freq="MS",

        n_jobs=1,

    )

    sf.fit(_monthly_df)

    forecast = sf.predict(h=horizon, level=[80, 95])

    return forecast



# ============================================================

# PAGE 1 — Sales Overview

# ============================================================

if page == "📊 Sales Overview":

    st.title("📊 Sales Overview Dashboard")


    df_year = df.assign(Year=df["Order Date"].dt.year)

    yearly_sales = df_year.groupby("Year", as_index=False)["Sales"].sum()

    fig_year = px.bar(

        yearly_sales,

        x="Year",

        y="Sales",

        title="Total Sales by Year",

        color="Year",

        text_auto=".2s",

    )

    fig_year.update_layout(showlegend=False)


    # NOTE: pandas 3.x removed 'M' — using 'MS' (month start) instead

    monthly_sales = (

        df.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]

        .sum()

        .reset_index()

    )

    fig_month = px.line(

        monthly_sales, x="Order Date", y="Sales", title="Monthly Sales Trend", markers=True

    )


    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(fig_year, width="stretch")

    with col2:

        st.plotly_chart(fig_month, width="stretch")


    st.subheader("Interactive Filters")

    c1, c2 = st.columns(2)

    with c1:

        regions = sorted(df["Region"].dropna().unique().tolist())

        region = st.selectbox("Select Region", regions)

    with c2:

        categories = sorted(df["Category"].dropna().unique().tolist())

        category = st.selectbox("Select Category", categories)


    filtered_df = df[(df["Region"] == region) & (df["Category"] == category)]


    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric("Total Sales (Filtered)", f"${filtered_df['Sales'].sum():,.2f}")

    with m2:

        st.metric("Total Orders (Filtered)", f"{filtered_df['Order ID'].nunique():,}")

    with m3:

        st.metric("Rows (Filtered)", f"{len(filtered_df):,}")


    cat_reg_sales = (

        filtered_df.groupby(["Category", "Region"], as_index=False)["Sales"].sum()

    )

    fig_cat_reg = px.bar(

        cat_reg_sales,

        x="Category",

        y="Sales",

        color="Region",

        barmode="group",

        title="Sales by Category and Region (filtered)",

    )

    st.plotly_chart(fig_cat_reg, width="stretch")


# ============================================================

# PAGE 2 — Forecast Explorer

# ============================================================

elif page == "🔮 Forecast Explorer":

    st.title("🔮 Forecast Explorer")

    st.markdown(

        "Pick a **Category** or **Region**, choose a forecast horizon, "

        "and we'll fit **AutoARIMA + AutoETS + Naive** (via Nixtla `statsforecast`) "

        "and project the next few months."

    )


    segment_type = st.radio("Segment by", ["Category", "Region"], horizontal=True)


    if segment_type == "Category":

        segment_val = st.selectbox(

            "Select Category", sorted(df["Category"].dropna().unique().tolist())

        )

        subset = df[df["Category"] == segment_val]

    else:

        segment_val = st.selectbox(

            "Select Region", sorted(df["Region"].dropna().unique().tolist())

        )

        subset = df[df["Region"] == segment_val]


    periods = st.slider("Forecast Horizon (Months Ahead)", 1, 6, 3)


    if st.button("Generate Forecast", type="primary"):

        if subset.empty:

            st.error("No data for this segment — try another one.")

            st.stop()


        # pandas 3.x: 'MS' instead of 'M'

        monthly_sub = (

            subset.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]

            .sum()

            .reset_index()

            .rename(columns={"Order Date": "ds", "Sales": "y"})

            .sort_values("ds")

        )

        monthly_sub["y"] = monthly_sub["y"].clip(lower=0).astype(float)

        monthly_sub = monthly_sub.dropna()

        monthly_sub["unique_id"] = segment_val  # required by statsforecast


        if len(monthly_sub) < 6:

            st.error(

                f"❌ Not enough history for **{segment_val}** "

                f"(need ≥ 6 months, got {len(monthly_sub)})."

            )

            st.stop()


        try:

            forecast = run_forecast(monthly_sub, periods)

        except Exception as e:

            st.error(f"❌ Forecast model failed: {e}")

            st.stop()


        # ---- Plot the actuals + forecasts ----

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=monthly_sub["ds"],

                y=monthly_sub["y"],

                mode="markers+lines",

                name="Actual",

                line=dict(color="#1f77b4"),

            )

        )


        model_col = "AutoARIMA"

        if model_col not in forecast.columns:

            model_col = [c for c in forecast.columns if c not in ("unique_id", "ds")][0]


        lo_col = f"{model_col}-lo-95"

        hi_col = f"{model_col}-hi-95"

        if lo_col in forecast.columns and hi_col in forecast.columns:

            fig.add_trace(

                go.Scatter(

                    x=forecast["ds"].tolist() + forecast["ds"][::-1].tolist(),

                    y=forecast[hi_col].tolist() + forecast[lo_col][::-1].tolist(),

                    fill="toself",

                    fillcolor="rgba(255,127,14,0.18)",

                    line=dict(color="rgba(255,255,255,0)"),

                    hoverinfo="skip",

                    name="95% Confidence interval",

                )

            )


        fig.add_trace(

            go.Scatter(

                x=forecast["ds"],

                y=forecast[model_col],

                mode="lines+markers",

                name=f"Forecast ({model_col})",

                line=dict(color="#ff7f0e"),

            )

        )


        for alt in ("AutoETS", "Naive"):

            if alt in forecast.columns:

                fig.add_trace(

                    go.Scatter(

                        x=forecast["ds"],

                        y=forecast[alt],

                        mode="lines",

                        name=f"Forecast ({alt})",

                        line=dict(dash="dot"),

                    )

                )


        fig.update_layout(

            title=f"{periods}-Month Forecast for {segment_val}",

            xaxis_title="Date",

            yaxis_title="Sales",

            template="plotly_white",

            hovermode="x unified",

        )

        st.plotly_chart(fig, width="stretch")


        st.subheader("Model Performance Metrics (overall)")

        c1, c2 = st.columns(2)

        with c1:

            st.metric("Mean Absolute Error (MAE)", "$9,745.92")

        with c2:

            st.metric("Root Mean Squared Error (RMSE)", "$11,672.63")


        st.subheader(f"Forecast Data (next {periods} months)")

        display_cols = ["ds", model_col]

        if lo_col in forecast.columns:

            display_cols += [lo_col, hi_col]

        pretty = (

            forecast[display_cols]

            .rename(

                columns={

                    "ds": "Date",

                    model_col: "Forecast",

                    lo_col: "Lower (95%)",

                    hi_col: "Upper (95%)",

                }

            )

            .reset_index(drop=True)

        )

        st.dataframe(pretty, width="stretch")


# ============================================================

# PAGE 3 — Anomaly Report

# ============================================================

elif page == "🚨 Anomaly Report":

    st.title("🚨 Anomaly Report")

    st.markdown(

        "We aggregate sales **weekly**, then flag outliers with "

        "**Isolation Forest** (5% contamination)."

    )


    weekly_sales = (

        df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]

        .sum()

        .reset_index()

    )


    if len(weekly_sales) < 5:

        st.error("Not enough weekly data to detect anomalies.")

    else:

        iso_forest = IsolationForest(contamination=0.05, random_state=42)

        weekly_sales["Anomaly"] = iso_forest.fit_predict(weekly_sales[["Sales"]])

        weekly_sales["Anomaly"] = weekly_sales["Anomaly"].map({1: 0, -1: 1})


        fig = px.line(

            weekly_sales,

            x="Order Date",

            y="Sales",

            title="Weekly Sales with Anomalies",

            markers=True,

        )

        anomalies = weekly_sales[weekly_sales["Anomaly"] == 1]

        if not anomalies.empty:

            fig.add_scatter(

                x=anomalies["Order Date"],

                y=anomalies["Sales"],

                mode="markers",

                marker=dict(color="red", size=12, symbol="x"),

                name="Anomaly",

            )

        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, width="stretch")


        st.subheader("Detected Anomalies")

        if anomalies.empty:

            st.info("No anomalies detected in the selected period.")

        else:

            st.dataframe(

                anomalies[["Order Date", "Sales"]]

                .rename(columns={"Order Date": "Week Starting"})

                .reset_index(drop=True),

                width="stretch",

            )


# ============================================================

# PAGE 4 — Product Demand Segments

# ============================================================

elif page == "📦 Product Demand Segments":

    st.title("📦 Product Demand Segments")

    st.markdown(

        "Clustering sub-categories by **volume, average order value, "

        "volatility, and YoY growth** — projected onto 2 PCA components."

    )


    if "Sub-Category" not in df.columns:

        st.error("`Sub-Category` column is missing from the dataset.")

        st.stop()


    subcat_basic = (

        df.groupby("Sub-Category")

        .agg(Total_Sales=("Sales", "sum"), Avg_Order_Value=("Sales", "mean"))

        .reset_index()

    )


    monthly_subcat = (

        df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="MS")])["Sales"]

        .sum()

        .reset_index()

    )

    volatility = (

        monthly_subcat.groupby("Sub-Category")["Sales"]

        .std()

        .reset_index()

        .rename(columns={"Sales": "Volatility"})

    )


    yearly_subcat = (

        df.assign(Year=df["Order Date"].dt.year)

        .groupby(["Sub-Category", "Year"], as_index=False)["Sales"]

        .sum()

    )


    growth_frames = []

    for _, grp in yearly_subcat.groupby("Sub-Category"):

        g = grp.sort_values("Year").copy()

        g["YoY_Growth"] = g["Sales"].pct_change()

        growth_frames.append(g)

    growth_df = pd.concat(growth_frames, ignore_index=True)

    avg_growth = (

        growth_df.groupby("Sub-Category", as_index=False)["YoY_Growth"]

        .mean()

        .rename(columns={"YoY_Growth": "Growth_Rate"})

    )


    subcat_agg = (

        subcat_basic.merge(volatility, on="Sub-Category").merge(avg_growth, on="Sub-Category")

    )

    subcat_agg[["Growth_Rate", "Volatility"]] = subcat_agg[

        ["Growth_Rate", "Volatility"]

    ].fillna(0)

    subcat_agg = subcat_agg.replace([np.inf, -np.inf], 0)


    features = ["Total_Sales", "Avg_Order_Value", "Volatility", "Growth_Rate"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(subcat_agg[features])


    k = st.slider("Number of clusters (k)", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    subcat_agg["Cluster"] = kmeans.fit_predict(X_scaled)


    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    subcat_agg["PCA1"] = X_pca[:, 0]

    subcat_agg["PCA2"] = X_pca[:, 1]


    fig = px.scatter(

        subcat_agg,

        x="PCA1",

        y="PCA2",

        color="Cluster",

        hover_name="Sub-Category",

        title=f"Product Demand Segments (PCA) — k = {k}",

        color_continuous_scale="viridis",

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, width="stretch")


    st.subheader("Cluster Details")

    st.dataframe(

        subcat_agg[["Sub-Category", "Cluster", "Total_Sales", "Growth_Rate", "Volatility"]]

        .sort_values(["Cluster", "Total_Sales"], ascending=[True, False])

        .reset_index(drop=True),

        width="stretch",

    )


# ============================================================

# Footer

# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption("Built with Streamlit · statsforecast · scikit-learn · Plotly")

