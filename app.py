import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", page_icon="📊")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='mixed')
    return df

df = load_data()

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", [
    "📊 Sales Overview", 
    "🔮 Forecast Explorer", 
    "🚨 Anomaly Report", 
    "📦 Product Demand Segments"
])

# ==========================================
# PAGE 1: Sales Overview Dashboard
# ==========================================
if page == "📊 Sales Overview":
    st.title("📊 Sales Overview Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        df['Year'] = df['Order Date'].dt.year
        yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()
        fig_year = px.bar(yearly_sales, x='Year', y='Sales', title='Total Sales by Year', color='Year')
        st.plotly_chart(fig_year, use_container_width=True)
        
    with col2:
        monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
        fig_month = px.line(monthly_sales, x='Order Date', y='Sales', title='Monthly Sales Trend (4 Years)')
        st.plotly_chart(fig_month, use_container_width=True)
        
    st.subheader("Interactive Filters")
    c1, c2 = st.columns(2)
    with c1: region = st.selectbox("Select Region", df['Region'].unique())
    with c2: category = st.selectbox("Select Category", df['Category'].unique())
        
    filtered_df = df[(df['Region'] == region) & (df['Category'] == category)]
    
    col3, col4 = st.columns(2)
    with col3: st.metric("Total Sales (Filtered)", f"${filtered_df['Sales'].sum():,.2f}")
    with col4: st.metric("Total Orders (Filtered)", f"{filtered_df['Order ID'].nunique():,}")
        
    cat_reg_sales = filtered_df.groupby(['Category', 'Region'])['Sales'].sum().reset_index()
    fig_cat_reg = px.bar(cat_reg_sales, x='Category', y='Sales', color='Region', barmode='group', title='Sales by Category and Region')
    st.plotly_chart(fig_cat_reg, use_container_width=True)

# ==========================================
# PAGE 2: Forecast Explorer
# ==========================================
elif page == "🔮 Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.markdown("Select a segment and forecast horizon to see the predicted sales using our champion model (**Facebook Prophet**).")
    
    segment_type = st.selectbox("Select Segment Type", ["Category", "Region"])
    if segment_type == "Category":
        segment_val = st.selectbox("Select Category", df['Category'].unique())
        subset = df[df['Category'] == segment_val]
    else:
        segment_val = st.selectbox("Select Region", df['Region'].unique())
        subset = df[df['Region'] == segment_val]
        
    periods = st.slider("Forecast Horizon (Months Ahead)", 1, 6, 3)
    
    if st.button("Generate Forecast"):
        with st.spinner("Training model and forecasting..."):
            monthly_sub = subset.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
            df_p = monthly_sub.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            
        fig = model.plot(forecast)
        plt = fig.gca()
        plt.set_title(f"{periods}-Month Forecast for {segment_val}")
        st.pyplot(fig)
        
        st.subheader("Model Performance Metrics (Overall)")
        col1, col2 = st.columns(2)
        with col1: st.metric("Mean Absolute Error (MAE)", "$9,745.92")
        with col2: st.metric("Root Mean Squared Error (RMSE)", "$11,672.63")
            
        st.subheader("Forecast Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

# ==========================================
# PAGE 3: Anomaly Report
# ==========================================
elif page == "🚨 Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.markdown("Detecting unusual sales patterns using **Isolation Forest**.")
    
    weekly_sales = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    weekly_sales['Anomaly'] = iso_forest.fit_predict(weekly_sales[['Sales']])
    weekly_sales['Anomaly'] = weekly_sales['Anomaly'].map({1: 0, -1: 1})
    
    fig = px.line(weekly_sales, x='Order Date', y='Sales', title='Weekly Sales with Anomalies')
    anomalies = weekly_sales[weekly_sales['Anomaly'] == 1]
    fig.add_scatter(x=anomalies['Order Date'], y=anomalies['Sales'], mode='markers', 
                    marker=dict(color='red', size=10, symbol='x'), name='Anomaly')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Detected Anomalies")
    st.dataframe(anomalies[['Order Date', 'Sales']].rename(columns={'Order Date': 'Week Starting'}).reset_index(drop=True))

# ==========================================
# PAGE 4: Product Demand Segments
# ==========================================
elif page == "📦 Product Demand Segments":
    st.title("📦 Product Demand Segments")
    st.markdown("Clustering sub-categories based on sales volume, volatility, and growth rate.")
    
    subcat_basic = df.groupby('Sub-Category').agg(Total_Sales=('Sales', 'sum'), Avg_Order_Value=('Sales', 'mean')).reset_index()
    monthly_subcat = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='M')])['Sales'].sum().reset_index()
    volatility = monthly_subcat.groupby('Sub-Category')['Sales'].std().reset_index().rename(columns={'Sales': 'Volatility'})
    
    yearly_subcat = df.groupby(['Sub-Category', df['Order Date'].dt.year])['Sales'].sum().reset_index()
    yearly_subcat.rename(columns={'Order Date': 'Year'}, inplace=True)
    growth_rates = []
    for subcat in yearly_subcat['Sub-Category'].unique():
        subcat_data = yearly_subcat[yearly_subcat['Sub-Category'] == subcat].sort_values('Year')
        subcat_data['YoY_Growth'] = subcat_data['Sales'].pct_change()
        growth_rates.append(subcat_data)
    growth_df = pd.concat(growth_rates)
    avg_growth = growth_df.groupby('Sub-Category')['YoY_Growth'].mean().reset_index().rename(columns={'YoY_Growth': 'Growth_Rate'})
    
    subcat_agg = subcat_basic.merge(volatility, on='Sub-Category').merge(avg_growth, on='Sub-Category')
    subcat_agg['Growth_Rate'] = subcat_agg['Growth_Rate'].fillna(0)
    subcat_agg['Volatility'] = subcat_agg['Volatility'].fillna(0)
    
    features = ['Total_Sales', 'Avg_Order_Value', 'Volatility', 'Growth_Rate']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subcat_agg[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    subcat_agg['Cluster'] = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    subcat_agg['PCA1'] = X_pca[:, 0]
    subcat_agg['PCA2'] = X_pca[:, 1]
    
    fig = px.scatter(subcat_agg, x='PCA1', y='PCA2', color='Cluster', hover_name='Sub-Category',
                     title='Product Demand Segments (PCA)', color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Cluster Details")
    st.dataframe(subcat_agg[['Sub-Category', 'Cluster', 'Total_Sales', 'Growth_Rate', 'Volatility']].sort_values('Cluster'))
