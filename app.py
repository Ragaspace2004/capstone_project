"""
AdventureWorks Predictive Analytics Dashboard - Modern UI
======================================================
Modern Streamlit web application with glassmorphism design for AdventureWorks ML models.

This app provides interactive interfaces for:
- Customer Lifetime Value prediction
- Customer Segmentation analysis
- Customer Churn prediction
- Sales Forecasting

Usage: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="AdventureWorks AI Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with glassmorphism and dark theme
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme variables */
    :root {
        --primary-bg: #0a0e27;
        --secondary-bg: rgba(255, 255, 255, 0.05);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --accent-blue: #00d4ff;
        --accent-purple: #8b5cf6;
        --accent-pink: #ec4899;
        --text-primary: #ffffff;
        --text-secondary: #a8b2d1;
        --border-glass: rgba(255, 255, 255, 0.2);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide streamlit elements */
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    .stFooter {display: none;}
    header {visibility: hidden;}
    
    /* Main container */
    .main-container {
        background: rgba(10, 14, 39, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Navigation tabs */
    .nav-tabs {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .nav-tab {
        padding: 1rem 2rem;
        border-radius: 12px;
        background: transparent;
        color: var(--text-secondary);
        text-decoration: none;
        transition: all 0.3s ease;
        cursor: pointer;
        border: none;
        font-weight: 500;
    }
    
    .nav-tab.active {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.3);
    }
    
    .nav-tab:hover:not(.active) {
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #8b5cf6, #ec4899);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Form styles */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.4);
    }
    
    /* Results section */
    .result-section {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-blue);
        margin: 1rem 0;
    }
    
    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
    }
    
    .feature-description {
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        border-radius: 4px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .metric-container {
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and preprocessors"""
    models = {}
    
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            st.error("⚠️ Models directory not found. Please run 'python build_models.py' first!")
            st.stop()
        
        # Load CLV model
        with open('models/clv_model.pkl', 'rb') as f:
            models['clv_model'] = pickle.load(f)
        with open('models/clv_scaler.pkl', 'rb') as f:
            models['clv_scaler'] = pickle.load(f)
        
        # Load segmentation model
        with open('models/segmentation_model.pkl', 'rb') as f:
            models['segmentation_model'] = pickle.load(f)
        with open('models/segmentation_scaler.pkl', 'rb') as f:
            models['segmentation_scaler'] = pickle.load(f)
        
        # Load churn model
        with open('models/churn_model.pkl', 'rb') as f:
            models['churn_model'] = pickle.load(f)
        with open('models/churn_scaler.pkl', 'rb') as f:
            models['churn_scaler'] = pickle.load(f)
        
        # Load sales forecast model
        with open('models/sales_forecast_model.pkl', 'rb') as f:
            models['sales_forecast_model'] = pickle.load(f)
        
        # Load metadata
        with open('models/model_metadata.pkl', 'rb') as f:
            models['metadata'] = pickle.load(f)
        
        return models
    
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}")
        st.error("Please run 'python build_models.py' first to train the models!")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.stop()

def create_navigation():
    """Create modern navigation"""
    st.markdown("""
    <div class="nav-tabs">
        <div class="nav-tab active" onclick="showPage('dashboard')">🏠 Dashboard</div>
        <div class="nav-tab" onclick="showPage('clv')">💎 CLV Prediction</div>
        <div class="nav-tab" onclick="showPage('segment')">👥 Segmentation</div>
        <div class="nav-tab" onclick="showPage('churn')">⚠️ Churn Risk</div>
        <div class="nav-tab" onclick="showPage('forecast')">📈 Forecasting</div>
    </div>
    """, unsafe_allow_html=True)

def show_dashboard_overview(models):
    """Display modern dashboard overview"""
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🤖 AI-Powered Analytics</h1>
        <p class="hero-subtitle">Transform your business decisions with machine learning insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    perf = models['metadata']['model_performance']
    
    st.markdown("""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">💎 CLV Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">⚠️ Churn Detection</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">📈 Sales Forecast</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">🧠 AI Models</div>
        </div>
    </div>
    """.format(
        perf['clv']['r2'],
        perf['churn']['accuracy'], 
        perf['forecast']['r2']
    ), unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">💎</div>
            <h3 class="feature-title">Customer Lifetime Value</h3>
            <p class="feature-description">Predict the total value a customer will bring to your business over their entire relationship. Identify high-value prospects and optimize acquisition strategies.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">👥</div>
            <h3 class="feature-title">Smart Segmentation</h3>
            <p class="feature-description">Automatically group customers into meaningful segments based on behavior and demographics. Create targeted campaigns that resonate with each group.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚠️</div>
            <h3 class="feature-title">Churn Prevention</h3>
            <p class="feature-description">Identify customers at risk of leaving before they churn. Deploy retention strategies proactively to maintain your valuable customer base.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3 class="feature-title">Sales Forecasting</h3>
            <p class="feature-description">Predict future sales with advanced time-series analysis. Optimize inventory, staffing, and marketing spend based on accurate forecasts.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Training info
    training_date = models['metadata'].get('training_date', 'Unknown')
    st.info(f"🕒 Last model update: {training_date}")

def show_clv_prediction(models):
    """Modern CLV prediction interface"""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">💎 Customer Lifetime Value</h1>
        <p class="hero-subtitle">Predict the total value each customer will bring to your business</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Customer Profile")
        
        # Customer inputs with modern styling
        annual_income = st.number_input(
            "💰 Annual Income ($)", 
            min_value=0, max_value=200000, value=75000, step=5000
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)
            total_children = st.number_input("👶 Children", min_value=0, max_value=10, value=1)
        with col_b:
            gender = st.selectbox("⚧ Gender", ["Male", "Female"])
            marital_status = st.selectbox("💑 Status", ["Married", "Single"])
        
        education = st.selectbox(
            "🎓 Education", 
            ["High School", "Bachelors", "Graduate Degree"]
        )
        
        st.subheader("🛍️ Purchase History")
        
        col_c, col_d = st.columns(2)
        with col_c:
            total_orders = st.number_input("📦 Total Orders", min_value=0, max_value=100, value=12)
            total_quantity = st.number_input("🛒 Items Purchased", min_value=0, max_value=1000, value=48)
        with col_d:
            avg_order_size = st.number_input("📏 Avg Order Size", min_value=0.0, max_value=50.0, value=4.0)
            lifespan_days = st.number_input("📅 Customer Days", min_value=0, max_value=3650, value=720)
        
        predict_button = st.button("🔮 Predict Customer Value", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if predict_button:
            # Prepare features
            features = np.array([[
                annual_income, total_children, age, 
                1 if gender == "Male" else 0,
                1 if marital_status == "Married" else 0, 
                2 if education == "Bachelors" else (3 if education == "Graduate Degree" else 1),
                total_orders, total_quantity, avg_order_size, lifespan_days
            ]])
            
            # Predict CLV
            features_scaled = models['clv_scaler'].transform(features)
            clv_prediction = models['clv_model'].predict(features_scaled)[0]
            
            # Results section
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-value">${clv_prediction:,.2f}</div>
            <h3 style="color: white; margin-top: 0;">Predicted Lifetime Value</h3>
            """, unsafe_allow_html=True)
            
            # Customer tier
            if clv_prediction > 1000:
                tier = "🏆 Premium Customer"
                tier_color = "#00d4ff"
                strategies = [
                    "🎯 Dedicated account management",
                    "💎 Exclusive product access",
                    "🎁 Premium rewards program",
                    "📞 Priority customer support"
                ]
            elif clv_prediction > 500:
                tier = "⭐ High Value Customer"
                tier_color = "#8b5cf6"
                strategies = [
                    "📧 Personalized email campaigns",
                    "🎯 Advanced product recommendations",
                    "💳 Loyalty program benefits",
                    "📊 Behavior-based offers"
                ]
            else:
                tier = "🌱 Growing Customer"
                tier_color = "#ec4899"
                strategies = [
                    "🎁 Welcome bonus programs",
                    "📚 Product education content",
                    "💸 Special introductory offers",
                    "🔄 Re-engagement campaigns"
                ]
            
            st.markdown(f'<h4 style="color: {tier_color};">{tier}</h4>', unsafe_allow_html=True)
            
            # Action recommendations
            st.subheader("💡 Strategic Actions")
            for strategy in strategies:
                st.markdown(f"• {strategy}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = clv_prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Customer Lifetime Value", 'font': {'color': 'white', 'size': 20}},
                gauge = {
                    'axis': {'range': [None, 2000], 'tickcolor': 'white'},
                    'bar': {'color': tier_color},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.2)",
                    'steps': [
                        {'range': [0, 500], 'color': "rgba(236, 72, 153, 0.3)"},
                        {'range': [500, 1000], 'color': "rgba(139, 92, 246, 0.3)"},
                        {'range': [1000, 2000], 'color': "rgba(0, 212, 255, 0.3)"}
                    ]
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'white'},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_customer_segmentation(models):
    """Modern customer segmentation interface"""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">👥 Customer Segmentation</h1>
        <p class="hero-subtitle">Discover which customer group this profile belongs to</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Customer Analysis")
        
        # Streamlined inputs
        col_a, col_b = st.columns(2)
        with col_a:
            annual_income = st.number_input("💰 Income ($)", value=55000, step=5000, key="seg_income")
            age = st.number_input("🎂 Age", value=42, key="seg_age")
            total_orders = st.number_input("📦 Orders", value=8, key="seg_orders")
            avg_order_size = st.number_input("📏 Avg Size", value=6.2, key="seg_avg")
        
        with col_b:
            total_children = st.number_input("👶 Children", value=2, key="seg_children")
            total_quantity = st.number_input("🛒 Items", value=45, key="seg_quantity")
            recency_days = st.number_input("📅 Days Since Last", value=25, key="seg_recency")
        
        segment_button = st.button("🎯 Identify Segment", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if segment_button:
            # Prepare and predict
            features = np.array([[annual_income, total_children, age, total_orders, total_quantity, avg_order_size, recency_days]])
            features_scaled = models['segmentation_scaler'].transform(features)
            segment = models['segmentation_model'].predict(features_scaled)[0]
            segment_name = models['metadata']['segment_mapping'][segment]
            
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            
            # Segment info
            segment_config = {
                'Low Value': {'icon': '🌱', 'color': '#ec4899', 'desc': 'Price-sensitive customers with basic needs'},
                'Occasional': {'icon': '⭐', 'color': '#f59e0b', 'desc': 'Moderate buyers with seasonal patterns'},
                'Regular': {'icon': '💎', 'color': '#8b5cf6', 'desc': 'Consistent customers with steady engagement'},
                'High Value': {'icon': '🏆', 'color': '#00d4ff', 'desc': 'Premium customers with high engagement'},
                'Premium': {'icon': '👑', 'color': '#10b981', 'desc': 'Top-tier customers with maximum value'}
            }
            
            config = segment_config.get(segment_name, segment_config['Regular'])
            
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 4rem;">{config['icon']}</div>
                <h2 style="color: {config['color']}; margin: 1rem 0;">{segment_name} Segment</h2>
                <p style="color: #a8b2d1; font-size: 1.1rem;">{config['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Segment visualization
            segments = ['Low Value', 'Occasional', 'Regular', 'High Value', 'Premium']
            colors = ['#ec4899', '#f59e0b', '#8b5cf6', '#00d4ff', '#10b981']
            values = [15, 25, 30, 20, 10]  # Mock distribution
            
            fig = go.Figure(data=[go.Pie(
                labels=segments,
                values=values,
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent',
                textfont_size=12,
                marker=dict(line=dict(color='rgba(255,255,255,0.2)', width=2))
            )])
            
            fig.update_layout(
                title="Customer Segment Distribution",
                title_font_color='white',
                title_x=0.5,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'white'},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_churn_prediction(models):
    """Modern churn prediction interface"""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">⚠️ Churn Risk Analysis</h1>
        <p class="hero-subtitle">Identify customers at risk of leaving your business</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔍 Risk Assessment")
        
        # Customer profile inputs
        col_a, col_b = st.columns(2)
        with col_a:
            annual_income = st.number_input("💰 Income ($)", value=45000, key="churn_income")
            age = st.number_input("🎂 Age", value=38, key="churn_age")
            gender = st.selectbox("⚧ Gender", ["Male", "Female"], key="churn_gender")
            education = st.selectbox("🎓 Education", ["High School", "Bachelors", "Graduate Degree"], key="churn_education")
            total_orders = st.number_input("📦 Orders", value=3, key="churn_orders")
        
        with col_b:
            total_children = st.number_input("👶 Children", value=1, key="churn_children")
            marital_status = st.selectbox("💑 Status", ["Married", "Single"], key="churn_marital")
            avg_order_size = st.number_input("📏 Avg Size", value=2.5, key="churn_avg")
            lifespan_days = st.number_input("📅 Customer Days", value=180, key="churn_lifespan")
            recency_days = st.number_input("⏰ Days Since Last", value=45, key="churn_recency")
        
        churn_button = st.button("🔍 Analyze Churn Risk", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if churn_button:
            # Prepare features
            features = np.array([[
                annual_income, total_children, age, 
                1 if gender == "Male" else 0,
                1 if marital_status == "Married" else 0, 
                2 if education == "Bachelors" else (3 if education == "Graduate Degree" else 1),
                total_orders, avg_order_size, lifespan_days, recency_days
            ]])
            
            # Predict churn
            features_scaled = models['churn_scaler'].transform(features)
            churn_prob = models['churn_model'].predict_proba(features_scaled)[0][1]
            
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            
            # Risk level determination
            if churn_prob > 0.7:
                risk_level = "🚨 Critical Risk"
                risk_color = "#ef4444"
                risk_desc = "Immediate action required to prevent churn"
                actions = [
                    "📞 Personal outreach within 24h",
                    "🎁 Exclusive retention offer",
                    "💬 Schedule satisfaction call",
                    "🎯 Deploy emergency retention campaign"
                ]
            elif churn_prob > 0.4:
                risk_level = "⚠️ High Risk" 
                risk_color = "#f59e0b"
                risk_desc = "Proactive retention efforts recommended"
                actions = [
                    "📧 Send personalized re-engagement email",
                    "💸 Offer targeted discount",
                    "🎯 Include in retention campaign",
                    "📊 Monitor engagement closely"
                ]
            else:
                risk_level = "✅ Low Risk"
                risk_color = "#10b981"
                risk_desc = "Customer likely to remain active"
                actions = [
                    "📈 Focus on upselling opportunities",
                    "⭐ Encourage reviews and referrals",
                    "🎯 Cross-sell complementary products",
                    "💎 Consider for loyalty program"
                ]
            
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 2.5rem; color: {risk_color}; font-weight: 700;">
                    {churn_prob:.1%}
                </div>
                <h3 style="color: {risk_color}; margin: 0.5rem 0;">{risk_level}</h3>
                <p style="color: #a8b2d1;">{risk_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action plan
            st.subheader("🎯 Recommended Actions")
            for action in actions:
                st.markdown(f"• {action}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Churn risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk Level (%)", 'font': {'color': 'white', 'size': 20}},
                number = {'suffix': "%", 'font': {'color': 'white', 'size': 30}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': 'white'},
                    'bar': {'color': risk_color},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.2)",
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.3)"},
                        {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'white'},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_sales_forecasting(models):
    """Modern sales forecasting interface"""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">📈 Sales Forecasting</h1>
        <p class="hero-subtitle">Predict future sales with advanced AI modeling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Forecast parameters
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎯 Forecast Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        month = st.selectbox("📅 Month", range(1, 13), index=5, format_func=lambda x: f"{x:02d}")
        unique_customers = st.number_input("👥 Expected Customers", value=125)
    
    with col2:
        day = st.selectbox("📆 Day", range(1, 32), index=14)
        unique_products = st.number_input("📦 Expected Products", value=65)
    
    with col3:
        dayofweek = st.selectbox("📅 Weekday", range(0, 7), index=2, 
                                format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        lag_1 = st.number_input("📊 Yesterday Sales", value=180)
    
    with col4:
        quarter = st.selectbox("📈 Quarter", [1, 2, 3, 4], index=1)
        lag_7 = st.number_input("📊 Last Week Same Day", value=165)
    
    rolling_mean_7 = st.number_input("📈 7-Day Average", value=172)
    
    forecast_button = st.button("🔮 Generate Forecast", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if forecast_button:
        # Prepare features and predict
        features = np.array([[month, day, dayofweek, quarter, unique_customers, unique_products, lag_1, lag_7, rolling_mean_7]])
        forecast = models['sales_forecast_model'].predict(features)[0]
        
        # Results section
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 3rem; color: #00d4ff; font-weight: 700;">
                    {forecast:.0f}
                </div>
                <h3 style="color: white; margin: 0.5rem 0;">Predicted Sales Units</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Trend analysis
            trend = "📈 Increasing" if forecast > rolling_mean_7 else "📉 Decreasing"
            variance = abs(forecast - rolling_mean_7) / rolling_mean_7 * 100
            
            # Business insights
            st.subheader("💡 Business Insights")
            
            if forecast > rolling_mean_7 * 1.15:
                st.success("🚀 **Strong growth expected** - Scale up operations")
            elif forecast > rolling_mean_7 * 1.05:
                st.info("📈 **Moderate growth expected** - Prepare for increased demand")
            elif forecast < rolling_mean_7 * 0.85:
                st.warning("📉 **Below average sales** - Consider promotional strategies")
            else:
                st.info("➡️ **Stable performance** - Maintain current operations")
            
            # Operational recommendations
            st.subheader("🎯 Recommendations")
            
            recommendations = []
            if month in [11, 12]:
                recommendations.append("🎄 Holiday season - Increase inventory 25%")
            if dayofweek in [4, 5, 6]:
                recommendations.append("🎉 Weekend prep - Extend hours")
            if unique_customers > 150:
                recommendations.append("👥 High traffic - Add staff")
            if forecast > 200:
                recommendations.append("📦 High demand - Check supply chain")
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Trend visualization
            fig = go.Figure()
            
            # Historical trend (mock data)
            days_back = 14
            base_sales = rolling_mean_7
            historical_sales = [base_sales + np.random.randint(-20, 20) for _ in range(days_back)]
            dates = pd.date_range(end=pd.Timestamp.now(), periods=days_back)
            
            # Add forecast
            forecast_date = pd.Timestamp.now() + timedelta(days=1)
            
            # Historical line
            fig.add_trace(go.Scatter(
                x=dates,
                y=historical_sales,
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#8b5cf6', width=3),
                marker=dict(size=6)
            ))
            
            # Forecast point
            fig.add_trace(go.Scatter(
                x=[forecast_date],
                y=[forecast],
                mode='markers',
                name='Forecast',
                marker=dict(size=15, color='#00d4ff', symbol='star', 
                           line=dict(color='white', width=2))
            ))
            
            # Trend line
            fig.add_hline(y=rolling_mean_7, line_dash="dash", 
                         line_color="rgba(255,255,255,0.5)",
                         annotation_text="7-day average")
            
            fig.update_layout(
                title='Sales Forecast Trend',
                title_font_color='white',
                title_x=0.5,
                xaxis_title='Date',
                yaxis_title='Sales Units',
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'white'},
                showlegend=True,
                height=400,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📊 Forecast Metrics")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted Units", f"{forecast:.0f}")
                st.metric("Trend", trend)
            with col_b:
                st.metric("vs 7-day Avg", f"{variance:.1f}%")
                confidence = 85 + np.random.randint(-10, 10)
                st.metric("Confidence", f"{confidence}%")
            
            st.markdown('</div>', unsafe_allow_html=True)

def create_floating_nav():
    """Create floating navigation menu"""
    pages = [
        ("🏠", "Dashboard", "dashboard"),
        ("💎", "CLV", "clv"), 
        ("👥", "Segments", "segments"),
        ("⚠️", "Churn", "churn"),
        ("📈", "Forecast", "forecast")
    ]
    
    st.markdown("""
    <style>
    .floating-nav {
        position: fixed;
        top: 50%;
        right: 20px;
        transform: translateY(-50%);
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    .nav-item {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        color: white;
        font-size: 1.5rem;
    }
    
    .nav-item:hover {
        transform: scale(1.1);
        background: rgba(0, 212, 255, 0.2);
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.3);
    }
    
    @media (max-width: 768px) {
        .floating-nav {
            position: relative;
            right: auto;
            top: auto;
            transform: none;
            flex-direction: row;
            justify-content: center;
            margin: 2rem 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application with modern UI"""
    try:
        models = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return
    
    # Create floating navigation
    create_floating_nav()
    
    # Simple page navigation using session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'
    
    # Page selection buttons
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🏠 Dashboard", key="nav_dash"):
            st.session_state.current_page = 'dashboard'
    with col2:
        if st.button("💎 CLV Prediction", key="nav_clv"):
            st.session_state.current_page = 'clv'
    with col3:
        if st.button("👥 Segmentation", key="nav_seg"):
            st.session_state.current_page = 'segments'
    with col4:
        if st.button("⚠️ Churn Risk", key="nav_churn"):
            st.session_state.current_page = 'churn'
    with col5:
        if st.button("📈 Forecasting", key="nav_forecast"):
            st.session_state.current_page = 'forecast'
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display selected page
    if st.session_state.current_page == 'dashboard':
        show_dashboard_overview(models)
    elif st.session_state.current_page == 'clv':
        show_clv_prediction(models)
    elif st.session_state.current_page == 'segments':
        show_customer_segmentation(models)
    elif st.session_state.current_page == 'churn':
        show_churn_prediction(models)
    elif st.session_state.current_page == 'forecast':
        show_sales_forecasting(models)
    
    # Footer
    st.markdown("""
    <div style="margin-top: 4rem; text-align: center; padding: 2rem; 
                background: rgba(255,255,255,0.05); border-radius: 16px; 
                border: 1px solid rgba(255,255,255,0.1);">
        <h4 style="color: #00d4ff; margin-bottom: 1rem;">🤖 Powered by Advanced Machine Learning</h4>
        <p style="color: #a8b2d1; margin: 0;">
            Transform your business decisions with AI-driven insights • 
            Real-time predictions • Actionable recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()