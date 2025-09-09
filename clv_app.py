"""
Customer Lifetime Value (CLV) Prediction App
==========================================
Streamlit web application for predicting customer lifetime value.

Users can input customer data and get real-time CLV predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os

# Page configuration
st.set_page_config(
    page_title="CLV Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_clv_model():
    """Load the trained CLV model and components"""
    try:
        # Check if model directory exists
        if not os.path.exists('clv_model'):
            st.error("❌ CLV model not found. Please run 'uv run python train_clv_model.py' first!")
            return None
        
        # Load model components
        with open('clv_model/clv_regressor.pkl', 'rb') as f:
            model = pickle.load(f)
            
        with open('clv_model/clv_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('clv_model/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
            
        with open('clv_model/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'metadata': metadata
        }
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def create_customer_input_form():
    """Create the customer data input form"""
    st.header("💰 Customer Lifetime Value Predictor")
    st.markdown("Enter customer information below to predict their lifetime value")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Demographics")
        
        # Age input
        age = st.number_input(
            "Age", 
            min_value=18, max_value=100, value=35,
            help="Customer's age in years"
        )
        
        # Income input
        annual_income = st.number_input(
            "Annual Income ($)", 
            min_value=0, max_value=200000, value=50000, step=5000,
            help="Customer's annual income in dollars"
        )
        
        # Children input
        total_children = st.number_input(
            "Number of Children", 
            min_value=0, max_value=10, value=1,
            help="Number of children in household"
        )
        
        # Gender selection
        gender = st.selectbox(
            "Gender", 
            ["Male", "Female"],
            help="Customer's gender"
        )
        
        # Marital status
        marital_status = st.selectbox(
            "Marital Status", 
            ["Married", "Single"],
            help="Customer's marital status"
        )
        
        # Education level
        education = st.selectbox(
            "Education Level", 
            ["High School", "Bachelors", "Graduate Degree", "Partial College"],
            help="Highest education level completed"
        )
    
    with col2:
        st.subheader("🛒 Purchase Behavior")
        
        # Total orders
        total_orders = st.number_input(
            "Total Historical Orders", 
            min_value=1, max_value=100, value=5,
            help="Total number of orders placed"
        )
        
        # Total quantity
        total_quantity = st.number_input(
            "Total Items Purchased", 
            min_value=1, max_value=1000, value=25,
            help="Total quantity of items purchased"
        )
        
        # Average order size
        avg_order_size = st.number_input(
            "Average Order Size", 
            min_value=0.1, max_value=50.0, value=5.0,
            help="Average number of items per order"
        )
        
        # Customer lifespan
        customer_lifespan_days = st.number_input(
            "Customer Lifespan (days)", 
            min_value=1, max_value=3650, value=365,
            help="Number of days as an active customer"
        )
        
        # Recency
        recency_days = st.number_input(
            "Days Since Last Purchase", 
            min_value=0, max_value=1000, value=30,
            help="Number of days since last order"
        )
    
    return {
        'age': age,
        'annual_income': annual_income,
        'total_children': total_children,
        'gender': gender,
        'marital_status': marital_status,
        'education': education,
        'total_orders': total_orders,
        'total_quantity': total_quantity,
        'avg_order_size': avg_order_size,
        'customer_lifespan_days': customer_lifespan_days,
        'recency_days': recency_days
    }

def encode_customer_data(customer_data, label_encoders):
    """Encode categorical variables"""
    # Create a copy
    encoded_data = customer_data.copy()
    
    # Encode gender
    try:
        encoded_data['gender_encoded'] = label_encoders['gender'].transform([customer_data['gender']])[0]
    except ValueError:
        # Handle unseen categories
        encoded_data['gender_encoded'] = 0
    
    # Encode marital status
    try:
        encoded_data['marital_status_encoded'] = label_encoders['marital_status'].transform([customer_data['marital_status']])[0]
    except ValueError:
        encoded_data['marital_status_encoded'] = 0
    
    # Encode education
    try:
        encoded_data['education_encoded'] = label_encoders['education'].transform([customer_data['education']])[0]
    except ValueError:
        encoded_data['education_encoded'] = 0
    
    return encoded_data

def predict_clv(customer_data, model_components):
    """Predict CLV for customer data"""
    try:
        # Encode categorical variables
        encoded_data = encode_customer_data(customer_data, model_components['label_encoders'])
        
        # Prepare feature vector
        feature_columns = model_components['metadata']['feature_columns']
        features = []
        
        for col in feature_columns:
            if col in encoded_data:
                features.append(encoded_data[col])
            else:
                features.append(0)  # Default value for missing features
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = model_components['scaler'].transform(features_array)
        
        # Predict CLV
        clv_prediction = model_components['model'].predict(features_scaled)[0]
        
        return max(0, clv_prediction)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0

def display_clv_results(clv_prediction, customer_data, model_components):
    """Display CLV prediction results with insights"""
    st.header("🎯 CLV Prediction Results")
    
    # Main prediction display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.success(f"**Predicted Customer Lifetime Value: ${clv_prediction:,.2f}**")
    
    with col2:
        # Get CLV stats from metadata
        clv_stats = model_components['metadata']['clv_stats']
        percentile = (clv_prediction / clv_stats['max']) * 100
        st.metric("CLV Percentile", f"{percentile:.1f}%")
    
    with col3:
        # Model performance
        r2_score = model_components['metadata']['performance']['r2']
        st.metric("Model Accuracy (R²)", f"{r2_score:.3f}")
    
    # CLV Category and recommendations
    st.subheader("📈 Customer Category & Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Determine customer category
        if clv_prediction > 100:
            category = "Premium Customer"
            color = "green"
            priority = "High"
        elif clv_prediction > 50:
            category = "Valuable Customer"
            color = "orange"
            priority = "Medium-High"
        elif clv_prediction > 20:
            category = "Standard Customer"
            color = "blue"
            priority = "Medium"
        else:
            category = "Developing Customer"
            color = "gray"
            priority = "Low-Medium"
        
        st.markdown(f"**Category:** :{color}[{category}]")
        st.markdown(f"**Priority Level:** {priority}")
    
    with col2:
        # CLV Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = clv_prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CLV ($)"},
            gauge = {
                'axis': {'range': [None, 200]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 50], 'color': "lightblue"},
                    {'range': [50, 100], 'color': "orange"},
                    {'range': [100, 200], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 150
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Business recommendations
    st.subheader("💡 Recommended Actions")
    
    if clv_prediction > 100:
        recommendations = [
            "🎯 Assign dedicated account manager",
            "💎 Offer premium products and exclusive deals",
            "📞 Priority customer service and support",
            "🎁 Personalized loyalty rewards program",
            "📧 VIP communication channels"
        ]
    elif clv_prediction > 50:
        recommendations = [
            "📧 Regular targeted email campaigns",
            "🎯 Cross-selling and upselling opportunities",
            "💰 Loyalty program enrollment",
            "📊 Monitor purchase patterns closely",
            "🎁 Seasonal promotional offers"
        ]
    elif clv_prediction > 20:
        recommendations = [
            "📱 Standard marketing campaigns",
            "🛒 Product recommendation system",
            "📊 Track engagement metrics",
            "💸 Competitive pricing strategies",
            "🔄 Regular check-in communications"
        ]
    else:
        recommendations = [
            "🎁 Welcome offers and incentives",
            "📚 Educational content about products",
            "💸 Special discounts to encourage purchases",
            "🔄 Re-engagement campaigns",
            "📈 Focus on increasing purchase frequency"
        ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Customer profile summary
    st.subheader("👤 Customer Profile Summary")
    
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        st.markdown(f"""
        **Demographics:**
        - Age: {customer_data['age']} years
        - Income: ${customer_data['annual_income']:,}
        - Children: {customer_data['total_children']}
        - Gender: {customer_data['gender']}
        - Marital Status: {customer_data['marital_status']}
        - Education: {customer_data['education']}
        """)
    
    with profile_col2:
        st.markdown(f"""
        **Purchase Behavior:**
        - Total Orders: {customer_data['total_orders']}
        - Items Purchased: {customer_data['total_quantity']}
        - Avg Order Size: {customer_data['avg_order_size']:.1f}
        - Customer for: {customer_data['customer_lifespan_days']} days
        - Last Purchase: {customer_data['recency_days']} days ago
        """)

def display_model_info(model_components):
    """Display model information in sidebar"""
    st.sidebar.header("🤖 Model Information")
    
    metadata = model_components['metadata']
    performance = metadata['performance']
    
    st.sidebar.markdown(f"""
    **Model Type:** {metadata['model_type']}
    **Training Date:** {metadata['training_date']}
    **Dataset Size:** {metadata['data_shape'][0]:,} customers
    
    **Performance Metrics:**
    - R² Score: {performance['r2']:.3f}
    - MAE: ${performance['mae']:.2f}
    - RMSE: ${performance['rmse']:.2f}
    
    **CLV Statistics:**
    - Min: ${metadata['clv_stats']['min']:.2f}
    - Max: ${metadata['clv_stats']['max']:.2f}
    - Mean: ${metadata['clv_stats']['mean']:.2f}
    - Median: ${metadata['clv_stats']['median']:.2f}
    """)
    
    # Feature importance
    st.sidebar.subheader("🔍 Top Features")
    feature_importance = performance['feature_importance'].head(5)
    
    for idx, row in feature_importance.iterrows():
        st.sidebar.markdown(f"**{row['feature']}**: {row['importance']:.3f}")

def main():
    """Main Streamlit application"""
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 10px;
    }
    .stSuccess {
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load model
    model_components = load_clv_model()
    if model_components is None:
        st.stop()
    
    # Display model info in sidebar
    display_model_info(model_components)
    
    # Main app
    st.markdown("""
    <div class="main-header">
        <h1>💰 Customer Lifetime Value Predictor</h1>
        <p>AI-Powered CLV Analysis for Strategic Customer Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get customer input
    customer_data = create_customer_input_form()
    
    # Prediction button
    if st.button("🔮 Predict Customer Lifetime Value", type="primary"):
        with st.spinner("Calculating CLV..."):
            clv_prediction = predict_clv(customer_data, model_components)
            
        # Display results
        display_clv_results(clv_prediction, customer_data, model_components)
    
    # Instructions
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        **Step 1:** Enter customer demographic information (age, income, family details)
        
        **Step 2:** Input purchase behavior data (orders, quantities, timing)
        
        **Step 3:** Click "Predict Customer Lifetime Value" to get:
        - Predicted CLV amount
        - Customer category and priority level
        - Specific business recommendations
        - Visual CLV gauge and metrics
        
        **Note:** This model is trained on AdventureWorks customer data and provides 
        predictions based on historical patterns and machine learning algorithms.
        """)

if __name__ == "__main__":
    main()
