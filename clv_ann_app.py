"""
Customer Lifetime Value (CLV) Prediction - Neural Network Interface
==================================================================
Streamlit app for testing CLV predictions using Artificial Neural Network model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import os
import tensorflow as tf
from tensorflow import keras

# Set page config
st.set_page_config(
    page_title="CLV ANN Predictor",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_ann_model():
    """Load the trained ANN model and preprocessors"""
    model_dir = 'clv_ann_model'
    
    try:
        # Load TensorFlow model
        model = keras.models.load_model(os.path.join(model_dir, 'clv_ann_model.h5'))
        
        # Load scaler
        with open(os.path.join(model_dir, 'clv_ann_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoders
        with open(os.path.join(model_dir, 'ann_label_encoders.pkl'), 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(model_dir, 'ann_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, label_encoders, metadata
    except Exception as e:
        st.error(f"Error loading ANN model: {e}")
        return None, None, None, None

def create_gauge_chart(value, title, max_value=10):
    """Create a gauge chart for CLV display"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': max_value/2},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value*0.3], 'color': 'lightgray'},
                {'range': [max_value*0.3, max_value*0.7], 'color': 'yellow'},
                {'range': [max_value*0.7, max_value], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value*0.8
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def main():
    st.title("🧠 Customer Lifetime Value Prediction")
    st.subheader("Artificial Neural Network Model with Regularization")
    
    # Load model
    model, scaler, label_encoders, metadata = load_ann_model()
    
    if model is None:
        st.error("❌ Could not load ANN model. Please train the model first by running `train_ann_model.py`")
        return
    
    # Display model info
    with st.expander("🔍 Model Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "Deep Neural Network")
            st.metric("R² Score", f"{metadata['performance']['r2']:.3f}")
        with col2:
            st.metric("MAE", f"${metadata['performance']['mae']:.2f}")
            st.metric("RMSE", f"${metadata['performance']['rmse']:.2f}")
        with col3:
            st.metric("Epochs Trained", metadata['performance']['epochs_trained'])
            st.metric("Training Date", metadata['training_date'][:10])
        
        st.write("**Regularization Techniques Used:**")
        reg_techniques = metadata['regularization']
        cols = st.columns(5)
        techniques = [
            ("Dropout", reg_techniques['dropout']),
            ("L2 Regularization", reg_techniques['l2_regularization']),
            ("Batch Normalization", reg_techniques['batch_normalization']),
            ("Early Stopping", reg_techniques['early_stopping']),
            ("LR Reduction", reg_techniques['learning_rate_reduction'])
        ]
        
        for i, (name, enabled) in enumerate(techniques):
            with cols[i]:
                st.success(f"✅ {name}" if enabled else f"❌ {name}")
    
    st.markdown("---")
    
    # Create input form
    st.subheader("📝 Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age", 18, 80, 45)
        yearly_income = st.number_input("Yearly Income ($)", 20000, 200000, 60000, step=5000)
        total_children = st.selectbox("Total Children", [0, 1, 2, 3, 4, 5])
        children_at_home = st.selectbox("Children at Home", list(range(total_children + 1)))
        
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", 
                                ["High School", "Some College", "Bachelor", "Graduate"])
        occupation = st.selectbox("Occupation", 
                                 ["Professional", "Management", "Skilled Manual", "Clerical"])
        house_owner = st.selectbox("House Owner", ["Yes", "No"])
    
    with col2:
        st.markdown("**Purchase Behavior**")
        total_orders = st.slider("Total Orders", 1, 50, 10)
        total_quantity = st.slider("Total Items Purchased", 1, 200, 25)
        unique_products = st.slider("Unique Products Bought", 1, 30, 8)
        
        customer_lifespan_days = st.slider("Customer Relationship (Days)", 30, 2000, 365)
        recency_days = st.slider("Days Since Last Purchase", 0, 365, 30)
        
        # Calculate derived metrics
        avg_order_size = total_quantity / total_orders
        purchase_frequency = total_orders / customer_lifespan_days
        
        st.metric("Average Order Size", f"{avg_order_size:.2f} items")
        st.metric("Purchase Frequency", f"{purchase_frequency:.4f} orders/day")
    
    # Predict button
    if st.button("🔮 Predict Customer Lifetime Value", type="primary"):
        # Prepare input data
        input_data = {
            'age': age,
            'MaritalStatus': 1 if marital_status == "Married" else 0,
            'Gender': 1 if gender == "Male" else 0,
            'YearlyIncome': yearly_income,
            'TotalChildren': total_children,
            'NumberChildrenAtHome': children_at_home,
            'Education': {"High School": 0, "Some College": 1, "Bachelor": 2, "Graduate": 3}[education],
            'Occupation': {"Professional": 0, "Management": 1, "Skilled Manual": 2, "Clerical": 3}[occupation],
            'HouseOwnerFlag': 1 if house_owner == "Yes" else 0,
            'total_orders': total_orders,
            'total_quantity': total_quantity,
            'avg_order_size': avg_order_size,
            'unique_products': unique_products,
            'customer_lifespan_days': customer_lifespan_days,
            'recency_days': recency_days,
            'purchase_frequency': purchase_frequency
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)[0][0]
        
        # Ensure positive prediction
        prediction = max(prediction, 0.01)
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Display prediction with gauge
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            gauge_fig = create_gauge_chart(
                value=prediction,
                title="Predicted CLV ($)",
                max_value=min(max(prediction * 1.5, 10), 50)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Prediction details
        st.info(f"**Predicted Customer Lifetime Value: ${prediction:.2f}**")
        
        # Customer segment
        if prediction < 1:
            segment = "Low Value"
            color = "🔴"
            recommendation = "Focus on engagement and retention strategies"
        elif prediction < 3:
            segment = "Medium Value"  
            color = "🟡"
            recommendation = "Implement targeted upselling campaigns"
        else:
            segment = "High Value"
            color = "🟢"
            recommendation = "Provide premium service and exclusive offers"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Customer Segment", f"{color} {segment}")
        with col2:
            st.metric("Monthly CLV", f"${prediction/12:.2f}")
        with col3:
            st.metric("Annual CLV", f"${prediction:.2f}")
        
        st.success(f"💡 **Recommendation:** {recommendation}")
        
        # Feature contribution (simplified interpretation)
        st.markdown("### 📊 Key Contributing Factors")
        
        factors = []
        if total_orders > 15:
            factors.append("High purchase frequency")
        if yearly_income > 80000:
            factors.append("High income bracket")
        if customer_lifespan_days > 500:
            factors.append("Long-term customer relationship")
        if recency_days < 60:
            factors.append("Recent purchase activity")
        if unique_products > 10:
            factors.append("Product diversity")
        
        if factors:
            for factor in factors:
                st.write(f"• {factor}")
        else:
            st.write("• Standard customer profile")

if __name__ == "__main__":
    main()
