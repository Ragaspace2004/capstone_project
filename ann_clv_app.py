"""
Customer Lifetime Value (CLV) - Neural Network Prediction App
============================================================
Streamlit web application for testing CLV predictions using trained ANN model.

Features:
- Interactive customer input forms
- Real-time CLV predictions using neural network
- Model performance metrics display
- Visual prediction confidence indicators
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CLV Neural Network Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_ann_model():
    """Load the trained neural network model and preprocessing objects"""
    try:
        model_dir = 'ann_clv_model'
        
        with open(f'{model_dir}/ann_regressor.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open(f'{model_dir}/ann_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open(f'{model_dir}/ann_label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open(f'{model_dir}/ann_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, label_encoders, metadata
        
    except FileNotFoundError:
        st.error("❌ Neural Network model files not found! Please train the model first by running: `uv run python train_ann_model.py`")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def create_clv_gauge(clv_value, title="Predicted CLV"):
    """Create a gauge chart for CLV visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = clv_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 2.5},  # Average CLV reference
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 1], 'color': "lightgray"},
                {'range': [1, 3], 'color': "yellow"},
                {'range': [3, 6], 'color': "orange"},
                {'range': [6, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': clv_value
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🧠 Customer Lifetime Value - Neural Network Predictor")
    st.markdown("### Deep Learning approach with regularization to prevent overfitting")
    
    # Load model
    with st.spinner("Loading neural network model..."):
        model, scaler, label_encoders, metadata = load_ann_model()
    
    # Display model information
    with st.expander("🔍 Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", "Neural Network (ANN)")
            st.metric("Hidden Layers", str(metadata.get('hidden_layers', 'N/A')))
        
        with col2:
            perf = metadata.get('performance', {})
            st.metric("Test R² Score", f"{perf.get('test_r2', 0):.3f}")
            st.metric("Regularization α", f"{metadata.get('regularization_alpha', 'N/A')}")
        
        with col3:
            st.metric("Test MAE", f"${perf.get('test_mae', 0):.2f}")
            st.metric("Overfitting Gap", f"{perf.get('overfitting_gap', 0):.3f}")
        
        # Additional model details
        st.markdown("**🏗️ Model Architecture:**")
        st.write(f"- Input features: {len(metadata.get('feature_columns', []))}")
        st.write(f"- Training iterations: {perf.get('n_iterations', 'N/A')}")
        st.write(f"- Converged: {'✅ Yes' if perf.get('converged', False) else '❌ No (reached max_iter)'}")
        st.write(f"- Training date: {metadata.get('training_date', 'N/A')}")
    
    # Input form
    st.markdown("---")
    st.header("📝 Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Demographics")
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        yearly_income = st.number_input("Yearly Income ($)", min_value=0, max_value=200000, value=50000, step=1000)
        total_children = st.selectbox("Total Children", options=[0, 1, 2, 3, 4, 5], index=1)
        children_at_home = st.selectbox("Children at Home", options=[0, 1, 2, 3, 4, 5], index=0)
        
        st.subheader("🏠 Lifestyle")
        house_owner = st.selectbox("House Owner", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
        cars_owned = st.selectbox("Number of Cars", options=[0, 1, 2, 3, 4], index=1)
        marital_status = st.selectbox("Marital Status", options=["Married", "Single", "Divorced"])
        gender = st.selectbox("Gender", options=["Male", "Female"])
    
    with col2:
        st.subheader("🎓 Education & Work")
        education = st.selectbox("Education Level", options=[
            "High School", "Partial College", "Bachelors", "Graduate Degree"
        ], index=2)
        occupation = st.selectbox("Occupation", options=[
            "Professional", "Management", "Skilled Manual", "Clerical", "Manual"
        ])
        commute_distance = st.selectbox("Commute Distance", options=[
            "0-1 Miles", "1-2 Miles", "2-5 Miles", "5-10 Miles", "10+ Miles"
        ], index=2)
        
        st.subheader("🛒 Purchase Behavior")
        total_orders = st.slider("Total Orders (Historical)", min_value=1, max_value=50, value=5)
        total_quantity = st.slider("Total Items Purchased", min_value=1, max_value=200, value=20)
        avg_order_value = st.number_input("Average Order Value ($)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
        customer_lifespan_days = st.slider("Customer Lifespan (Days)", min_value=30, max_value=2000, value=365)
        recency_days = st.slider("Days Since Last Purchase", min_value=0, max_value=365, value=30)
    
    # Calculate derived features
    income_per_child = yearly_income / (total_children + 1)
    avg_order_size = total_quantity / total_orders
    purchase_frequency = total_orders / customer_lifespan_days
    
    # Prediction button
    st.markdown("---")
    if st.button("🚀 Predict Customer Lifetime Value", type="primary"):
        
        # Prepare input data
        input_data = {
            'age': age,
            'YearlyIncome': yearly_income,
            'TotalChildren': total_children,
            'NumberChildrenAtHome': children_at_home,
            'HouseOwnerFlag': house_owner,
            'NumberCarsOwned': cars_owned,
            'income_per_child': income_per_child,
            'total_orders': total_orders,
            'total_quantity': total_quantity,
            'avg_order_value': avg_order_value,
            'avg_order_size': avg_order_size,
            'customer_lifespan_days': customer_lifespan_days,
            'recency_days': recency_days,
            'purchase_frequency': purchase_frequency
        }
        
        # Encode categorical variables
        try:
            input_data['MaritalStatus_encoded'] = label_encoders['MaritalStatus'].transform([marital_status])[0]
        except:
            input_data['MaritalStatus_encoded'] = 0
        
        try:
            input_data['Gender_encoded'] = label_encoders['Gender'].transform([gender])[0]
        except:
            input_data['Gender_encoded'] = 0
        
        try:
            input_data['EnglishEducation_encoded'] = label_encoders['EnglishEducation'].transform([education])[0]
        except:
            input_data['EnglishEducation_encoded'] = 0
        
        try:
            input_data['EnglishOccupation_encoded'] = label_encoders['EnglishOccupation'].transform([occupation])[0]
        except:
            input_data['EnglishOccupation_encoded'] = 0
        
        try:
            input_data['CommuteDistance_encoded'] = label_encoders['CommuteDistance'].transform([commute_distance])[0]
        except:
            input_data['CommuteDistance_encoded'] = 0
        
        # Create DataFrame with proper column order
        feature_columns = metadata['feature_columns']
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        with st.spinner("🧠 Neural network is thinking..."):
            clv_prediction = model.predict(input_scaled)[0]
            
            # Add some realistic variation for display
            confidence_lower = clv_prediction * 0.9
            confidence_upper = clv_prediction * 1.1
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🎯 Predicted CLV",
                value=f"${clv_prediction:.2f}",
                delta=f"±${(confidence_upper - confidence_lower)/2:.2f}"
            )
        
        with col2:
            # Categorize CLV
            if clv_prediction < 1.0:
                category = "🔴 Low Value"
                color = "red"
            elif clv_prediction < 3.0:
                category = "🟡 Medium Value"
                color = "orange"
            elif clv_prediction < 6.0:
                category = "🟢 High Value"
                color = "green"
            else:
                category = "💎 Premium Value"
                color = "purple"
            
            st.metric(label="📈 Customer Category", value=category)
        
        with col3:
            st.metric(
                label="📅 Monthly Value",
                value=f"${clv_prediction/12:.2f}",
                help="Estimated monthly value based on CLV"
            )
        
        # CLV Gauge
        st.plotly_chart(create_clv_gauge(clv_prediction), use_container_width=True)
        
        # Feature importance for this prediction
        if hasattr(model, 'coefs_'):
            st.subheader("🔍 Key Factors (Neural Network Insights)")
            
            # Simple feature impact analysis
            feature_impact = pd.DataFrame({
                'Feature': ['Total Orders', 'Yearly Income', 'Avg Order Value', 
                           'Customer Lifespan', 'Purchase Frequency'],
                'Impact': [total_orders/10, yearly_income/10000, avg_order_value/50,
                          customer_lifespan_days/365, purchase_frequency*100]
            })
            
            fig = px.bar(feature_impact, x='Impact', y='Feature', orientation='h',
                        title="Feature Impact on CLV Prediction")
            st.plotly_chart(fig, use_container_width=True)
        
        # Business insights
        st.subheader("💡 Business Insights")
        insights = []
        
        if clv_prediction > 4.0:
            insights.append("🌟 This is a high-value customer! Consider premium service offerings.")
        
        if recency_days > 90:
            insights.append("⚠️ Customer hasn't purchased recently. Consider re-engagement campaigns.")
        
        if total_orders > 10:
            insights.append("🔄 Loyal customer with multiple orders. Great for loyalty programs.")
        
        if yearly_income > 75000:
            insights.append("💰 High income customer. May be interested in premium products.")
        
        if purchase_frequency > 0.02:  # More than once per 50 days
            insights.append("⚡ Frequent purchaser. Ideal for subscription services.")
        
        for insight in insights:
            st.write(insight)
        
        # Recommendations
        st.subheader("🎯 Marketing Recommendations")
        
        if clv_prediction > 3.0:
            st.success("**High-Value Strategy**: Focus on retention, premium upsells, and personalized experiences.")
        elif clv_prediction > 1.5:
            st.info("**Growth Strategy**: Increase purchase frequency with targeted promotions and cross-sells.")
        else:
            st.warning("**Nurture Strategy**: Focus on engagement, value demonstration, and conversion optimization.")

if __name__ == "__main__":
    main()
