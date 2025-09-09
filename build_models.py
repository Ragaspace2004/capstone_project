"""
AdventureWorks ML Models Builder
==================================
This script builds comprehensive machine learning models for AdventureWorks dataset:
- Customer Lifetime Value (CLV)
- Customer Segmentation
- Customer Churn Prediction  
- Sales Forecasting

All trained models are saved as PKL files for use in the Streamlit application.

Usage: python build_models.py
"""

import warnings
import sys
import os
import subprocess
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Database and ML imports
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def check_and_install_packages():
    """Check and install required packages with UV package manager and compatibility fixes"""
    print("🔧 Checking and installing required packages with UV...")
    
    # Check if UV is available
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        print("✅ UV package manager detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ UV not found. Please install UV first:")
        print("   - Windows: powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'")
        print("   - Or visit: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    # Core packages with compatibility fixes for NumPy 2.x issues
    print("📦 Installing core packages with UV...")
    
    core_packages = [
        "numpy==1.26.4",  # Compatible version to avoid SciPy issues
        "scipy==1.11.4",
        "scikit-learn==1.4.0",
        "pandas>=2.0.0,<2.5.0"
    ]
    
    # Install core packages first
    for package in core_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run(["uv", "pip", "install", package], 
                         capture_output=True, check=True, text=True)
            print(f"✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}")
            print(f"   Error: {e}")
            return False
    
    # Additional packages
    additional_packages = [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "sqlalchemy>=2.0.0",
        "pymysql>=1.0.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "tqdm>=4.65.0"
    ]
    
    for package in additional_packages:
        try:
            subprocess.run(["uv", "pip", "install", package], 
                         capture_output=True, check=True)
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not install {package}")
    
    print("📦 UV package installation completed!")
    return True

class AdventureWorksMLPipeline:
    """Complete ML pipeline for AdventureWorks dataset"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.feature_names = {}
        
        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'root',
            'database': 'adventureworks'
        }
        
        # Create models directory
        if not os.path.exists('models'):
            os.makedirs('models')
            print("📁 Created models directory")
    
    def connect_database(self):
        """Establish database connection"""
        try:
            connection_string = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}?charset=utf8mb4"
            self.engine = create_engine(connection_string)
            print("✅ Database connection established")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def load_data(self):
        """Load all datasets from database"""
        print("\n📊 Loading datasets from database...")
        
        try:
            # Load all tables
            self.customers_df = pd.read_sql("SELECT * FROM customer_trans", self.engine)
            self.sales_df = pd.read_sql("SELECT * FROM sales_trans", self.engine)
            self.products_df = pd.read_sql("SELECT * FROM products_trans", self.engine)
            self.territories_df = pd.read_sql("SELECT * FROM territories_trans", self.engine)
            
            print(f"✅ Customers: {len(self.customers_df)} records")
            print(f"✅ Sales: {len(self.sales_df)} records")
            print(f"✅ Products: {len(self.products_df)} records")
            print(f"✅ Territories: {len(self.territories_df)} records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("\n🔧 Preprocessing data...")
        
        # Convert date columns
        self.sales_df['order_date'] = pd.to_datetime(self.sales_df['order_date'])
        self.customers_df['birth_date'] = pd.to_datetime(self.customers_df['birth_date'])
        
        # Create customer age
        current_date = pd.Timestamp.now()
        self.customers_df['age'] = (current_date - self.customers_df['birth_date']).dt.days // 365
        
        # Handle missing values
        self.customers_df['annual_income'].fillna(self.customers_df['annual_income'].median(), inplace=True)
        self.customers_df['total_children'].fillna(0, inplace=True)
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_marital = LabelEncoder()
        le_education = LabelEncoder()
        
        self.customers_df['gender_encoded'] = le_gender.fit_transform(self.customers_df['gender'].fillna('Unknown'))
        self.customers_df['marital_status_encoded'] = le_marital.fit_transform(self.customers_df['marital_status'].fillna('Unknown'))
        self.customers_df['education_encoded'] = le_education.fit_transform(self.customers_df['education_level'].fillna('Unknown'))
        
        # Create aggregated customer features
        customer_sales = self.sales_df.groupby('customer_id').agg({
            'order_quantity': ['count', 'sum', 'mean'],
            'order_date': ['min', 'max']
        }).reset_index()
        
        customer_sales.columns = ['customer_id', 'total_orders', 'total_quantity', 'avg_order_size', 'first_order', 'last_order']
        
        # Calculate customer metrics
        customer_sales['customer_lifespan_days'] = (customer_sales['last_order'] - customer_sales['first_order']).dt.days
        customer_sales['recency_days'] = (current_date - customer_sales['last_order']).dt.days
        
        # Merge customer and sales data
        self.customer_features = self.customers_df.merge(customer_sales, on='customer_id', how='left')
        self.customer_features.fillna(0, inplace=True)
        
        print("✅ Data preprocessing completed")
        print(f"Final dataset shape: {self.customer_features.shape}")
    
    def prepare_datasets(self):
        """Prepare datasets for different ML tasks"""
        print("\n📋 Preparing datasets for ML tasks...")
        
        # 1. CLV Dataset
        self.clv_features = ['annual_income', 'total_children', 'age', 'gender_encoded', 
                           'marital_status_encoded', 'education_encoded', 'total_orders', 
                           'total_quantity', 'avg_order_size', 'customer_lifespan_days']
        
        # Create CLV target
        self.customer_features['clv_target'] = (
            self.customer_features['total_quantity'] * 
            self.customer_features['avg_order_size'] * 
            (1 + self.customer_features['total_orders'] / 100)
        )
        
        # Remove outliers for CLV
        clv_data = self.customer_features[self.customer_features['total_orders'] > 0].copy()
        Q1 = clv_data['clv_target'].quantile(0.25)
        Q3 = clv_data['clv_target'].quantile(0.75)
        IQR = Q3 - Q1
        clv_data = clv_data[(clv_data['clv_target'] >= Q1 - 1.5*IQR) & 
                           (clv_data['clv_target'] <= Q3 + 1.5*IQR)]
        
        X_clv = clv_data[self.clv_features]
        y_clv = clv_data['clv_target']
        self.X_clv_train, self.X_clv_test, self.y_clv_train, self.y_clv_test = train_test_split(
            X_clv, y_clv, test_size=0.2, random_state=42)
        
        # 2. Segmentation Dataset
        self.segmentation_features = ['annual_income', 'total_children', 'age', 'total_orders', 
                                    'total_quantity', 'avg_order_size', 'recency_days']
        self.X_segment = clv_data[self.segmentation_features]
        
        # 3. Churn Dataset
        self.customer_features['is_churned'] = (self.customer_features['recency_days'] > 180).astype(int)
        churn_data = self.customer_features[self.customer_features['total_orders'] > 0].copy()
        
        self.churn_features = ['annual_income', 'total_children', 'age', 'gender_encoded', 
                             'marital_status_encoded', 'education_encoded', 'total_orders', 
                             'avg_order_size', 'customer_lifespan_days', 'recency_days']
        
        X_churn = churn_data[self.churn_features]
        y_churn = churn_data['is_churned']
        self.X_churn_train, self.X_churn_test, self.y_churn_train, self.y_churn_test = train_test_split(
            X_churn, y_churn, test_size=0.2, random_state=42)
        
        # 4. Sales Forecasting Dataset
        daily_sales = self.sales_df.groupby('order_date').agg({
            'order_quantity': 'sum',
            'customer_id': 'nunique',
            'product_id': 'nunique'
        }).reset_index()
        
        daily_sales.columns = ['date', 'total_quantity', 'unique_customers', 'unique_products']
        
        # Create time features
        daily_sales['year'] = daily_sales['date'].dt.year
        daily_sales['month'] = daily_sales['date'].dt.month
        daily_sales['day'] = daily_sales['date'].dt.day
        daily_sales['dayofweek'] = daily_sales['date'].dt.dayofweek
        daily_sales['quarter'] = daily_sales['date'].dt.quarter
        
        # Create lag features
        daily_sales['lag_1'] = daily_sales['total_quantity'].shift(1)
        daily_sales['lag_7'] = daily_sales['total_quantity'].shift(7)
        daily_sales['rolling_mean_7'] = daily_sales['total_quantity'].rolling(7).mean()
        
        daily_sales_clean = daily_sales.dropna()
        
        self.forecast_features = ['month', 'day', 'dayofweek', 'quarter', 'unique_customers', 
                                'unique_products', 'lag_1', 'lag_7', 'rolling_mean_7']
        X_forecast = daily_sales_clean[self.forecast_features]
        y_forecast = daily_sales_clean['total_quantity']
        
        self.X_forecast_train, self.X_forecast_test, self.y_forecast_train, self.y_forecast_test = train_test_split(
            X_forecast, y_forecast, test_size=0.2, random_state=42, shuffle=False)
        
        print("✅ All datasets prepared successfully")
    
    def train_clv_model(self):
        """Train Customer Lifetime Value model"""
        print("\n💰 Training CLV Model...")
        
        # Scale features
        self.scalers['clv'] = StandardScaler()
        X_clv_train_scaled = self.scalers['clv'].fit_transform(self.X_clv_train)
        X_clv_test_scaled = self.scalers['clv'].transform(self.X_clv_test)
        
        # Train model
        self.models['clv'] = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.models['clv'].fit(X_clv_train_scaled, self.y_clv_train)
        
        # Evaluate
        y_pred = self.models['clv'].predict(X_clv_test_scaled)
        mae = mean_absolute_error(self.y_clv_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_clv_test, y_pred))
        r2 = r2_score(self.y_clv_test, y_pred)
        
        self.model_performance['clv'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"CLV Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    def train_segmentation_model(self):
        """Train Customer Segmentation model"""
        print("\n👥 Training Segmentation Model...")
        
        # Scale features
        self.scalers['segmentation'] = StandardScaler()
        X_segment_scaled = self.scalers['segmentation'].fit_transform(self.X_segment)
        
        # Train KMeans
        self.models['segmentation'] = KMeans(n_clusters=5, random_state=42)
        segment_labels = self.models['segmentation'].fit_predict(X_segment_scaled)
        
        # Create segment mapping
        self.segment_mapping = {0: 'Low Value', 1: 'Occasional', 2: 'Regular', 3: 'High Value', 4: 'Premium'}
        
        print(f"Segmentation Model - Created 5 customer segments")
    
    def train_churn_model(self):
        """Train Customer Churn Prediction model"""
        print("\n⚠️ Training Churn Model...")
        
        # Scale features
        self.scalers['churn'] = StandardScaler()
        X_churn_train_scaled = self.scalers['churn'].fit_transform(self.X_churn_train)
        X_churn_test_scaled = self.scalers['churn'].transform(self.X_churn_test)
        
        # Train model
        self.models['churn'] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.models['churn'].fit(X_churn_train_scaled, self.y_churn_train)
        
        # Evaluate
        y_pred = self.models['churn'].predict(X_churn_test_scaled)
        accuracy = accuracy_score(self.y_churn_test, y_pred)
        
        self.model_performance['churn'] = {'accuracy': accuracy}
        
        print(f"Churn Model - Accuracy: {accuracy:.3f}")
    
    def train_forecast_model(self):
        """Train Sales Forecasting model"""
        print("\n📈 Training Sales Forecast Model...")
        
        # Train model
        self.models['forecast'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['forecast'].fit(self.X_forecast_train, self.y_forecast_train)
        
        # Evaluate
        y_pred = self.models['forecast'].predict(self.X_forecast_test)
        mae = mean_absolute_error(self.y_forecast_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_forecast_test, y_pred))
        r2 = r2_score(self.y_forecast_test, y_pred)
        
        self.model_performance['forecast'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"Forecast Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    def save_models(self):
        """Save all trained models and metadata"""
        print("\n💾 Saving models to PKL files...")
        
        # Save individual models and scalers
        model_files = [
            ('models/clv_model.pkl', self.models['clv']),
            ('models/clv_scaler.pkl', self.scalers['clv']),
            ('models/segmentation_model.pkl', self.models['segmentation']),
            ('models/segmentation_scaler.pkl', self.scalers['segmentation']),
            ('models/churn_model.pkl', self.models['churn']),
            ('models/churn_scaler.pkl', self.scalers['churn']),
            ('models/sales_forecast_model.pkl', self.models['forecast'])
        ]
        
        for filename, model_obj in model_files:
            with open(filename, 'wb') as f:
                pickle.dump(model_obj, f)
            print(f"✅ Saved {filename}")
        
        # Save metadata
        model_metadata = {
            'clv_features': self.clv_features,
            'churn_features': self.churn_features,
            'forecast_features': self.forecast_features,
            'segmentation_features': self.segmentation_features,
            'segment_mapping': self.segment_mapping,
            'model_performance': self.model_performance,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(model_metadata, f)
        
        print("✅ Saved model_metadata.pkl")
        print(f"\n🎉 All models saved successfully in 'models/' directory!")
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline"""
        print("🚀 Starting AdventureWorks ML Pipeline...")
        print("=" * 60)
        
        # Connect to database
        if not self.connect_database():
            return False
        
        # Load data
        if not self.load_data():
            return False
        
        # Preprocess data
        self.preprocess_data()
        
        # Prepare datasets
        self.prepare_datasets()
        
        # Train all models
        self.train_clv_model()
        self.train_segmentation_model()
        self.train_churn_model()
        self.train_forecast_model()
        
        # Save models
        self.save_models()
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print summary
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print(f"CLV Model R²: {self.model_performance['clv']['r2']:.3f}")
        print(f"Churn Model Accuracy: {self.model_performance['churn']['accuracy']:.3f}")
        print(f"Forecast Model R²: {self.model_performance['forecast']['r2']:.3f}")
        print(f"Segmentation: 5 customer segments created")
        
        print("\n📁 FILES CREATED:")
        for file in os.listdir('models'):
            print(f"- models/{file}")
        
        print("\n🚀 Ready to run Streamlit app: python -m streamlit run app.py")
        
        return True

def main():
    """Main execution function"""
    print("🏗️  AdventureWorks ML Models Builder")
    print("=" * 50)
    
    # Optional: Install packages (uncomment if needed)
    # check_and_install_packages()
    
    # Initialize and run pipeline
    pipeline = AdventureWorksMLPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ Build completed successfully!")
        print("Run 'python app.py' or 'streamlit run app.py' to start the web application.")
    else:
        print("\n❌ Build failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
