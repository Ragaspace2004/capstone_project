import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Database connection
HOST = "localhost"
PORT = 3306
USER = "root"
PASSWORD = "root"
DATABASE = "adventureworks"

engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?charset=utf8mb4")

class SalesForecasting:
    def __init__(self):
        self.model = None
        self.forecast_data = None
        
    def load_data(self):
        """Load and prepare sales data for forecasting"""
        query = """
        SELECT 
            DATE(s.order_date) as order_date,
            YEAR(s.order_date) as year,
            MONTH(s.order_date) as month,
            QUARTER(s.order_date) as quarter,
            DAYOFWEEK(s.order_date) as day_of_week,
            COUNT(*) as total_orders,
            SUM(s.order_quantity) as total_quantity,
            AVG(s.order_quantity) as avg_order_size,
            COUNT(DISTINCT s.customer_id) as unique_customers,
            COUNT(DISTINCT s.product_id) as unique_products,
            t.continent,
            pc.category_name
        FROM sales_trans s
        JOIN territories_trans t ON s.territory_id = t.territory_id
        JOIN products_trans p ON s.product_id = p.product_id
        JOIN product_subcategories_trans ps ON p.subcategory_id = ps.subcategory_id
        JOIN product_categories_trans pc ON ps.category_id = pc.category_id
        WHERE s.order_date IS NOT NULL
        GROUP BY DATE(s.order_date), YEAR(s.order_date), MONTH(s.order_date), 
                 QUARTER(s.order_date), DAYOFWEEK(s.order_date), t.continent, pc.category_name
        ORDER BY order_date
        """
        
        self.data = pd.read_sql(query, engine)
        self.data['order_date'] = pd.to_datetime(self.data['order_date'])
        print(f"Loaded {len(self.data)} daily sales records")
        return self.data
    
    def create_aggregated_features(self):
        """Create monthly and quarterly aggregations for forecasting"""
        # Monthly aggregation
        monthly_data = self.data.groupby(['year', 'month']).agg({
            'total_orders': 'sum',
            'total_quantity': 'sum',
            'unique_customers': 'sum',
            'unique_products': 'sum'
        }).reset_index()
        
        monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        monthly_data['month_sin'] = np.sin(2 * np.pi * monthly_data['month'] / 12)
        monthly_data['month_cos'] = np.cos(2 * np.pi * monthly_data['month'] / 12)
        
        # Add lag features
        monthly_data['prev_month_orders'] = monthly_data['total_orders'].shift(1)
        monthly_data['prev_month_quantity'] = monthly_data['total_quantity'].shift(1)
        monthly_data['rolling_avg_3m'] = monthly_data['total_quantity'].rolling(3).mean()
        
        self.monthly_data = monthly_data.dropna()
        print(f"Created monthly aggregation: {len(self.monthly_data)} records")
        return self.monthly_data
    
    def train_sales_forecast_model(self):
        """Train Random Forest model for sales forecasting"""
        features = ['month', 'month_sin', 'month_cos', 'prev_month_orders', 
                   'prev_month_quantity', 'rolling_avg_3m', 'unique_customers', 'unique_products']
        target = 'total_quantity'
        
        X = self.monthly_data[features]
        y = self.monthly_data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(X_train, y_train)
        
        # Model evaluation
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("SALES FORECASTING MODEL PERFORMANCE:")
        print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
        print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance.to_string(index=False))
        
        return self.model
    
    def generate_forecast(self, months_ahead=6):
        """Generate sales forecast for next N months"""
        last_date = self.monthly_data['date'].max()
        forecast_data = []
        
        for i in range(1, months_ahead + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            month = forecast_date.month
            
            # Use latest available data for features
            latest_data = self.monthly_data.iloc[-1]
            
            forecast_features = {
                'month': month,
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'prev_month_orders': latest_data['total_orders'],
                'prev_month_quantity': latest_data['total_quantity'],
                'rolling_avg_3m': latest_data['rolling_avg_3m'],
                'unique_customers': latest_data['unique_customers'],
                'unique_products': latest_data['unique_products']
            }
            
            features_df = pd.DataFrame([forecast_features])
            predicted_quantity = self.model.predict(features_df)[0]
            
            forecast_data.append({
                'date': forecast_date,
                'month': month,
                'year': forecast_date.year,
                'predicted_quantity': predicted_quantity,
                'month_name': forecast_date.strftime('%B')
            })
        
        self.forecast_data = pd.DataFrame(forecast_data)
        
        print(f"\nSALES FORECAST FOR NEXT {months_ahead} MONTHS:")
        print(self.forecast_data[['month_name', 'year', 'predicted_quantity']].to_string(index=False))
        
        return self.forecast_data


class CustomerLifetimeValue:
    def __init__(self):
        self.model = None
        self.customer_features = None
    
    def prepare_clv_data(self):
        """Prepare customer data for CLV prediction"""
        query = """
        SELECT 
            c.customer_id,
            c.annual_income,
            c.total_children,
            c.gender,
            c.marital_status,
            c.education_level,
            COUNT(s.customer_id) as total_orders,
            SUM(s.order_quantity) as total_quantity,
            AVG(s.order_quantity) as avg_order_size,
            MIN(s.order_date) as first_order_date,
            MAX(s.order_date) as last_order_date,
            DATEDIFF(MAX(s.order_date), MIN(s.order_date)) as customer_lifespan_days,
            COUNT(DISTINCT YEAR(s.order_date)) as active_years,
            COUNT(DISTINCT p.product_id) as unique_products_bought
        FROM customer_trans c
        LEFT JOIN sales_trans s ON c.customer_id = s.customer_id
        LEFT JOIN products_trans p ON s.product_id = p.product_id
        WHERE c.annual_income IS NOT NULL
        GROUP BY c.customer_id
        HAVING total_orders > 0
        """
        
        self.customer_data = pd.read_sql(query, engine)
        
        # Feature engineering for CLV
        self.customer_data['recency_days'] = (pd.Timestamp.now() - pd.to_datetime(self.customer_data['last_order_date'])).dt.days
        self.customer_data['frequency'] = self.customer_data['total_orders'] / np.maximum(self.customer_data['customer_lifespan_days'], 1) * 365
        self.customer_data['monetary'] = self.customer_data['total_quantity'] * self.customer_data['avg_order_size']
        
        # Gender and marital status encoding
        self.customer_data['gender_M'] = (self.customer_data['gender'] == 'M').astype(int)
        self.customer_data['marital_M'] = (self.customer_data['marital_status'] == 'M').astype(int)
        
        # Education level encoding (ordinal)
        education_order = {'Partial High School': 1, 'High School': 2, 'Partial College': 3, 'Bachelors': 4, 'Graduate Degree': 5}
        self.customer_data['education_level_encoded'] = self.customer_data['education_level'].map(education_order).fillna(0)
        
        # Target variable: Future CLV (proxy using current value + growth potential)
        self.customer_data['clv_score'] = (
            self.customer_data['monetary'] * 
            (1 + self.customer_data['frequency'] / 365) * 
            np.maximum(self.customer_data['active_years'], 1)
        )
        
        print(f"Prepared CLV data for {len(self.customer_data)} customers")
        return self.customer_data
    
    def train_clv_model(self):
        """Train model to predict Customer Lifetime Value"""
        features = ['annual_income', 'total_children', 'gender_M', 'marital_M', 
                   'education_level_encoded', 'total_orders', 'total_quantity',
                   'avg_order_size', 'customer_lifespan_days', 'frequency', 'monetary']
        target = 'clv_score'
        
        X = self.customer_data[features]
        y = self.customer_data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
        self.model.fit(X_train, y_train)
        
        # Evaluation
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("CUSTOMER LIFETIME VALUE MODEL PERFORMANCE:")
        print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
        print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nCLV Feature Importance:")
        print(feature_importance.to_string(index=False))
        
        return self.model
    
    def predict_customer_segments(self):
        """Segment customers based on CLV predictions"""
        self.customer_data['predicted_clv'] = self.model.predict(self.customer_data[
            ['annual_income', 'total_children', 'gender_M', 'marital_M', 
             'education_level_encoded', 'total_orders', 'total_quantity',
             'avg_order_size', 'customer_lifespan_days', 'frequency', 'monetary']
        ])
        
        # Create CLV segments
        self.customer_data['clv_segment'] = pd.cut(
            self.customer_data['predicted_clv'], 
            bins=5, 
            labels=['Low Value', 'Basic', 'Standard', 'High Value', 'Premium']
        )
        
        segment_summary = self.customer_data.groupby('clv_segment').agg({
            'customer_id': 'count',
            'predicted_clv': ['mean', 'min', 'max'],
            'annual_income': 'mean',
            'total_orders': 'mean'
        }).round(2)
        
        print("\nCUSTOMER LIFETIME VALUE SEGMENTS:")
        print(segment_summary)
        
        return self.customer_data


def main():
    print("="*60)
    print("ADVENTUREWORKS PREDICTIVE ANALYTICS")
    print("="*60)
    
    # Sales Forecasting
    print("\n1. SALES FORECASTING MODEL")
    print("-" * 30)
    sales_forecaster = SalesForecasting()
    sales_forecaster.load_data()
    sales_forecaster.create_aggregated_features()
    sales_forecaster.train_sales_forecast_model()
    sales_forecaster.generate_forecast(months_ahead=6)
    
    print("\n" + "="*60)
    
    # Customer Lifetime Value
    print("\n2. CUSTOMER LIFETIME VALUE ANALYSIS")
    print("-" * 35)
    clv_analyzer = CustomerLifetimeValue()
    clv_analyzer.prepare_clv_data()
    clv_analyzer.train_clv_model()
    clv_analyzer.predict_customer_segments()
    
    print("\n" + "="*60)
    print("PREDICTIVE MODELS TRAINING COMPLETE!")
    print("Next: Customer segmentation and churn prediction models...")

if __name__ == "__main__":
    main()
