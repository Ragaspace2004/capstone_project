import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns

# Database connection
HOST = "localhost"
PORT = 3306
USER = "root"
PASSWORD = "root"
DATABASE = "adventureworks"

engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?charset=utf8mb4")

def check_data_completeness():
    """Check row counts and data completeness for all cleaned tables"""
    print("=== DATA COMPLETENESS CHECK ===")
    
    # Original raw tables
    raw_tables = {
        'customers': 'CustomerKey',
        'territories': 'SalesTerritoryKey', 
        'product_categories': 'ProductCategoryKey',
        'product_subcategories': 'ProductSubcategoryKey',
        'products': 'ProductKey',
        'sales': 'OrderDate'
    }
    
    # Cleaned tables
    clean_tables = {
        'customer_trans': 'customer_id',
        'territories_trans': 'territory_id',
        'product_categories_trans': 'category_id', 
        'product_subcategories_trans': 'subcategory_id',
        'products_trans': 'product_id',
        'sales_trans': 'order_date'
    }
    
    print("\nRAW vs CLEANED TABLE COMPARISON:")
    print("-" * 60)
    print(f"{'Table':<25} {'Raw Rows':<12} {'Clean Rows':<12} {'% Retained':<12}")
    print("-" * 60)
    
    for (raw_table, raw_key), (clean_table, clean_key) in zip(raw_tables.items(), clean_tables.items()):
        try:
            # Count raw rows
            raw_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {raw_table}", engine)['count'][0]
            
            # Count cleaned rows  
            clean_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {clean_table}", engine)['count'][0]
            
            # Calculate retention percentage
            retention = (clean_count / raw_count * 100) if raw_count > 0 else 0
            
            print(f"{raw_table:<25} {raw_count:<12} {clean_count:<12} {retention:.1f}%")
            
        except Exception as e:
            print(f"{raw_table:<25} ERROR: {e}")
    
    print("\nCLEANED TABLES OVERVIEW:")
    print("-" * 40)
    for table, key_col in clean_tables.items():
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", engine)['count'][0]
            print(f"{table}: {count:,} rows")
        except Exception as e:
            print(f"{table}: ERROR - {e}")

def generate_insights():
    """Generate 10 unique business insights from the cleaned data"""
    print("\n" + "="*60)
    print("10 UNIQUE BUSINESS INSIGHTS FROM ADVENTUREWORKS DATA")
    print("="*60)
    
    insights = []
    
    # 1. Customer Demographics Analysis
    try:
        customers = pd.read_sql("""
            SELECT gender, marital_status, COUNT(*) as count, 
                   AVG(annual_income) as avg_income,
                   AVG(total_children) as avg_children
            FROM customer_trans 
            WHERE gender IS NOT NULL AND marital_status IS NOT NULL
            GROUP BY gender, marital_status
        """, engine)
        
        print("\n1. CUSTOMER DEMOGRAPHICS INSIGHTS:")
        print(customers.to_string(index=False))
        
        # Find highest earning segment
        top_earner = customers.loc[customers['avg_income'].idxmax()]
        insights.append(f"Highest earning customer segment: {top_earner['gender']} + {top_earner['marital_status']} (${top_earner['avg_income']:,.0f} avg income)")
        
    except Exception as e:
        print(f"Error in customer demographics: {e}")
    
    # 2. Sales by Year Analysis
    try:
        yearly_sales = pd.read_sql("""
            SELECT YEAR(order_date) as year, 
                   COUNT(*) as total_orders,
                   SUM(order_quantity) as total_quantity
            FROM sales_trans
            WHERE order_date IS NOT NULL
            GROUP BY YEAR(order_date)
            ORDER BY year
        """, engine)
        
        print("\n2. YEARLY SALES TRENDS:")
        print(yearly_sales.to_string(index=False))
        
        if len(yearly_sales) > 1:
            growth = ((yearly_sales['total_orders'].iloc[-1] - yearly_sales['total_orders'].iloc[0]) / 
                     yearly_sales['total_orders'].iloc[0] * 100)
            insights.append(f"Sales growth from {yearly_sales['year'].iloc[0]} to {yearly_sales['year'].iloc[-1]}: {growth:.1f}%")
        
    except Exception as e:
        print(f"Error in yearly sales: {e}")
    
    # 3. Top Selling Products
    try:
        top_products = pd.read_sql("""
            SELECT p.product_name, p.product_color,
                   COUNT(*) as order_frequency,
                   SUM(s.order_quantity) as total_sold
            FROM sales_trans s
            JOIN products_trans p ON s.product_id = p.product_id
            GROUP BY p.product_id
            ORDER BY total_sold DESC
            LIMIT 5
        """, engine)
        
        print("\n3. TOP 5 BEST SELLING PRODUCTS:")
        print(top_products.to_string(index=False))
        
        insights.append(f"Best selling product: {top_products['product_name'].iloc[0]} ({top_products['total_sold'].iloc[0]} units sold)")
        
    except Exception as e:
        print(f"Error in top products: {e}")
    
    # 4. Geographic Sales Distribution
    try:
        geo_sales = pd.read_sql("""
            SELECT t.continent, t.country, 
                   COUNT(*) as order_count,
                   SUM(s.order_quantity) as total_quantity
            FROM sales_trans s
            JOIN territories_trans t ON s.territory_id = t.territory_id
            GROUP BY t.continent, t.country
            ORDER BY total_quantity DESC
        """, engine)
        
        print("\n4. GEOGRAPHIC SALES DISTRIBUTION:")
        print(geo_sales.head(10).to_string(index=False))
        
        top_country = geo_sales.iloc[0]
        insights.append(f"Top performing country: {top_country['country']} ({top_country['total_quantity']} units)")
        
    except Exception as e:
        print(f"Error in geographic sales: {e}")
    
    # 5. Customer Income vs Purchase Behavior
    try:
        income_analysis = pd.read_sql("""
            SELECT 
                CASE 
                    WHEN c.annual_income < 30000 THEN 'Low (<30K)'
                    WHEN c.annual_income < 60000 THEN 'Medium (30-60K)'
                    WHEN c.annual_income < 100000 THEN 'High (60-100K)'
                    ELSE 'Very High (>100K)'
                END as income_bracket,
                COUNT(DISTINCT s.customer_id) as customer_count,
                AVG(s.order_quantity) as avg_order_size
            FROM customer_trans c
            JOIN sales_trans s ON c.customer_id = s.customer_id
            WHERE c.annual_income IS NOT NULL
            GROUP BY income_bracket
            ORDER BY customer_count DESC
        """, engine)
        
        print("\n5. INCOME vs PURCHASE BEHAVIOR:")
        print(income_analysis.to_string(index=False))
        
        insights.append(f"Most active income bracket: {income_analysis.iloc[0]['income_bracket']} ({income_analysis.iloc[0]['customer_count']} customers)")
        
    except Exception as e:
        print(f"Error in income analysis: {e}")
    
    # 6. Product Category Performance
    try:
        category_performance = pd.read_sql("""
            SELECT pc.category_name,
                   COUNT(DISTINCT p.product_id) as product_count,
                   COUNT(s.product_id) as total_orders,
                   SUM(s.order_quantity) as total_sold
            FROM product_categories_trans pc
            JOIN product_subcategories_trans ps ON pc.category_id = ps.category_id
            JOIN products_trans p ON ps.subcategory_id = p.subcategory_id
            LEFT JOIN sales_trans s ON p.product_id = s.product_id
            GROUP BY pc.category_id, pc.category_name
            ORDER BY total_sold DESC
        """, engine)
        
        print("\n6. PRODUCT CATEGORY PERFORMANCE:")
        print(category_performance.to_string(index=False))
        
        insights.append(f"Top product category: {category_performance.iloc[0]['category_name']} ({category_performance.iloc[0]['total_sold']} units)")
        
    except Exception as e:
        print(f"Error in category performance: {e}")
    
    # 7. Seasonal Sales Patterns
    try:
        seasonal = pd.read_sql("""
            SELECT 
                QUARTER(order_date) as quarter,
                MONTH(order_date) as month,
                MONTHNAME(order_date) as month_name,
                COUNT(*) as order_count,
                SUM(order_quantity) as quantity_sold
            FROM sales_trans
            WHERE order_date IS NOT NULL
            GROUP BY QUARTER(order_date), MONTH(order_date), MONTHNAME(order_date)
            ORDER BY quantity_sold DESC
        """, engine)
        
        print("\n7. SEASONAL SALES PATTERNS:")
        print(seasonal.to_string(index=False))
        
        best_month = seasonal.iloc[0]
        insights.append(f"Peak sales month: {best_month['month_name']} Q{best_month['quarter']} ({best_month['quantity_sold']} units)")
        
    except Exception as e:
        print(f"Error in seasonal analysis: {e}")
    
    # 8. Customer Family Size Impact
    try:
        family_impact = pd.read_sql("""
            SELECT 
                CASE 
                    WHEN c.total_children = 0 THEN 'No Children'
                    WHEN c.total_children BETWEEN 1 AND 2 THEN '1-2 Children'
                    ELSE '3+ Children'
                END as family_size,
                COUNT(DISTINCT c.customer_id) as customer_count,
                COUNT(s.customer_id) as total_orders,
                AVG(s.order_quantity) as avg_order_size
            FROM customer_trans c
            JOIN sales_trans s ON c.customer_id = s.customer_id
            WHERE c.total_children IS NOT NULL
            GROUP BY family_size
            ORDER BY total_orders DESC
        """, engine)
        
        print("\n8. FAMILY SIZE vs PURCHASING:")
        print(family_impact.to_string(index=False))
        
        top_family = family_impact.iloc[0]
        insights.append(f"Most active family segment: {top_family['family_size']} ({top_family['total_orders']} orders)")
        
    except Exception as e:
        print(f"Error in family analysis: {e}")
    
    # 9. Education Level Analysis
    try:
        education_analysis = pd.read_sql("""
            SELECT c.education_level,
                   COUNT(DISTINCT c.customer_id) as customer_count,
                   AVG(c.annual_income) as avg_income,
                   COUNT(s.customer_id) as total_orders
            FROM customer_trans c
            LEFT JOIN sales_trans s ON c.customer_id = s.customer_id
            WHERE c.education_level IS NOT NULL
            GROUP BY c.education_level
            ORDER BY avg_income DESC
        """, engine)
        
        print("\n9. EDUCATION LEVEL INSIGHTS:")
        print(education_analysis.to_string(index=False))
        
        top_edu = education_analysis.iloc[0]
        insights.append(f"Highest earning education level: {top_edu['education_level']} (${top_edu['avg_income']:,.0f} avg income)")
        
    except Exception as e:
        print(f"Error in education analysis: {e}")
    
    # 10. Product Price Analysis
    try:
        price_analysis = pd.read_sql("""
            SELECT 
                CASE 
                    WHEN p.product_price < 100 THEN 'Budget (<$100)'
                    WHEN p.product_price < 500 THEN 'Mid-range ($100-500)'
                    WHEN p.product_price < 1000 THEN 'Premium ($500-1000)'
                    ELSE 'Luxury (>$1000)'
                END as price_range,
                COUNT(DISTINCT p.product_id) as product_count,
                AVG(p.product_price) as avg_price,
                SUM(s.order_quantity) as total_sold
            FROM products_trans p
            LEFT JOIN sales_trans s ON p.product_id = s.product_id
            WHERE p.product_price IS NOT NULL
            GROUP BY price_range
            ORDER BY total_sold DESC
        """, engine)
        
        print("\n10. PRODUCT PRICE RANGE ANALYSIS:")
        print(price_analysis.to_string(index=False))
        
        best_price = price_analysis.iloc[0]
        insights.append(f"Best selling price range: {best_price['price_range']} ({best_price['total_sold']} units sold)")
        
    except Exception as e:
        print(f"Error in price analysis: {e}")
    
    # Summary of Key Insights
    print("\n" + "="*60)
    print("KEY BUSINESS INSIGHTS SUMMARY:")
    print("="*60)
    for i, insight in enumerate(insights, 1):
        print(f"{i:2d}. {insight}")
    
    print(f"\nAnalysis complete! Found {len(insights)} actionable insights.")
    print("\nData is ready for Machine Learning & Deep Learning analysis!")
    
    return insights

if __name__ == "__main__":
    check_data_completeness()
    generate_insights()
