import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DECIMAL, TIMESTAMP, text
from sqlalchemy.sql import func
import pymysql

# Database connection
HOST = "localhost"
PORT = 3306
USER = "root"
PASSWORD = "root"
DATABASE = "adventureworks"

engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?charset=utf8mb4")
metadata = MetaData()

def cleanup_trans_tables():
    """Drop all existing *_trans tables to avoid foreign key constraint issues"""
    with engine.begin() as conn:
        # Disable foreign key checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        
        # Drop all _trans tables if they exist
        tables_to_drop = [
            'sales_trans', 'products_trans', 'product_subcategories_trans', 
            'product_categories_trans', 'territories_trans'
        ]
        
        for table in tables_to_drop:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                print(f"Dropped {table}")
            except Exception as e:
                print(f"Could not drop {table}: {e}")
        
        # Re-enable foreign key checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        
    print("Cleanup completed")

def clean_territories():
    # Read raw data
    df = pd.read_sql("SELECT * FROM territories", engine)
    
    # Clean
    df['SalesTerritoryKey'] = pd.to_numeric(df['SalesTerritoryKey'], errors='coerce').astype('Int64')
    df['Region'] = df['Region'].str.strip().replace('', None)
    df['Country'] = df['Country'].str.strip().replace('', None)
    df['Continent'] = df['Continent'].str.strip().replace('', None)
    df = df.dropna(subset=['SalesTerritoryKey'])
    
    # Define table
    territories_trans = Table('territories_trans', metadata,
        Column('territory_id', Integer, primary_key=True),
        Column('region', String(50)),
        Column('country', String(50)),
        Column('continent', String(50)),
        Column('created_at', TIMESTAMP, server_default=func.now()),
        Column('updated_at', TIMESTAMP, server_default=func.now(), onupdate=func.now())
    )
    
    # Drop and create
    territories_trans.create(engine)
    
    # Insert
    df.rename(columns={'SalesTerritoryKey': 'territory_id', 'Region': 'region', 'Country': 'country', 'Continent': 'continent'}, inplace=True)
    df[['territory_id', 'region', 'country', 'continent']].to_sql('territories_trans', engine, if_exists='append', index=False)
    print(f"Cleaned territories: {len(df)} rows inserted")

def clean_product_categories():
    df = pd.read_sql("SELECT * FROM product_categories", engine)
    df['ProductCategoryKey'] = pd.to_numeric(df['ProductCategoryKey'], errors='coerce').astype('Int64')
    df['CategoryName'] = df['CategoryName'].str.strip().replace('', None)
    df = df.dropna(subset=['ProductCategoryKey'])
    
    product_categories_trans = Table('product_categories_trans', metadata,
        Column('category_id', Integer, primary_key=True),
        Column('category_name', String(50)),
        Column('created_at', TIMESTAMP, server_default=func.now()),
        Column('updated_at', TIMESTAMP, server_default=func.now(), onupdate=func.now())
    )
    product_categories_trans.create(engine)
    
    df.rename(columns={'ProductCategoryKey': 'category_id', 'CategoryName': 'category_name'}, inplace=True)
    df[['category_id', 'category_name']].to_sql('product_categories_trans', engine, if_exists='append', index=False)
    print(f"Cleaned product_categories: {len(df)} rows inserted")

def clean_product_subcategories():
    df = pd.read_sql("SELECT * FROM product_subcategories", engine)
    df['ProductSubcategoryKey'] = pd.to_numeric(df['ProductSubcategoryKey'], errors='coerce').astype('Int64')
    df['ProductCategoryKey'] = pd.to_numeric(df['ProductCategoryKey'], errors='coerce').astype('Int64')
    df['SubcategoryName'] = df['SubcategoryName'].str.strip().replace('', None)
    df = df.dropna(subset=['ProductSubcategoryKey'])
    
    product_subcategories_trans = Table('product_subcategories_trans', metadata,
        Column('subcategory_id', Integer, primary_key=True),
        Column('category_id', Integer),
        Column('subcategory_name', String(50)),
        Column('created_at', TIMESTAMP, server_default=func.now()),
        Column('updated_at', TIMESTAMP, server_default=func.now(), onupdate=func.now())
    )
    product_subcategories_trans.create(engine)
    
    df.rename(columns={'ProductSubcategoryKey': 'subcategory_id', 'ProductCategoryKey': 'category_id', 'SubcategoryName': 'subcategory_name'}, inplace=True)
    df[['subcategory_id', 'category_id', 'subcategory_name']].to_sql('product_subcategories_trans', engine, if_exists='append', index=False)
    print(f"Cleaned product_subcategories: {len(df)} rows inserted")

def clean_products():
    df = pd.read_sql("SELECT * FROM products", engine)
    df['ProductKey'] = pd.to_numeric(df['ProductKey'], errors='coerce').astype('Int64')
    df['ProductSubcategoryKey'] = pd.to_numeric(df['ProductSubcategoryKey'], errors='coerce').astype('Int64')
    df['ProductSKU'] = df['ProductSKU'].str.strip().replace('', None)
    df['ProductName'] = df['ProductName'].str.strip().replace('', None)
    df['ModelName'] = df['ModelName'].str.strip().replace('', None)
    df['ProductDescription'] = df['ProductDescription'].str.strip().replace('', None)
    df['ProductColor'] = df['ProductColor'].str.strip().replace('', None)
    df['ProductSize'] = df['ProductSize'].str.strip().replace('', None)
    df['ProductStyle'] = df['ProductStyle'].str.strip().replace('', None)
    df['ProductCost'] = pd.to_numeric(df['ProductCost'].str.replace('[$,]', '', regex=True), errors='coerce').astype('float64')
    df['ProductPrice'] = pd.to_numeric(df['ProductPrice'].str.replace('[$,]', '', regex=True), errors='coerce').astype('float64')
    df = df.dropna(subset=['ProductKey'])
    
    products_trans = Table('products_trans', metadata,
        Column('product_id', Integer, primary_key=True),
        Column('subcategory_id', Integer),
        Column('product_sku', String(50)),
        Column('product_name', String(100)),
        Column('model_name', String(100)),
        Column('product_description', String(500)),
        Column('product_color', String(20)),
        Column('product_size', String(20)),
        Column('product_style', String(10)),
        Column('product_cost', DECIMAL(10,2)),
        Column('product_price', DECIMAL(10,2)),
        Column('created_at', TIMESTAMP, server_default=func.now()),
        Column('updated_at', TIMESTAMP, server_default=func.now(), onupdate=func.now())
    )
    products_trans.create(engine)
    
    df.rename(columns={
        'ProductKey': 'product_id',
        'ProductSubcategoryKey': 'subcategory_id',
        'ProductSKU': 'product_sku',
        'ProductName': 'product_name',
        'ModelName': 'model_name',
        'ProductDescription': 'product_description',
        'ProductColor': 'product_color',
        'ProductSize': 'product_size',
        'ProductStyle': 'product_style',
        'ProductCost': 'product_cost',
        'ProductPrice': 'product_price'
    }, inplace=True)
    df[['product_id', 'subcategory_id', 'product_sku', 'product_name', 'model_name', 'product_description', 'product_color', 'product_size', 'product_style', 'product_cost', 'product_price']].to_sql('products_trans', engine, if_exists='append', index=False)
    print(f"Cleaned products: {len(df)} rows inserted")

def clean_sales():
    df = pd.read_sql("SELECT * FROM sales", engine)
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
    df['StockDate'] = pd.to_datetime(df['StockDate'], errors='coerce')
    df['OrderNumber'] = df['OrderNumber'].str.strip().replace('', None)
    df['ProductKey'] = pd.to_numeric(df['ProductKey'], errors='coerce').astype('Int64')
    df['CustomerKey'] = pd.to_numeric(df['CustomerKey'], errors='coerce').astype('Int64')
    df['TerritoryKey'] = pd.to_numeric(df['TerritoryKey'], errors='coerce').astype('Int64')
    df['OrderLineItem'] = pd.to_numeric(df['OrderLineItem'], errors='coerce').astype('Int64')
    df['OrderQuantity'] = pd.to_numeric(df['OrderQuantity'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['OrderDate', 'ProductKey', 'CustomerKey', 'TerritoryKey'])
    
    sales_trans = Table('sales_trans', metadata,
        Column('order_date', TIMESTAMP),
        Column('stock_date', TIMESTAMP),
        Column('order_number', String(50)),
        Column('product_id', Integer),
        Column('customer_id', Integer),
        Column('territory_id', Integer),
        Column('order_line_item', Integer),
        Column('order_quantity', Integer),
        Column('created_at', TIMESTAMP, server_default=func.now()),
        Column('updated_at', TIMESTAMP, server_default=func.now(), onupdate=func.now())
    )
    sales_trans.create(engine)
    
    df.rename(columns={
        'OrderDate': 'order_date',
        'StockDate': 'stock_date',
        'OrderNumber': 'order_number',
        'ProductKey': 'product_id',
        'CustomerKey': 'customer_id',
        'TerritoryKey': 'territory_id',
        'OrderLineItem': 'order_line_item',
        'OrderQuantity': 'order_quantity'
    }, inplace=True)
    df[['order_date', 'stock_date', 'order_number', 'product_id', 'customer_id', 'territory_id', 'order_line_item', 'order_quantity']].to_sql('sales_trans', engine, if_exists='append', index=False)
    print(f"Cleaned sales: {len(df)} rows inserted")

if __name__ == "__main__":
    cleanup_trans_tables()
    clean_territories()
    clean_product_categories()
    clean_product_subcategories()
    clean_products()
    clean_sales()
    print("All tables cleaned and normalized!")
