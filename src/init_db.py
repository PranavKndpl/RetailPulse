import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

"""
For every delivered order, combine order info, item prices, product category, and customer reviews into one result.
"""

SQL_CREATE_VIEW = """
DROP VIEW IF EXISTS master_orders_view;

CREATE VIEW master_orders_view AS
SELECT 
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    
    -- Calculate Delay (Negative = Early, Positive = Late)
    (DATE_PART('day', o.order_delivered_customer_date::timestamp) - 
     DATE_PART('day', o.order_estimated_delivery_date::timestamp)) AS delivery_delay_days,
    
    i.price,
    i.freight_value,
    p.product_category_name,
    r.review_score,
    r.review_comment_message

FROM olist_orders_dataset o
LEFT JOIN olist_order_items_dataset i ON o.order_id = i.order_id
LEFT JOIN olist_products_dataset p ON i.product_id = p.product_id
LEFT JOIN olist_order_reviews_dataset r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered';
"""

def init_db():
    print("Initializing Database Views...")
    with engine.connect() as conn:
        conn.execute(text(SQL_CREATE_VIEW))
        conn.commit()
    print("Master View created.")
    

    df = pd.read_sql("SELECT * FROM master_orders_view LIMIT 3", engine)
    print("\nData Preview (First 3 Rows):")
    print(df[['order_id', 'price', 'review_score']])

if __name__ == "__main__":
    init_db()