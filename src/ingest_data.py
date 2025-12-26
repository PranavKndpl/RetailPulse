import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

def ingest_data():
    raw_data_path = "data/archive/"
    
    if not os.path.exists(raw_data_path):
        print(f"Error: Folder '{raw_data_path}' not found.")
        return

    files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
    
    if not files:
        print(f"No CSV files found in {raw_data_path}")
        return

    print(f"Found {len(files)} CSV files. Starting ingestion...")

    for file in files:
        table_name = file.replace(".csv", "")
        file_path = os.path.join(raw_data_path, file)
        
        print(f"   -> Processing {file}...", end=" ")
        
        try:
            df = pd.read_csv(file_path)
            
            df.to_sql(table_name, engine, if_exists='replace', index=False) # replace if exists - no duplicate data

            print("Done.")
            
        except Exception as e:
            print(f"Failed: {e}")

    print("\nAll data ingested successfully!")

if __name__ == "__main__":
    ingest_data()