import pandas as pd
from sqlalchemy import text

class DataRepository:
    def __init__(self, engine):
        self.engine = engine

    def save_table(self, df: pd.DataFrame, table_name: str):
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)

    def list_tables(self):
        query = text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        tables = pd.read_sql(query, self.engine)
        return tables["table_name"].tolist() if not tables.empty else []

    def load_table(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql(f'SELECT * FROM "{table_name}"', self.engine)

    def get_columns(self, table_name: str):
        df = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 0', self.engine)
        return df.columns.tolist()

    def merge_tables(self, left: str, right: str, join_col: str) -> pd.DataFrame:
        df_left = self.load_table(left)
        df_right = self.load_table(right)
        return pd.merge(df_left, df_right, on=join_col, how="inner")






