import pandas as pd

class DataIngestor:
    def ingest(self, file_obj) -> pd.DataFrame:
        if file_obj.name.endswith(".csv"):
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj)

        df.columns = (
            df.columns
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^\w_]", "", regex=True)
        )

        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

        df = df.dropna(how="all")

        return df
