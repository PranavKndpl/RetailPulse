import pandas as pd
import numpy as np
from typing import BinaryIO


class DataIngestor:
    SUPPORTED_EXTENSIONS = (".csv", ".xlsx")

    def ingest(self, file_obj: BinaryIO) -> pd.DataFrame:
 
        self._validate_file(file_obj)

        df = self._read_file(file_obj)
        df = self._standardize_columns(df)
        df = self._cast_types(df)
        df = self._handle_missing(df)

        if df.empty:
            raise ValueError("Uploaded file contains no usable rows after cleaning.")

        return df


    def _validate_file(self, file_obj):
        name = file_obj.name.lower()
        if not name.endswith(self.SUPPORTED_EXTENSIONS):
            raise ValueError(
                "Unsupported file type. Please upload a CSV or Excel (.xlsx) file."
            )

    def _read_file(self, file_obj) -> pd.DataFrame:
        name = file_obj.name.lower()

        if name.endswith(".xlsx"):
            try:
                return pd.read_excel(file_obj)
            except ImportError:
                raise ValueError(
                    "Excel support is unavailable. Please ensure 'openpyxl' is installed."
                )

        for encoding in ("utf-8", "latin1", "ISO-8859-1", "cp1252"):
            try:
                return pd.read_csv(file_obj, encoding=encoding)
            except UnicodeDecodeError:
                file_obj.seek(0)

        raise ValueError("Could not read CSV file due to unsupported encoding.")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^\w_]", "", regex=True)
        )
        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in df.columns:
            # Datetime inference
            if any(k in col for k in ("date", "time", "day")):
                df[col] = pd.to_datetime(df[col], errors="coerce")

            # Numeric inference
            elif df[col].dtype == "object":
                sample = df[col].dropna().astype(str).head(10)
                if any(any(c.isdigit() for c in s) for s in sample):
                    cleaned = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("$", "", regex=False)
                    )
                    numeric = pd.to_numeric(cleaned, errors="coerce")
                    df[col] = numeric.where(numeric.notna(), df[col])

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
        if len(date_cols) > 0:
            df = df.dropna(subset=date_cols)

        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].fillna(0)

        obj_cols = df.select_dtypes(include=["object"]).columns
        df[obj_cols] = df[obj_cols].fillna("Unknown")

        return df
