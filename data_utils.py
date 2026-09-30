"""
data_utils.py

Basic data loading and inspection utilities for the EAF POC.
"""

from pathlib import Path

import pandas as pd


def load_data(path):
    """
    Load the EAF CSV dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the CSV file.

    Returns
    -------
    pandas.DataFrame
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got: {path.suffix}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "The CSV file was loaded but contains no rows."
        )

    # Remove completely empty columns.
    df = df.dropna(
        axis=1,
        how="all",
    )

    # Normalise column names.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def inspect_data(df):
    """
    Return a compact textual summary of the dataset.

    This is intentionally simple for the POC.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    lines = []

    lines.append(
        f"Rows: {len(df):,}"
    )

    lines.append(
        f"Columns: {len(df.columns):,}"
    )

    lines.append(
        f"Duplicate rows: "
        f"{df.duplicated().sum():,}"
    )

    lines.append("")

    lines.append("Column information:")

    info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "missing": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "missing_pct": [
                round(
                    df[column].isna().mean() * 100,
                    2,
                )
                for column in df.columns
            ],
            "unique": [
                int(df[column].nunique(
                    dropna=True
                ))
                for column in df.columns
            ],
        }
    )

    lines.append(
        info.to_string(
            index=False
        )
    )

    return "\n".join(lines)


def numeric_columns(df):
    """
    Return numeric columns.
    """

    return list(
        df.select_dtypes(
            include="number"
        ).columns
    )


def missing_value_summary(df):
    """
    Return missing-value statistics.
    """

    result = pd.DataFrame(
        {
            "missing_count":
                df.isna().sum(),

            "missing_pct":
                df.isna().mean() * 100,
        }
    )

    return result.sort_values(
        "missing_count",
        ascending=False,
    )


def remove_duplicate_rows(df):
    """
    Remove exact duplicate rows.
    """

    return df.drop_duplicates(
        keep="first"
    ).reset_index(
        drop=True
    )
