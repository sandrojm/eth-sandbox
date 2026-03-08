"""
Helper utilities for EHR data processing.
"""


def get_col(df, name):
    """Find column by name, case-insensitive, handles underscores."""
    name_lower = name.lower().replace('_', '')
    for col in df.columns:
        col_lower = col.lower().replace('_', '')
        if col_lower == name_lower:
            return col
    return None
