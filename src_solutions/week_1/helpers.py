"""
Helper utilities for EHR data processing.
"""

def get_col(df, name):
    """Find column by name, case-insensitive."""
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None
