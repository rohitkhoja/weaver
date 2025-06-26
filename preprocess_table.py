import pandas as pd
import re
from sqlalchemy.dialects.mysql import dialect

def rename_sql_keywords(df):
    sql_keywords = set(dialect().preparer.reserved_words)
    new_columns = []
    for col in df.columns:
        if col.lower() in sql_keywords:
            new_col = f"{col}_1"
            counter = 1
            while new_col.upper() in sql_keywords or new_col in new_columns:
                counter += 1
                new_col = f"{col}_{counter}"
            new_columns.append(new_col)
        else:
            new_columns.append(col)
    return new_columns
def clean_table(table_name, df):
    # Clean table name
    clean_table_name = re.sub(r'[^\w\s]', '', table_name)  # Remove special characters
    clean_table_name = re.sub(r'\s+', '_', clean_table_name)  # Replace spaces with underscores
    if clean_table_name[0].isdigit():
        clean_table_name = f"table_{clean_table_name}"  # Prefix if name starts with a digit
    clean_table_name = clean_table_name[:64]
    # Clean column names
    clean_columns = []
    for col in df.columns:
        clean_col = re.sub(r'[^\w\s]', '', col)  # Remove special characters
        clean_col = re.sub(r'\s+', '_', clean_col)  # Replace spaces with underscores
        clean_columns.append(clean_col)

    df.columns = clean_columns
    df.columns = rename_sql_keywords(df)
    df = df.where(pd.notnull(df), None)

    return clean_table_name, df