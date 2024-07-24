import pandas as pd

# Define paths
jsonl_file = 'corpus.jsonl'
parquet_file = 'corpus.parquet'

# Define schema (optional, but can improve performance)
schema = {
    "_id": str,
    "title": str,
    "text": str
}

# Read JSONL file into pandas DataFrame
df = pd.read_json(jsonl_file, lines=True, dtype=schema)

# Write DataFrame to Parquet file
df.to_parquet(parquet_file, index=False)

print("Conversion completed successfully.")
