import pandas as pd
from pathlib import Path

# Define project directory
BASE_DIR = Path(__file__).resolve().parent

# Define data paths
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "churn-analysis.csv"
OUTPUT_FILE = DATA_DIR / "new_customers.csv"

# Load original dataset
df = pd.read_csv(INPUT_FILE, sep=";")

# Remove target variable and select five sample customers
new_customers = df.drop(columns=["churn"]).head(5)

# Save sample customers
new_customers.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

print("Sample dataset created successfully!")
print(f"File saved to: {OUTPUT_FILE}")