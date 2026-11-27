import pandas as pd

# 1. Load the dataset
file_path = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
df = pd.read_csv(file_path)

# 2. Check how many rows (URLs) and columns (features) we have
print("Dataset shape (rows, columns):", df.shape)

# 3. See the names of all columns
print("\nColumn names:")
print(df.columns.tolist())

# 4. Look at the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 5. Check missing values
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 6. Check the target label balance (1 = legitimate, 0 = phishing in PhiUSIIL)
if "label" in df.columns:
    print("\nLabel counts:")
    print(df["label"].value_counts())