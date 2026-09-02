import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/processed_device_level.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Display number of rows and columns
print("Shape:", df.shape)

# Check for missing values
print(df.isnull().sum())