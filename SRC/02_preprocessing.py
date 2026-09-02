import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/processed_device_level.csv")

# Display original columns
print("Original Columns:")
print(df.columns)

# Remove unnecessary columns
df = df.drop(
    columns=[
        "Device_ID",
        "Purchase_Date",
        "Maintenance_Report"
    ]
)

# Display remaining columns
print("\nColumns After Removing:")
print(df.columns)

# Separate input features and output feature

X = df.drop("Risk_Class_Label", axis=1)

y = df["Risk_Class_Label"]

print("\nInput Features (X):")
print(X.head())

print("\nOutput Feature (y):")
print(y.head())

from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder object
encoder = LabelEncoder()

# List of categorical columns
categorical_columns = [
    "Device_Type",
    "Manufacturer",
    "Model",
    "Country",
    "Maintenance_Frequency",
    "Maintenance_Class",
    "Risk_Class"
]

# Encode each categorical column
for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])

print("\nEncoded Input Features:")
print(X.head())

print("\nUnique values in Risk_Class_Label:")
print(df["Risk_Class_Label"].unique())