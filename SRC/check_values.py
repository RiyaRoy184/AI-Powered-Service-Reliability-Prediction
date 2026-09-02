import pandas as pd

df = pd.read_csv("dataset/processed_device_level.csv")

print("Device_Type")
print(df["Device_Type"].unique())

print("\nManufacturer")
print(df["Manufacturer"].unique())

print("\nModel")
print(df["Model"].unique())

print("\nCountry")
print(df["Country"].unique())

print("\nMaintenance_Frequency")
print(df["Maintenance_Frequency"].unique())

print("\nMaintenance_Class")
print(df["Maintenance_Class"].unique())