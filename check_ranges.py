import pandas as pd

df = pd.read_csv("dataset/processed_device_level.csv")

columns = [
    "Age",
    "Maintenance_Cost",
    "Downtime",
    "Maintenance_Frequency",
    "Failure_Event_Count",
    "Maintenance_Class",
    "Operational_Hours_Est",
    "Expected_Lifespan_Est",
    "MTBF",
    "Cost_Per_Hour",
    "Lifespan_Usage_Ratio"
]

print("\n===== DATASET RANGES =====\n")

for col in columns:
    print(
        f"{col}: "
        f"MIN = {df[col].min()} | "
        f"MAX = {df[col].max()}"
    )

print("\n===== UNIQUE INTEGER VALUES =====\n")

for col in [
    "Maintenance_Frequency",
    "Maintenance_Class"
]:
    print(f"{col}: {sorted(df[col].dropna().unique())}")