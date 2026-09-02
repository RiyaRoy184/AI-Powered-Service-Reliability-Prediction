import joblib
import pandas as pd
model = joblib.load("models/model.pkl")

encoders = joblib.load("models/encoders.pkl")

device = {
    "Device_Type": "MRI Scanner",
    "Age": 5,
    "Manufacturer": "MedEquip",
    "Model": "Model-300",
    "Country": "USA",
    "Maintenance_Cost": 12000,
    "Downtime": 15,
    "Maintenance_Frequency": 3,
    "Failure_Event_Count": 2,
    "Maintenance_Class": 1,
    "Operational_Hours_Est": 15000,
    "Expected_Lifespan_Est": 60000,
    "MTBF": 4200,
    "Cost_Per_Hour": 250,
    "Lifespan_Usage_Ratio": 0.25
}
input_df = pd.DataFrame([device])
categorical_columns = [
    "Device_Type",
    "Manufacturer",
    "Model",
    "Country"
]
for column in categorical_columns:
    input_df[column] = encoders[column].transform(input_df[column])

prediction = model.predict(input_df)
if prediction[0] == 0:
    print("Predicted Risk : Low Risk")

elif prediction[0] == 1:
    print("Predicted Risk : Medium Risk")

else:
    print("Predicted Risk : High Risk")



#Reliability Score

if prediction[0] == 0:
    reliability = 95

elif prediction[0] == 1:
    reliability = 75

else:
    reliability = 45

print("Reliability Score:", reliability, "%")

#maintence recommendation

if prediction[0] == 0:
    recommendation = "Device is operating normally. Continue routine maintenance."

elif prediction[0] == 1:
    recommendation = "Schedule preventive maintenance within the next 7 days."

else:
    recommendation = "Immediate maintenance required. Inspect the device before further use."

print("Maintenance Recommendation:")
print(recommendation)