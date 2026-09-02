import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(
    __name__,
    template_folder="WEBSITE/templates",
    static_folder="WEBSITE/static"
)

model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")

# -------------------------
# Login Page
# -------------------------
@app.route("/")
def login():
    return render_template("login.html")

# -------------------------
# Dashboard
# -------------------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# -------------------------
# Prediction Page
# -------------------------
@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

# -------------------------
# History Page
# -------------------------
@app.route("/history")
def history():
    return render_template("history.html")

# -------------------------
# Analytics Page
# -------------------------
@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

# -------------------------
# Reports Page
# -------------------------
@app.route("/reports")
def reports():
    return render_template("reports.html")

# -------------------------
# Devices Page
# -------------------------
@app.route("/devices")
def devices():
    return render_template("devices.html")

# -------------------------
# Profile Page
# -------------------------
@app.route("/profile")
def profile():
    return render_template("profile.html")

# -------------------------
# Predict Route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    device = {
        "Device_Type": request.form["Device_Type"],
        "Age": int(request.form["Age"]),
        "Manufacturer": request.form["Manufacturer"],
        "Model": request.form["Model"],
        "Country": request.form["Country"],
        "Maintenance_Cost": float(request.form["Maintenance_Cost"]),
        "Downtime": float(request.form["Downtime"]),
        "Maintenance_Frequency": int(request.form["Maintenance_Frequency"]),
        "Failure_Event_Count": int(request.form["Failure_Event_Count"]),
        "Maintenance_Class": int(request.form["Maintenance_Class"]),
        "Operational_Hours_Est": float(request.form["Operational_Hours_Est"]),
        "Expected_Lifespan_Est": float(request.form["Expected_Lifespan_Est"]),
        "MTBF": float(request.form["MTBF"]),
        "Cost_Per_Hour": float(request.form["Cost_Per_Hour"]),
        "Lifespan_Usage_Ratio": float(request.form["Lifespan_Usage_Ratio"])
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([device])

    # Encode categorical columns
    categorical_columns = [
        "Device_Type",
        "Manufacturer",
        "Model",
        "Country"
    ]

    for column in categorical_columns:
        input_df[column] = encoders[column].transform(input_df[column])

    # Predict
    prediction = model.predict(input_df)[0]

    # Convert prediction into label
    if prediction == 0:
        risk = "Low Risk"
        reliability = 95
        recommendation = "Device is operating normally. Continue routine monitoring."

    elif prediction == 1:
        risk = "Medium Risk"
        reliability = 72
        recommendation = "Schedule preventive maintenance soon."

    else:
        risk = "High Risk"
        reliability = 45
        recommendation = "Immediate maintenance required. Inspect the device before further use."

    # Send result to HTML page
    return render_template(
        "result.html",
        device=device["Device_Type"],
        risk=risk,
        reliability=reliability,
        recommendation=recommendation
    )



if __name__ == "__main__":
    app.run(debug=True)