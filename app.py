# ============================================================
# AI POWERED SERVICE RELIABILITY PREDICTION
# SMART HEALTHCARE
# ============================================================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)

import pandas as pd
import joblib
import os
from datetime import datetime


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(
    __name__,
    template_folder="WEBSITE/templates",
    static_folder="WEBSITE/static"
)

# Secret key for login session
app.secret_key = "smart-healthcare-secret-key"


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/encoders.pkl"
DATASET_PATH = "dataset/processed_device_level.csv"


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    print("Machine Learning Model Loaded Successfully")

except Exception as e:

    model = None

    print("Error loading model:")
    print(e)


# ============================================================
# LOAD ENCODERS
# ============================================================

try:

    encoders = joblib.load(ENCODER_PATH)

    print("Encoders Loaded Successfully")

except Exception as e:

    encoders = {}

    print("Error loading encoders:")
    print(e)


# ============================================================
# LOAD DATASET
# ============================================================

try:

    dataset = pd.read_csv(DATASET_PATH)

    print("Dataset Loaded Successfully")
    print("Dataset Shape:", dataset.shape)

except Exception as e:

    dataset = pd.DataFrame()

    print("Error loading dataset:")
    print(e)


# ============================================================
# TEMPORARY PREDICTION HISTORY
# ============================================================
#
# This stores predictions while the Flask application is running.
#
# Later, this can be replaced with SQLite/MySQL database.
#
# ============================================================

prediction_history = []


# ============================================================
# RISK INFORMATION
# ============================================================

RISK_INFORMATION = {

    0: {
        "risk": "Low Risk",
        "reliability": 95,
        "recommendation":
            "Device is operating normally. Continue routine monitoring.",
        "color": "success",
        "icon": "fa-circle-check"
    },

    1: {
        "risk": "Medium Risk",
        "reliability": 72,
        "recommendation":
            "Schedule preventive maintenance soon.",
        "color": "warning",
        "icon": "fa-triangle-exclamation"
    },

    2: {
        "risk": "High Risk",
        "reliability": 45,
        "recommendation":
            "Immediate maintenance required. Inspect the device before further use.",
        "color": "danger",
        "icon": "fa-circle-exclamation"
    }

}


# ============================================================
# HELPER FUNCTION
# ============================================================

def convert_encoder_value(encoder, value):
    """
    Converts form values into the format expected by LabelEncoder.

    This makes the prediction process more robust when an encoder
    contains strings or numeric values.
    """

    classes = list(encoder.classes_)

    # Direct match
    if value in classes:
        return value

    # Compare string representation
    for item in classes:

        if str(item) == str(value):
            return item

    # Numeric comparison
    try:

        numeric_value = float(value)

        for item in classes:

            try:

                if float(item) == numeric_value:
                    return item

            except (ValueError, TypeError):
                pass

    except (ValueError, TypeError):
        pass

    raise ValueError(
        f"Value '{value}' is not present in the trained encoder."
    )


# ============================================================
# HELPER FUNCTION
# ============================================================

def encode_input(input_df):
    """
    Applies the same categorical encoding used during training.
    """

    categorical_columns = [
        "Device_Type",
        "Manufacturer",
        "Model",
        "Country",
        "Maintenance_Frequency",
        "Maintenance_Class"
    ]

    for column in categorical_columns:

        if column in input_df.columns and column in encoders:

            encoder = encoders[column]

            value = input_df.loc[0, column]

            converted_value = convert_encoder_value(
                encoder,
                value
            )

            input_df.loc[0, column] = converted_value

            input_df[column] = encoder.transform(
                input_df[column]
            )

    return input_df


# ============================================================
# LOGIN
# ============================================================

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()

        password = request.form.get("password", "").strip()

        # Simple login for mini project
        if username == "admin" and password == "admin123":

            session["logged_in"] = True

            session["username"] = username

            return redirect(
                url_for("dashboard")
            )

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")


# ============================================================
# LOGIN PROTECTION
# ============================================================

def check_login():

    return session.get("logged_in", False)


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if not check_login():

        return redirect(
            url_for("login")
        )

    total_predictions = len(
        prediction_history
    )

    low_risk = sum(
        1 for item in prediction_history
        if item["risk"] == "Low Risk"
    )

    medium_risk = sum(
        1 for item in prediction_history
        if item["risk"] == "Medium Risk"
    )

    high_risk = sum(
        1 for item in prediction_history
        if item["risk"] == "High Risk"
    )

    if total_predictions > 0:

        average_reliability = round(
            sum(
                item["reliability"]
                for item in prediction_history
            ) / total_predictions,
            2
        )

    else:

        average_reliability = 0


    return render_template(
        "dashboard.html",

        total_predictions=total_predictions,

        low_risk=low_risk,

        medium_risk=medium_risk,

        high_risk=high_risk,

        average_reliability=average_reliability,

        username=session.get(
            "username",
            "Administrator"
        )
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route("/prediction")
def prediction():

    if not check_login():

        return redirect(
            url_for("login")
        )

    return render_template(
        "prediction.html"
    )


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if not check_login():

        return redirect(
            url_for("login")
        )


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        return render_template(
            "result.html",

            error="Machine learning model could not be loaded."
        )


    try:

        # ----------------------------------------------------
        # GET FORM DATA
        # ----------------------------------------------------

        device_type = request.form.get(
            "Device_Type"
        )

        age = request.form.get(
            "Age"
        )

        manufacturer = request.form.get(
            "Manufacturer"
        )

        model_name = request.form.get(
            "Model"
        )

        country = request.form.get(
            "Country"
        )

        maintenance_cost = request.form.get(
            "Maintenance_Cost"
        )

        downtime = request.form.get(
            "Downtime"
        )

        maintenance_frequency = request.form.get(
            "Maintenance_Frequency"
        )

        failure_event_count = request.form.get(
            "Failure_Event_Count"
        )

        maintenance_class = request.form.get(
            "Maintenance_Class"
        )

        operational_hours = request.form.get(
            "Operational_Hours_Est"
        )

        expected_lifespan = request.form.get(
            "Expected_Lifespan_Est"
        )

        mtbf = request.form.get(
            "MTBF"
        )

        cost_per_hour = request.form.get(
            "Cost_Per_Hour"
        )

        lifespan_usage_ratio = request.form.get(
            "Lifespan_Usage_Ratio"
        )


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = {

            "Device_Type": [device_type],

            "Age": [
                float(age)
            ],

            "Manufacturer": [
                manufacturer
            ],

            "Model": [
                model_name
            ],

            "Country": [
                country
            ],

            "Maintenance_Cost": [
                float(maintenance_cost)
            ],

            "Downtime": [
                float(downtime)
            ],

            "Maintenance_Frequency": [
                int(maintenance_frequency)
            ],

            "Failure_Event_Count": [
                int(failure_event_count)
            ],

            "Maintenance_Class": [
                int(maintenance_class)
            ],

            "Operational_Hours_Est": [
                float(operational_hours)
            ],

            "Expected_Lifespan_Est": [
                float(expected_lifespan)
            ],

            "MTBF": [
                float(mtbf)
            ],

            "Cost_Per_Hour": [
                float(cost_per_hour)
            ],

            "Lifespan_Usage_Ratio": [
                float(lifespan_usage_ratio)
            ]
        }


        input_df = pd.DataFrame(
            input_data
        )


        # ----------------------------------------------------
        # ENCODE CATEGORICAL VARIABLES
        # ----------------------------------------------------

        input_df = encode_input(
            input_df
        )


        # ----------------------------------------------------
        # MATCH TRAINING FEATURE ORDER
        # ----------------------------------------------------

        if hasattr(model, "feature_names_in_"):

            feature_order = list(
                model.feature_names_in_
            )

            input_df = input_df[
                feature_order
            ]


        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        print("\n===================================")

        print("Prediction Input")

        print("===================================")

        print(input_df)


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction_value = model.predict(
            input_df
        )[0]


        # Convert NumPy integer to normal integer
        prediction_value = int(
            prediction_value
        )


        print(
            "Predicted Class:",
            prediction_value
        )


        # ----------------------------------------------------
        # GET RISK INFORMATION
        # ----------------------------------------------------

        if prediction_value in RISK_INFORMATION:

            risk_info = RISK_INFORMATION[
                prediction_value
            ]

        else:

            risk_info = {

                "risk": "Unknown Risk",

                "reliability": 0,

                "recommendation":
                    "Unable to generate a maintenance recommendation.",

                "color": "secondary",

                "icon": "fa-question"
            }


        # ----------------------------------------------------
        # DEVICE DISPLAY NAME
        # ----------------------------------------------------

        device_display = device_type


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history_item = {

            "device": device_display,

            "risk": risk_info["risk"],

            "reliability":
                risk_info["reliability"],

            "recommendation":
                risk_info["recommendation"],

            "date":
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )
        }


        prediction_history.insert(
            0,
            history_item
        )


        # ----------------------------------------------------
        # RENDER RESULT
        # ----------------------------------------------------

        return render_template(

            "result.html",

            device=device_display,

            risk=risk_info["risk"],

            reliability=risk_info[
                "reliability"
            ],

            recommendation=risk_info[
                "recommendation"
            ],

            risk_color=risk_info[
                "color"
            ],

            risk_icon=risk_info[
                "icon"
            ],

            prediction_class=prediction_value,

            input_data={
                "Device Type": device_type,
                "Age": age,
                "Manufacturer": manufacturer,
                "Model": model_name,
                "Country": country,
                "Maintenance Cost": maintenance_cost,
                "Downtime": downtime,
                "Maintenance Frequency":
                    maintenance_frequency,
                "Failure Event Count":
                    failure_event_count,
                "Maintenance Class":
                    maintenance_class,
                "Operational Hours":
                    operational_hours,
                "Expected Lifespan":
                    expected_lifespan,
                "MTBF": mtbf,
                "Cost Per Hour":
                    cost_per_hour,
                "Lifespan Usage Ratio":
                    lifespan_usage_ratio
            }

        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print("\nPrediction Error:")
        print(e)

        return render_template(

            "result.html",

            error=str(e)

        )


# ============================================================
# RESULT PAGE
# ============================================================

@app.route("/result")
def result():

    if not check_login():

        return redirect(
            url_for("login")
        )

    return render_template(
        "result.html"
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    if not check_login():

        return redirect(
            url_for("login")
        )

    return render_template(

        "history.html",

        history=prediction_history

    )


# ============================================================
# ANALYTICS
# ============================================================

@app.route("/analytics")
def analytics():

    if not check_login():

        return redirect(
            url_for("login")
        )


    return render_template(

        "analytics.html",

        rf_accuracy=94.63,

        dt_accuracy=70.02,

        svm_accuracy=83.53,

        rf_precision=89,

        rf_recall=97,

        rf_f1=92,

        dt_precision=58,

        dt_recall=70,

        dt_f1=61,

        svm_precision=81,

        svm_recall=61,

        svm_f1=62

    )


# ============================================================
# REPORTS
# ============================================================

@app.route("/reports")
def reports():

    if not check_login():

        return redirect(
            url_for("login")
        )


    total_predictions = len(
        prediction_history
    )

    low_risk = sum(
        1 for item in prediction_history
        if item["risk"] == "Low Risk"
    )

    medium_risk = sum(
        1 for item in prediction_history
        if item["risk"] == "Medium Risk"
    )

    high_risk = sum(
        1 for item in prediction_history
        if item["risk"] == "High Risk"
    )


    return render_template(

        "reports.html",

        total_predictions=
            total_predictions,

        low_risk=
            low_risk,

        medium_risk=
            medium_risk,

        high_risk=
            high_risk

    )


# ============================================================
# DEVICES
# ============================================================

@app.route("/devices")
def devices():

    if not check_login():

        return redirect(
            url_for("login")
        )


    if dataset.empty:

        device_records = []

    else:

        # Show maximum 100 records
        device_records = dataset.head(
            100
        ).to_dict(
            orient="records"
        )


    return render_template(

        "devices.html",

        devices=device_records

    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
def profile():

    if not check_login():

        return redirect(
            url_for("login")
        )


    return render_template(

        "profile.html",

        username=session.get(
            "username",
            "Administrator"
        )

    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {

        "status": "running",

        "model_loaded":
            model is not None,

        "dataset_loaded":
            not dataset.empty

    }


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")

    print("AI POWERED SERVICE RELIABILITY SYSTEM")

    print("==========================================")

    print("Starting Flask application...")

    print("Login:")
    print("Username: admin")
    print("Password: admin123")

    print("==========================================\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )