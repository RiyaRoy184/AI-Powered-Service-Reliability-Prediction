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
    session,
    jsonify
)

import pandas as pd
import joblib
import os
import sqlite3

from datetime import datetime


# ============================================================
# FLASK CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

WEBSITE_DIR = os.path.join(
    BASE_DIR,
    "WEBSITE"
)

app = Flask(
    __name__,
    template_folder=os.path.join(
        WEBSITE_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        WEBSITE_DIR,
        "static"
    )
)

app.secret_key = "smart-healthcare-secret-key"


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "MODELS",
    "model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "MODELS",
    "encoders.pkl"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "DATASET",
    "processed_device_level.csv"
)

DATABASE_PATH = os.path.join(
    WEBSITE_DIR,
    "database",
    "hospital.db"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Machine Learning Model Loaded Successfully"
    )

except Exception as e:

    model = None

    print(
        "Error loading model:"
    )

    print(e)


# ============================================================
# LOAD ENCODERS
# ============================================================

try:

    encoders = joblib.load(
        ENCODER_PATH
    )

    print(
        "Encoders Loaded Successfully"
    )

except Exception as e:

    encoders = {}

    print(
        "Error loading encoders:"
    )

    print(e)


# ============================================================
# LOAD DATASET
# ============================================================

try:

    dataset = pd.read_csv(
        DATASET_PATH
    )

    print(
        "Dataset Loaded Successfully"
    )

    print(
        "Dataset Shape:",
        dataset.shape
    )

except Exception as e:

    dataset = pd.DataFrame()

    print(
        "Error loading dataset:"
    )

    print(e)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

def init_db():

    os.makedirs(
        os.path.dirname(
            DATABASE_PATH
        ),
        exist_ok=True
    )

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            device_type TEXT,
            age REAL,
            manufacturer TEXT,
            model TEXT,
            country TEXT,

            maintenance_cost REAL,
            downtime REAL,
            maintenance_frequency INTEGER,
            failure_event_count INTEGER,
            maintenance_class INTEGER,

            operational_hours REAL,
            expected_lifespan REAL,
            mtbf REAL,
            cost_per_hour REAL,
            lifespan_usage_ratio REAL,

            risk_level TEXT,
            reliability_score INTEGER,
            recommendation TEXT,

            created_at TEXT

        )
    """)

    conn.commit()

    conn.close()

    print(
        "Database initialized successfully."
    )


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(data):

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (

            device_type,
            age,
            manufacturer,
            model,
            country,

            maintenance_cost,
            downtime,
            maintenance_frequency,
            failure_event_count,
            maintenance_class,

            operational_hours,
            expected_lifespan,
            mtbf,
            cost_per_hour,
            lifespan_usage_ratio,

            risk_level,
            reliability_score,
            recommendation,
            created_at

        )

        VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
    """, (

        data["device_type"],
        data["age"],
        data["manufacturer"],
        data["model"],
        data["country"],

        data["maintenance_cost"],
        data["downtime"],
        data["maintenance_frequency"],
        data["failure_event_count"],
        data["maintenance_class"],

        data["operational_hours"],
        data["expected_lifespan"],
        data["mtbf"],
        data["cost_per_hour"],
        data["lifespan_usage_ratio"],

        data["risk_level"],
        data["reliability_score"],
        data["recommendation"],
        data["created_at"]

    ))

    conn.commit()

    conn.close()

    print(
        "Prediction saved successfully."
    )


# ============================================================
# GET PREDICTION HISTORY
# ============================================================

def get_prediction_history():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return records


# ============================================================
# GET DATABASE STATISTICS
# ============================================================

def get_prediction_statistics():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = conn.cursor()


    # Total predictions

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
    """)

    total_predictions = cursor.fetchone()[0]


    # Low Risk

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE risk_level = 'Low Risk'
    """)

    low_risk = cursor.fetchone()[0]


    # Medium Risk

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE risk_level = 'Medium Risk'
    """)

    medium_risk = cursor.fetchone()[0]


    # High Risk

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE risk_level = 'High Risk'
    """)

    high_risk = cursor.fetchone()[0]


    # Average Reliability

    cursor.execute("""
        SELECT AVG(reliability_score)
        FROM predictions
    """)

    average_reliability = cursor.fetchone()[0]

    conn.close()


    if average_reliability is None:

        average_reliability = 0

    else:

        average_reliability = round(
            average_reliability,
            2
        )


    return {

        "total_predictions":
            total_predictions,

        "low_risk":
            low_risk,

        "medium_risk":
            medium_risk,

        "high_risk":
            high_risk,

        "average_reliability":
            average_reliability

    }


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
# CONVERT ENCODER VALUE
# ============================================================

def convert_encoder_value(
    encoder,
    value
):

    classes = list(
        encoder.classes_
    )


    # Direct match

    if value in classes:

        return value


    # String comparison

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

            except (
                ValueError,
                TypeError
            ):

                pass

    except (
        ValueError,
        TypeError
    ):

        pass


    raise ValueError(
        f"Value '{value}' is not present in the trained encoder."
    )


# ============================================================
# ENCODE INPUT DATA
# ============================================================

def encode_input(input_df):

    categorical_columns = [

        "Device_Type",
        "Manufacturer",
        "Model",
        "Country",
        "Maintenance_Frequency",
        "Maintenance_Class"

    ]


    for column in categorical_columns:

        if (
            column in input_df.columns
            and column in encoders
        ):

            encoder = encoders[column]

            value = input_df.loc[
                0,
                column
            ]

            converted_value = (
                convert_encoder_value(
                    encoder,
                    value
                )
            )

            input_df.loc[
                0,
                column
            ] = converted_value

            input_df[column] = (
                encoder.transform(
                    input_df[column]
                )
            )


    return input_df


# ============================================================
# LOGIN PROTECTION
# ============================================================

def check_login():

    return session.get(
        "logged_in",
        False
    )


# ============================================================
# LOGIN PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()


        if (
            username == "admin"
            and password == "admin123"
        ):

            session["logged_in"] = True

            # Keep login username separate
            # from profile information

            session["login_username"] = "admin"

            return redirect(
                url_for("dashboard")
            )


        return render_template(
            "login.html",
            error="Invalid username or password."
        )


    return render_template(
        "login.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if not check_login():

        return redirect(
            url_for("login")
        )


    statistics = (
        get_prediction_statistics()
    )


    profile_data = session.get(
        "profile_data",
        {}
    )


    display_name = profile_data.get(
        "full_name",
        "Administrator"
    )


    return render_template(

        "dashboard.html",

        total_predictions=
            statistics["total_predictions"],

        low_risk=
            statistics["low_risk"],

        medium_risk=
            statistics["medium_risk"],

        high_risk=
            statistics["high_risk"],

        average_reliability=
            statistics["average_reliability"],

        username=
            display_name

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
# PROCESS PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if not check_login():

        return redirect(
            url_for("login")
        )


    if model is None:

        return render_template(

            "result.html",

            error=
                "Machine learning model could not be loaded."

        )


    try:

        # ====================================================
        # GET FORM VALUES
        # ====================================================

        device_type = request.form.get(
            "Device_Type",
            ""
        )

        age = request.form.get(
            "Age",
            ""
        )

        manufacturer = request.form.get(
            "Manufacturer",
            ""
        )

        model_name = request.form.get(
            "Model",
            ""
        )

        country = request.form.get(
            "Country",
            ""
        )

        maintenance_cost = request.form.get(
            "Maintenance_Cost",
            ""
        )

        downtime = request.form.get(
            "Downtime",
            ""
        )

        maintenance_frequency = request.form.get(
            "Maintenance_Frequency",
            ""
        )

        failure_event_count = request.form.get(
            "Failure_Event_Count",
            ""
        )

        maintenance_class = request.form.get(
            "Maintenance_Class",
            ""
        )

        operational_hours = request.form.get(
            "Operational_Hours_Est",
            ""
        )

        expected_lifespan = request.form.get(
            "Expected_Lifespan_Est",
            ""
        )

        mtbf = request.form.get(
            "MTBF",
            ""
        )

        cost_per_hour = request.form.get(
            "Cost_Per_Hour",
            ""
        )

        lifespan_usage_ratio = request.form.get(
            "Lifespan_Usage_Ratio",
            ""
        )


        # ====================================================
        # CREATE DATAFRAME
        # ====================================================

        input_data = {

            "Device_Type": [
                device_type
            ],

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


        # ====================================================
        # ENCODE DATA
        # ====================================================

        input_df = encode_input(
            input_df
        )


        # ====================================================
        # MATCH MODEL FEATURE ORDER
        # ====================================================

        if hasattr(
            model,
            "feature_names_in_"
        ):

            feature_order = list(
                model.feature_names_in_
            )

            input_df = input_df[
                feature_order
            ]


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction_value = model.predict(
            input_df
        )[0]

        prediction_value = int(
            prediction_value
        )


        print(
            "Predicted Class:",
            prediction_value
        )


        # ====================================================
        # GET RISK INFORMATION
        # ====================================================

        risk_info = RISK_INFORMATION.get(

            prediction_value,

            {

                "risk":
                    "Unknown Risk",

                "reliability":
                    0,

                "recommendation":
                    "Unable to generate a maintenance recommendation.",

                "color":
                    "secondary",

                "icon":
                    "fa-question"

            }

        )


        # ====================================================
        # SAVE PREDICTION
        # ====================================================

        prediction_data = {

            "device_type":
                device_type,

            "age":
                float(age),

            "manufacturer":
                manufacturer,

            "model":
                model_name,

            "country":
                country,

            "maintenance_cost":
                float(maintenance_cost),

            "downtime":
                float(downtime),

            "maintenance_frequency":
                int(maintenance_frequency),

            "failure_event_count":
                int(failure_event_count),

            "maintenance_class":
                int(maintenance_class),

            "operational_hours":
                float(operational_hours),

            "expected_lifespan":
                float(expected_lifespan),

            "mtbf":
                float(mtbf),

            "cost_per_hour":
                float(cost_per_hour),

            "lifespan_usage_ratio":
                float(lifespan_usage_ratio),

            "risk_level":
                risk_info["risk"],

            "reliability_score":
                risk_info["reliability"],

            "recommendation":
                risk_info["recommendation"],

            "created_at":
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )

        }


        save_prediction(
            prediction_data
        )


        # ====================================================
        # RESULT PAGE
        # ====================================================

        return render_template(

            "result.html",

            device=device_type,

            risk=risk_info["risk"],

            reliability=
                risk_info["reliability"],

            recommendation=
                risk_info["recommendation"],

            risk_color=
                risk_info["color"],

            risk_icon=
                risk_info["icon"],

            prediction_class=
                prediction_value,

            input_data={

                "Device Type":
                    device_type,

                "Age":
                    age,

                "Manufacturer":
                    manufacturer,

                "Model":
                    model_name,

                "Country":
                    country,

                "Maintenance Cost":
                    maintenance_cost,

                "Downtime":
                    downtime,

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

                "MTBF":
                    mtbf,

                "Cost Per Hour":
                    cost_per_hour,

                "Lifespan Usage Ratio":
                    lifespan_usage_ratio

            }

        )


    except Exception as e:

        print(
            "Prediction Error:",
            e
        )

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
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():

    if not check_login():

        return redirect(
            url_for("login")
        )


    history_data = (
        get_prediction_history()
    )


    return render_template(

        "history.html",

        history=history_data

    )


# ============================================================
# CLEAR PREDICTION HISTORY
# ============================================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    if not check_login():

        return redirect(
            url_for("login")
        )


    try:

        conn = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM predictions
        """)

        conn.commit()

        conn.close()

        print(
            "Prediction history cleared successfully."
        )

    except Exception as e:

        print(
            "Error clearing history:",
            e
        )


    return redirect(
        url_for("history")
    )


# ============================================================
# ANALYTICS PAGE
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
        dt_accuracy=70.20,
        svm_accuracy=83.53,

        rf_roc_auc=99.34,
        dt_roc_auc=85.84,
        svm_roc_auc=94.37,

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
# ABOUT PROJECT
# ============================================================

@app.route("/about")
def about():

    if not check_login():

        return redirect(
            url_for("login")
        )


    return render_template(
        "about.html"
    )


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
def reports():

    if not check_login():

        return redirect(
            url_for("login")
        )


    statistics = (
        get_prediction_statistics()
    )


    return render_template(

        "reports.html",

        total_predictions=
            statistics["total_predictions"],

        low_risk=
            statistics["low_risk"],

        medium_risk=
            statistics["medium_risk"],

        high_risk=
            statistics["high_risk"]

    )


# ============================================================
# DEVICES PAGE
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

        device_records = (
            dataset.head(100)
            .to_dict(
                orient="records"
            )
        )


    return render_template(

        "devices.html",

        devices=device_records

    )


# ============================================================
# PROFILE PAGE
# ============================================================
# IMPORTANT:
# methods=["GET", "POST"] allows the Save Changes button
# to submit the form using POST.
# ============================================================

@app.route(
    "/profile",
    methods=["GET", "POST"]
)
def profile():

    # Check login

    if not check_login():

        return redirect(
            url_for("login")
        )


    # ========================================================
    # SAVE PROFILE
    # ========================================================

    if request.method == "POST":

        # Get existing profile information

        profile_data = session.get(
            "profile_data",
            {}
        ).copy()


        # Get email

        email = request.form.get(
            "email",
            ""
        ).strip()


        # Get role

        role = request.form.get(
            "role",
            "Hospital Administrator"
        ).strip()


        # ----------------------------------------------------
        # Save only profile information
        # Login username and password are NOT changed.
        # ----------------------------------------------------

        profile_data["email"] = email

        profile_data["role"] = role


        # Save profile information in session

        session["profile_data"] = profile_data

        session.modified = True


        print(
            "Profile updated successfully."
        )


        # Return to profile page

        return redirect(
            url_for("profile")
        )


    # ========================================================
    # DISPLAY PROFILE
    # ========================================================

    profile_data = session.get(
        "profile_data",
        {}
    )


    # Login username always remains admin

    login_username = session.get(
        "login_username",
        "admin"
    )


    return render_template(

        "profile.html",

        username=login_username,

        email=profile_data.get(
            "email",
            "admin@smarthealthcare.com"
        ),

        role=profile_data.get(
            "role",
            "Hospital Administrator"
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

        "status":
            "running",

        "model_loaded":
            model is not None,

        "dataset_loaded":
            not dataset.empty,

        "database_exists":
            os.path.exists(
                DATABASE_PATH
            )

    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    # Initialize database

    init_db()


    print()
    print(
        "=============================================="
    )

    print(
        " AI POWERED SERVICE RELIABILITY PREDICTION"
    )

    print(
        "        SMART HEALTHCARE SYSTEM"
    )

    print(
        "=============================================="
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "Username: admin"
    )

    print(
        "Password: admin123"
    )

    print(
        "=============================================="
    )

    print()


    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )