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


    # --------------------------------------------------------
    # Total predictions
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
    """)

    total_predictions = cursor.fetchone()[0]


    # --------------------------------------------------------
    # Low Risk
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE risk_level = 'Low Risk'
    """)

    low_risk = cursor.fetchone()[0]


    # --------------------------------------------------------
    # Medium Risk
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE risk_level = 'Medium Risk'
    """)

    medium_risk = cursor.fetchone()[0]


    # --------------------------------------------------------
    # High Risk
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE risk_level = 'High Risk'
    """)

    high_risk = cursor.fetchone()[0]


    # --------------------------------------------------------
    # Average Reliability
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Direct match
    # --------------------------------------------------------

    if value in classes:

        return value


    # --------------------------------------------------------
    # String comparison
    # --------------------------------------------------------

    for item in classes:

        if str(item) == str(value):

            return item


    # --------------------------------------------------------
    # Numeric comparison
    # --------------------------------------------------------

    try:

        numeric_value = float(
            value
        )

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


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if model is None:

        return render_template(

            "result.html",

            error=
                "Machine learning model could not be loaded."

        )


    try:

        # ====================================================
        # GET BASIC FORM VALUES
        # ====================================================

        device_type = request.form.get(
            "Device_Type",
            ""
        ).strip()

        age = request.form.get(
            "Age",
            ""
        ).strip()

        manufacturer = request.form.get(
            "Manufacturer",
            ""
        ).strip()

        model_name = request.form.get(
            "Model",
            ""
        ).strip()

        country = request.form.get(
            "Country",
            ""
        ).strip()

        maintenance_cost = request.form.get(
            "Maintenance_Cost",
            ""
        ).strip()

        downtime = request.form.get(
            "Downtime",
            ""
        ).strip()

        maintenance_frequency = request.form.get(
            "Maintenance_Frequency",
            ""
        ).strip()

        failure_event_count = request.form.get(
            "Failure_Event_Count",
            ""
        ).strip()

        maintenance_class = request.form.get(
            "Maintenance_Class",
            ""
        ).strip()

        operational_hours = request.form.get(
            "Operational_Hours_Est",
            ""
        ).strip()

        expected_lifespan = request.form.get(
            "Expected_Lifespan_Est",
            ""
        ).strip()


        # ====================================================
        # CHECK REQUIRED VALUES
        # ====================================================

        required_values = {

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
                expected_lifespan

        }


        for field_name, field_value in required_values.items():

            if field_value == "":

                raise ValueError(
                    f"{field_name} is required."
                )


        # ====================================================
        # CONVERT BASIC NUMERIC VALUES
        # ====================================================

        age_value = float(
            age
        )

        maintenance_cost_value = float(
            maintenance_cost
        )

        downtime_value = float(
            downtime
        )

        maintenance_frequency_value = int(
            maintenance_frequency
        )

        failure_event_count_value = int(
            failure_event_count
        )

        maintenance_class_value = int(
            maintenance_class
        )

        operational_hours_value = float(
            operational_hours
        )

        expected_lifespan_value = float(
            expected_lifespan
        )


        # ====================================================
        # SERVER-SIDE VALIDATION
        # ====================================================

        # Age

        if age_value < 0:

            raise ValueError(
                "Age cannot be negative."
            )


        # Maintenance Cost

        if maintenance_cost_value < 0:

            raise ValueError(
                "Maintenance Cost cannot be negative."
            )


        # Downtime

        if downtime_value < 0:

            raise ValueError(
                "Downtime cannot be negative."
            )


        # Failure Event Count

        if failure_event_count_value < 0:

            raise ValueError(
                "Failure Event Count cannot be negative."
            )


        # Operational Hours

        if (
            operational_hours_value < 0
            or
            operational_hours_value > 108469
        ):

            raise ValueError(
                "Operational Hours must be between 0 and 108469."
            )


        # Expected Lifespan

        if expected_lifespan_value != 12:

            raise ValueError(
                "Expected Lifespan must be 12 years."
            )


        # Maintenance Frequency

        if (
            maintenance_frequency_value < 1
            or
            maintenance_frequency_value > 5
        ):

            raise ValueError(
                "Maintenance Frequency must be between 1 and 5."
            )


        # Maintenance Class

        if (
            maintenance_class_value < 1
            or
            maintenance_class_value > 3
        ):

            raise ValueError(
                "Maintenance Class must be between 1 and 3."
            )


        # ====================================================
        # AUTO-CALCULATE DERIVED FEATURES
        # ====================================================

        # ----------------------------------------------------
        # MTBF
        #
        # Current application formula:
        #
        # Operational Hours
        # -----------------------------
        # Failure Events + 1
        # ----------------------------------------------------

        mtbf_value = (
            operational_hours_value
            /
            (
                failure_event_count_value
                + 1
            )
        )


        # ----------------------------------------------------
        # COST PER HOUR
        #
        # Maintenance Cost
        # -----------------------------
        # Operational Hours
        # ----------------------------------------------------

        if operational_hours_value > 0:

            cost_per_hour_value = (
                maintenance_cost_value
                /
                operational_hours_value
            )

        else:

            cost_per_hour_value = 0


        # ----------------------------------------------------
        # LIFESPAN USAGE RATIO
        #
        # Age
        # -----------------------------
        # Expected Lifespan
        # ----------------------------------------------------

        if expected_lifespan_value > 0:

            lifespan_usage_ratio_value = (
                age_value
                /
                expected_lifespan_value
            )

        else:

            lifespan_usage_ratio_value = 0


        # ====================================================
        # ROUND DERIVED VALUES
        # ====================================================

        mtbf = round(
            mtbf_value
        )

        cost_per_hour = round(
            cost_per_hour_value,
            2
        )

        lifespan_usage_ratio = round(
            lifespan_usage_ratio_value,
            2
        )


        print()
        print(
            "----------------------------------------------"
        )

        print(
            "AUTO-CALCULATED VALUES"
        )

        print(
            "MTBF:",
            mtbf
        )

        print(
            "Cost Per Hour:",
            cost_per_hour
        )

        print(
            "Lifespan Usage Ratio:",
            lifespan_usage_ratio
        )

        print(
            "----------------------------------------------"
        )


        # ====================================================
        # CREATE MACHINE LEARNING DATAFRAME
        # ====================================================

        input_data = {

            "Device_Type": [
                device_type
            ],

            "Age": [
                age_value
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
                maintenance_cost_value
            ],

            "Downtime": [
                downtime_value
            ],

            "Maintenance_Frequency": [
                maintenance_frequency_value
            ],

            "Failure_Event_Count": [
                failure_event_count_value
            ],

            "Maintenance_Class": [
                maintenance_class_value
            ],

            "Operational_Hours_Est": [
                operational_hours_value
            ],

            "Expected_Lifespan_Est": [
                expected_lifespan_value
            ],

            "MTBF": [
                mtbf
            ],

            "Cost_Per_Hour": [
                cost_per_hour
            ],

            "Lifespan_Usage_Ratio": [
                lifespan_usage_ratio
            ]

        }


        input_df = pd.DataFrame(
            input_data
        )


        # ====================================================
        # ENCODE CATEGORICAL DATA
        # ====================================================

        input_df = encode_input(
            input_df
        )


        # ====================================================
        # MATCH TRAINING FEATURE ORDER
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
        # MACHINE LEARNING PREDICTION
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
        # PREPARE DATABASE DATA
        # ====================================================

        prediction_data = {

            "device_type":
                device_type,

            "age":
                age_value,

            "manufacturer":
                manufacturer,

            "model":
                model_name,

            "country":
                country,

            "maintenance_cost":
                maintenance_cost_value,

            "downtime":
                downtime_value,

            "maintenance_frequency":
                maintenance_frequency_value,

            "failure_event_count":
                failure_event_count_value,

            "maintenance_class":
                maintenance_class_value,

            "operational_hours":
                operational_hours_value,

            "expected_lifespan":
                expected_lifespan_value,

            "mtbf":
                mtbf,

            "cost_per_hour":
                cost_per_hour,

            "lifespan_usage_ratio":
                lifespan_usage_ratio,

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


        # ====================================================
        # SAVE PREDICTION
        # ====================================================

        save_prediction(
            prediction_data
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        return render_template(

            "result.html",

            device=device_type,

            risk=
                risk_info["risk"],

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
                    age_value,

                "Manufacturer":
                    manufacturer,

                "Model":
                    model_name,

                "Country":
                    country,

                "Maintenance Cost":
                    maintenance_cost_value,

                "Downtime":
                    downtime_value,

                "Maintenance Frequency":
                    maintenance_frequency_value,

                "Failure Event Count":
                    failure_event_count_value,

                "Maintenance Class":
                    maintenance_class_value,

                "Operational Hours":
                    operational_hours_value,

                "Expected Lifespan":
                    expected_lifespan_value,

                "MTBF":
                    mtbf,

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

        print()
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

@app.route(
    "/profile",
    methods=["GET", "POST"]
)
def profile():

    if not check_login():

        return redirect(
            url_for("login")
        )


    # ========================================================
    # SAVE PROFILE
    # ========================================================

    if request.method == "POST":

        profile_data = session.get(
            "profile_data",
            {}
        ).copy()


        email = request.form.get(
            "email",
            ""
        ).strip()


        role = request.form.get(
            "role",
            "Hospital Administrator"
        ).strip()


        # Save profile information only

        profile_data["email"] = email

        profile_data["role"] = role


        session["profile_data"] = (
            profile_data
        )

        session.modified = True


        print(
            "Profile updated successfully."
        )


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


    login_username = session.get(
        "login_username",
        "admin"
    )


    return render_template(

        "profile.html",

        username=
            login_username,

        email=
            profile_data.get(
                "email",
                "admin@smarthealthcare.com"
            ),

        role=
            profile_data.get(
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

    # --------------------------------------------------------
    # Initialize database
    # --------------------------------------------------------

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