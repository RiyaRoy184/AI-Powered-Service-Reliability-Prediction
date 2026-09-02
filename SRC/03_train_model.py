import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    auc
)

import matplotlib.pyplot as plt


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset/processed_device_level.csv")


# ==========================================================
# 2. REMOVE UNNECESSARY COLUMNS
# ==========================================================

df = df.drop(columns=[
    "Device_ID",
    "Purchase_Date",
    "Maintenance_Report"
])


# ==========================================================
# 3. SEPARATE INPUT AND OUTPUT
# ==========================================================

X = df.drop(columns=[
    "Risk_Class",
    "Risk_Class_Label"
])

y = df["Risk_Class_Label"]


# ==========================================================
# 4. ENCODE CATEGORICAL COLUMNS
# ==========================================================

encoders = {}

categorical_columns = [
    "Device_Type",
    "Manufacturer",
    "Model",
    "Country",
    "Maintenance_Frequency",
    "Maintenance_Class"
]

for column in categorical_columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(X[column])

    encoders[column] = encoder


# ==========================================================
# 5. SPLIT DATASET
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data  :", X_test.shape)


# ==========================================================
# 6. CREATE MACHINE LEARNING MODELS
# ==========================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=8,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=2,
        class_weight="balanced",
        random_state=42
    ),

  "Decision Tree": DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=35,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
),

    "SVM": SVC(
        kernel="rbf",
        C=1,
        gamma="scale",
        probability=True,
        random_state=42
    )
}


# ==========================================================
# 7. VARIABLES FOR MODEL COMPARISON
# ==========================================================

best_model = None
best_accuracy = 0

random_forest_model = models["Random Forest"]
random_forest_accuracy = 0


# ==========================================================
# 8. CREATE RESULTS FOLDER
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================================
# 9. TRAIN AND EVALUATE ALL MODELS
# ==========================================================

for name, model in models.items():

    print("\n===================================")
    print("Algorithm :", name)
    print("===================================")


    # ------------------------------------------------------
    # Train Model
    # ------------------------------------------------------

    model.fit(X_train, y_train)


    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    y_pred = model.predict(X_test)


    # ======================================================
    # ACCURACY
    # ======================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    if name == "Random Forest":
        random_forest_accuracy = accuracy

    print(
        "Accuracy :",
        round(accuracy * 100, 2),
        "%"
    )


    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix\n")

    print(cm)


    # ======================================================
    # CONFUSION MATRIX PLOT
    # ======================================================

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ]
    )

    disp.plot(cmap="Blues")

    plt.title(
        f"{name} Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        f"results/{name}_Confusion_Matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    # ======================================================
    # ROC-AUC CALCULATION
    # ======================================================

    # Convert classes into binary format
    y_test_binary = label_binarize(
        y_test,
        classes=[0, 1, 2]
    )


    # Get prediction probabilities
    y_score = model.predict_proba(
        X_test
    )


    # Calculate weighted multiclass ROC-AUC
    roc_auc = roc_auc_score(
        y_test_binary,
        y_score,
        multi_class="ovr",
        average="weighted"
    )

    print(
        "ROC-AUC :",
        round(roc_auc * 100, 2),
        "%"
    )


    # ======================================================
    # ROC CURVE
    # ======================================================

    plt.figure(figsize=(8, 6))


    class_names = [
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]


    # Plot ROC curve for each class
    for i in range(3):

        fpr, tpr, _ = roc_curve(
            y_test_binary[:, i],
            y_score[:, i]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{class_names[i]} "
                f"(AUC = {class_auc:.2f})"
            )
        )


    # ------------------------------------------------------
    # Random Classifier Reference Line
    # ------------------------------------------------------

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random Classifier"
    )


    # ------------------------------------------------------
    # ROC Graph Labels
    # ------------------------------------------------------

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"{name} ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ------------------------------------------------------
    # Save ROC Curve
    # ------------------------------------------------------

    plt.savefig(
        f"results/{name}_ROC_Curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    # ======================================================
    # CHECK FOR BEST MODEL
    # ======================================================

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model


# ==========================================================
# 10. SAVE RANDOM FOREST MODEL
# ==========================================================

joblib.dump(
    random_forest_model,
    "models/model.pkl"
)

joblib.dump(
    encoders,
    "models/encoders.pkl"
)


# ==========================================================
# 11. FINAL OUTPUT
# ==========================================================

print("\n===================================")

print(
    "Random Forest Model Saved Successfully!"
)

print(
    "Random Forest Accuracy :",
    round(
        random_forest_accuracy * 100,
        2
    ),
    "%"
)

print("===================================")