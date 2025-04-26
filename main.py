import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
import xgboost as xgb
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings("ignore")

# 1. DATA COLLECTION AND CLEANING


# Load the dataset
def load_data(file_path):
    """
    Load dataset from CSV file
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


# Basic data exploration
def explore_data(df):
    """
    Perform basic exploration of the dataset
    """
    print("\nBasic information about the dataset:")
    print(df.info())

    print("\nSummary statistics:")
    print(df.describe())

    print("\nMissing values in each column:")
    print(df.isnull().sum())

    return df


# Data cleaning
def clean_data(df):
    """
    Clean and preprocess the dataset
    """
    print("\nCleaning data...")

    # Create a copy to avoid modifying the original
    clean_df = df.copy()

    # Check for duplicate rows
    duplicates = clean_df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows. Removing them...")
        clean_df = clean_df.drop_duplicates()

    # Handle missing values
    # For numerical columns, impute with median
    numerical_cols = clean_df.select_dtypes(include=["int64", "float64"]).columns
    for col in numerical_cols:
        if clean_df[col].isnull().sum() > 0:
            median_val = clean_df[col].median()
            clean_df[col].fillna(median_val, inplace=True)

    # For categorical columns, impute with mode
    categorical_cols = clean_df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if clean_df[col].isnull().sum() > 0:
            mode_val = clean_df[col].mode()[0]
            clean_df[col].fillna(mode_val, inplace=True)

    # Create a binary target variable for delay prediction (1 if delayed, 0 otherwise)
    if "arr_del15" in clean_df.columns:
        clean_df["is_delayed"] = (clean_df["arr_del15"] > 0).astype(int)
    elif "arr_delay" in clean_df.columns:
        clean_df["is_delayed"] = (clean_df["arr_delay"] > 15).astype(int)

    print(
        f"Data cleaning complete. Dataset now has {clean_df.shape[0]} rows and {clean_df.shape[1]} columns"
    )

    return clean_df


# 2. FEATURE ENGINEERING


def engineer_features(df):
    """
    Create and transform features to improve model performance
    """
    print("\nEngineering features...")

    # Create a copy to avoid modifying the original
    feature_df = df.copy()

    # Create new features that might be predictive

    # Calculate the ratio of delayed flights to total flights for each carrier
    if (
        "carrier" in feature_df.columns
        and "arr_flights" in feature_df.columns
        and "arr_del15" in feature_df.columns
    ):
        carrier_delay_stats = feature_df.groupby("carrier").agg(
            {"arr_flights": "sum", "arr_del15": "sum"}
        )
        carrier_delay_stats["delay_ratio"] = (
            carrier_delay_stats["arr_del15"] / carrier_delay_stats["arr_flights"]
        )
        feature_df = pd.merge(
            feature_df, carrier_delay_stats["delay_ratio"], on="carrier", how="left"
        )

    # Calculate the same for airports
    if (
        "airport" in feature_df.columns
        and "arr_flights" in feature_df.columns
        and "arr_del15" in feature_df.columns
    ):
        airport_delay_stats = feature_df.groupby("airport").agg(
            {"arr_flights": "sum", "arr_del15": "sum"}
        )
        airport_delay_stats["airport_delay_ratio"] = (
            airport_delay_stats["arr_del15"] / airport_delay_stats["arr_flights"]
        )
        feature_df = pd.merge(
            feature_df,
            airport_delay_stats["airport_delay_ratio"],
            on="airport",
            how="left",
        )

    # Create season feature based on month
    if "month" in feature_df.columns:
        feature_df["season"] = pd.cut(
            feature_df["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Fall"],
            include_lowest=True,
        )

    # Calculate the proportion of different types of delays
    delay_types = [
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    ]
    if (
        all(col in feature_df.columns for col in delay_types)
        and "arr_delay" in feature_df.columns
    ):
        for delay_type in delay_types:
            feature_df[f"{delay_type}_ratio"] = feature_df[delay_type] / feature_df[
                "arr_delay"
            ].replace(0, 1)

    # Categorize flights by their total delay time
    if "arr_delay" in feature_df.columns:
        feature_df["delay_category"] = pd.cut(
            feature_df["arr_delay"],
            bins=[-1, 0, 15, 30, 60, float("inf")],
            labels=["No Delay", "Minor", "Moderate", "Significant", "Severe"],
        )

    # One-hot encode categorical variables
    categorical_cols = ["carrier", "airport", "season", "delay_category"]
    categorical_cols = [col for col in categorical_cols if col in feature_df.columns]

    if categorical_cols:
        # Keep the original DataFrame before one-hot encoding for reference
        feature_df_before_ohe = feature_df.copy()

        # One-hot encode the categorical columns
        for col in categorical_cols:
            one_hot = pd.get_dummies(feature_df[col], prefix=col, drop_first=True)
            feature_df = pd.concat([feature_df, one_hot], axis=1)
            feature_df.drop(col, axis=1, inplace=True)

    print(
        f"Feature engineering complete. Dataset now has {feature_df.shape[0]} rows and {feature_df.shape[1]} columns"
    )

    return feature_df


# 3. MODEL TRAINING AND EVALUATION


def prepare_data_for_modeling(df, target_col="is_delayed"):
    """
    Prepare data for model training
    """
    print("\nPreparing data for modeling...")

    # Create a copy to avoid modifying the original
    model_df = df.copy()

    # Drop non-numerical columns that might cause issues
    non_numerical_cols = ["carrier_name", "airport_name"]
    for col in non_numerical_cols:
        if col in model_df.columns:
            model_df.drop(col, axis=1, inplace=True)

    # Ensure the target column exists
    if target_col not in model_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset")

    # Separate features and target
    X = model_df.drop(target_col, axis=1)
    y = model_df[target_col]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate different models
    """
    print("\nTraining and evaluating models...")

    # Initialize models with default parameters
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42),
    }

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Store results
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "auc": auc,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        # Save each model
        save_model(model, name=f"flight_delay_{name.lower().replace(' ', '_')}_model")

    return results


def visualize_all_results(y_test, results):
    """
    Visualize results for all models
    """
    plt.figure(figsize=(12, 6))

    # Plot ROC curve for each model
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_pred_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot confusion matrices for all models
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for (name, result), ax in zip(results.items(), axes):
        cm = result["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} Confusion Matrix")

    plt.tight_layout()
    plt.show()

    # Create metrics comparison table
    metrics_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Accuracy": [result["accuracy"] for result in results.values()],
            "AUC": [result["auc"] for result in results.values()],
            "Precision (Class 1)": [
                result["classification_report"]["1"]["precision"]
                for result in results.values()
            ],
            "Recall (Class 1)": [
                result["classification_report"]["1"]["recall"]
                for result in results.values()
            ],
            "F1-Score (Class 1)": [
                result["classification_report"]["1"]["f1-score"]
                for result in results.values()
            ],
        }
    ).set_index("Model")

    print("\nModel Performance Comparison:")
    print(metrics_df)


# 4. DEPLOYMENT PREPARATION


def save_model(model, name="flight_delay_model"):
    """
    Save the trained model to a file
    """
    print(f"\nSaving model to {name}.joblib...")
    joblib.dump(model, f"{name}.joblib")
    print(f"Model saved successfully!")


def predict_delay(model, input_data):
    """
    Use the trained model to predict flight delays for new data
    """
    # Make predictions
    delay_probability = model.predict_proba(input_data)[:, 1]
    is_delayed = model.predict(input_data)

    return is_delayed, delay_probability


# 5. REGRESSION TASK: PREDICT DELAY DURATION


def train_regression_model(df):
    """
    Train a regression model to predict delay duration
    """
    print("\nTraining regression model for delay duration prediction...")

    # Create a copy to avoid modifying the original
    reg_df = df.copy()

    # We'll use 'arr_delay' as our target for regression
    if "arr_delay" not in reg_df.columns:
        print("Column 'arr_delay' not found. Cannot train regression model.")
        return None

    # Prepare data for regression
    X = reg_df.drop(["arr_delay", "is_delayed"], axis=1, errors="ignore")
    y = reg_df["arr_delay"]

    # Remove non-numerical columns
    non_numerical_cols = ["carrier_name", "airport_name"]
    for col in non_numerical_cols:
        if col in X.columns:
            X.drop(col, axis=1, inplace=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost Regressor
    reg_model = xgb.XGBRegressor(random_state=42)
    reg_model.fit(X_train, y_train)

    # Make predictions
    y_pred = reg_model.predict(X_test)

    # Evaluate model
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = reg_model.score(X_test, y_test)

    print(f"Regression Model - RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([0, max(y_test)], [0, max(y_test)], "r--")
    plt.xlabel("Actual Delay (minutes)")
    plt.ylabel("Predicted Delay (minutes)")
    plt.title("Actual vs Predicted Delay Duration")
    plt.grid(True)
    plt.show()

    # Save regression model
    save_model(reg_model, name="flight_delay_duration_regression_model")

    return reg_model


# 6. MAIN FUNCTION TO RUN THE ENTIRE PIPELINE


def flight_delay_prediction_pipeline(file_path):
    """
    Run the entire flight delay prediction pipeline
    """
    # Set up proper plotting
    plt.style.use("ggplot")

    # 1. Load and explore data
    df = load_data(file_path)
    explore_data(df)

    # 2. Clean data
    clean_df = clean_data(df)

    # 3. Engineer features
    feature_df = engineer_features(clean_df)

    # 4. Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(feature_df)

    # 5. Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # 6. Visualize results for all models
    visualize_all_results(y_test, results)

    # 7. Train regression model for delay duration
    reg_model = train_regression_model(feature_df)

    # Return all important components
    return df, clean_df, feature_df, results, reg_model


# To run the pipeline:
# flight_delay_prediction_pipeline('your_flight_data.csv')

# Demo usage
if __name__ == "__main__":
    # Add your file path here
    file_path = "Airline_Delay_Cause.csv"
    df, clean_df, feature_df, results, reg_model = flight_delay_prediction_pipeline(
        file_path
    )

    # Demo how to use the model for prediction
    print("\nDEMO: Using the model for prediction")

    # Take a sample from the test data - ensure we drop the same columns as during training
    sample_data = feature_df.drop(
        ["is_delayed", "carrier_name", "airport_name"], axis=1, errors="ignore"
    ).sample(5, random_state=42)

    # Make predictions with each model
    for model_name, result in results.items():
        print(f"\nPredictions using {model_name}:")

        try:
            is_delayed, delay_prob = predict_delay(result["model"], sample_data)

            # Display results
            for i, (idx, row) in enumerate(sample_data.iterrows()):
                print(f"Flight {i+1}:")
                if "carrier" in df.columns:
                    carrier = df.loc[idx, "carrier"] if idx in df.index else "Unknown"
                    print(f"  Carrier: {carrier}")
                if "airport" in df.columns:
                    airport = df.loc[idx, "airport"] if idx in df.index else "Unknown"
                    print(f"  Airport: {airport}")
                print(f"  Delay Probability: {delay_prob[i]:.2f}")
                print(f"  Predicted: {'Delayed' if is_delayed[i] else 'On Time'}")
                print()
        except Exception as e:
            print(f"Error making predictions with {model_name}: {str(e)}")
