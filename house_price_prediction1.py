
# ============================================================
# PROJECT: HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ============================================================

# CELL 1: Import Libraries

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")


# ============================================================
# CELL 2: Load Dataset
# ============================================================

file_path = r"C:/Users/ak737/Downloads/Housing-selected-columns.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# CELL 3: Basic Information
# ============================================================

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# CELL 4: Data Cleaning
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove extra spaces from text values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Convert numeric columns to numeric data type
numeric_columns = [
    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "stories"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# Standardize yes/no values in categorical columns
categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning"
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].str.lower()

        df[column] = df[column].replace({
            "yes ": "yes",
            "no ": "no"
        })

# Remove rows where target price is missing
df = df.dropna(subset=["price"])

# Fill missing numeric values with their median
for column in numeric_columns:
    if column in df.columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

# Fill missing categorical values with the most common value
for column in categorical_columns:
    if column in df.columns:
        if df[column].notna().any():
            df[column] = df[column].fillna(
                df[column].mode()[0]
            )

print("\nData cleaning completed.")
print("Dataset shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# CELL 5: Price Distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["price"],
    bins=30,
    kde=True
)

plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.show()


# ============================================================
# CELL 6: Area vs Price
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="area",
    y="price"
)

plt.title("House Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.tight_layout()
plt.show()



# ============================================================
# CELL 10: Encode Categorical Columns
# ============================================================

# Convert yes/no columns into numeric values
# LabelEncoder assigns numbers to the categories

le = LabelEncoder()

df["mainroad"] = le.fit_transform(df["mainroad"])
df["guestroom"] = le.fit_transform(df["guestroom"])
df["basement"] = le.fit_transform(df["basement"])
df["hotwaterheating"] = le.fit_transform(df["hotwaterheating"])
df["airconditioning"] = le.fit_transform(df["airconditioning"])

print("\nDataset after encoding:")
print(df.head())


# ============================================================
# CELL 11: Correlation Heatmap
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# CELL 12: Select Features and Target
# ============================================================

# Features are the columns used to predict the price
# Target is the price column

X = df.drop("price", axis=1)
y = df["price"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: price")


# ============================================================
# CELL 13: Split Dataset
# ============================================================

# 70% data for training and 30% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ============================================================
# CELL 14: Create Machine Learning Models
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR())
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ])
}


# ============================================================
# CELL 15: Train Models and Make Predictions
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():

    print("\nTraining:", model_name)

    # Train the model
    model.fit(X_train, y_train)

    # Predict house prices
    y_pred = model.predict(X_test)

    predictions[model_name] = y_pred

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))


# ============================================================
# CELL 16: Compare Model Performance
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Save the results table
os.makedirs("project_outputs", exist_ok=True)

results_df.to_csv(
    "project_outputs/model_comparison.csv",
    index=False
)


# ============================================================
# CELL 17: Plot RMSE Comparison
# ============================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE"
)

plt.title("RMSE Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ============================================================
# CELL 18: Actual vs Predicted Prices
# ============================================================

# Select the model with the lowest RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

best_predictions = predictions[best_model_name]

print("\nBest model based on RMSE:", best_model_name)

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.7
)

# Reference line for perfect predictions
min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.show()




# ============================================================
# CELL 20: Predict Price of a New House
# ============================================================

# Example values for a new house.
# Keep the feature names and order the same as X.columns.
# For encoded yes/no columns, 0 and 1 represent the
# categories assigned by LabelEncoder.

new_house = pd.DataFrame([{
    "area": 5000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": 1,
    "guestroom": 0,
    "basement": 0,
    "hotwaterheating": 0,
    "airconditioning": 1
}])

# Arrange columns in the same order as training data
new_house = new_house[X.columns]

predicted_price = best_model.predict(new_house)

print("\nNew House Details:")
print(new_house)

print("\nPredicted House Price:")
print(round(predicted_price[0], 2))


# ============================================================
# CELL 21: Save Trained Models
# ============================================================

# Save the trained models
# No encoders dictionary is used in this project

joblib.dump(
    models,
    "project_outputs/house_price_models.pkl"
)

# Save the best model separately
joblib.dump(
    best_model,
    "project_outputs/best_house_price_model.pkl"
)

print("\nModels saved successfully!")

print("\nFiles saved in the project_outputs folder:")
print("1. house_price_models.pkl")
print("2. best_house_price_model.pkl")
print("3. model_comparison.csv")


# ============================================================
# CELL 22: Final Project Summary
# ============================================================

print("\n========== PROJECT SUMMARY ==========")

print("Total rows used:", len(df))
print("Number of input features:", X.shape[1])
print("Number of models trained:", len(models))
print("Best model:", best_model_name)

best_result = results_df.iloc[0]

print("Best model RMSE:", round(best_result["RMSE"], 2))
print("Best model MAE:", round(best_result["MAE"], 2))
print("Best model R2 Score:", round(best_result["R2 Score"], 4))

print("\nHouse Price Prediction project completed!")

