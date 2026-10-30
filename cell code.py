# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================
# 2. UPLOAD AND READ DATASET
# ==========================================
uploaded = files.upload()
df = pd.read_csv(next(iter(uploaded.keys())))

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ==========================================
# 3. SELECT FEATURES AND TARGET
# ==========================================
target_column = "THIS YEAR LIVE STORAGE (BCM)"

feature_columns = [
    "LIVE CAP. AT FRL (BCM)",
    "THIS SEASON LEVEL (MTS)",
    "LAST YEAR LIVE STORAGE (BCM)",
    "Normal Storage (BCM)"
]

# Keep the required columns
data = df[feature_columns + [target_column]].copy()

# Convert columns to numeric values
for col in feature_columns + [target_column]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove rows where the target is missing
data = data.dropna(subset=[target_column]).reset_index(drop=True)

X = data[feature_columns]
y = data[target_column]


# ==========================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ==========================================
# 5. DEFINE REGRESSION MODELS
# ==========================================
models = {
    "Linear Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        LinearRegression()
    ),

    "Polynomial Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    ),

    "Ridge Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "KNN Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5)
    )
}


# ==========================================
# 6. TRAIN, PREDICT, AND EVALUATE
# ==========================================
results = []
predictions = pd.DataFrame({
    "Actual_Storage_BCM": y_test.to_numpy()
}, index=y_test.index)

for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Predict test-set values
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    predictions[name + "_Prediction"] = y_pred

    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")


# ==========================================
# 7. CREATE FINAL CSV FILES
# ==========================================
results_df = pd.DataFrame(results)

# Sort models by RMSE (lower is better)
results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

results_df.to_csv("model_comparison.csv", index=False)

predictions = predictions.sort_index()
predictions.to_csv("model_predictions.csv", index_label="Original_Row_Index")


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================
print("\nMODEL COMPARISON")
display(results_df.round(4))

print("\nMODEL PREDICTIONS")
display(predictions.head(10).round(4))


# ==========================================
# 9. DOWNLOAD FINAL CSV FILES
# ==========================================
files.download("model_comparison.csv")
files.download("model_predictions.csv")
