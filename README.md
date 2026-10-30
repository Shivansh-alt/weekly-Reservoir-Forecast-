# Weekly-Reservoir-Forecast
# Reservoir Water Storage Prediction Using Machine Learning

## 1. Project Overview

This project uses machine learning regression algorithms to predict reservoir water storage based on reservoir-related data, including live storage capacity, current water level, previous year's storage, and normal storage.

The project implements and compares four regression algorithms:

- Linear Regression
- Polynomial Regression
- Ridge Regression
- K-Nearest Neighbors (KNN) Regression

The models are evaluated using performance metrics to identify which algorithm provides the most accurate predictions on the test dataset.

## 2. Objectives

The main objectives of this project are:

- To analyze reservoir water storage data.
- To preprocess the dataset for machine learning.
- To train different regression models.
- To predict reservoir water storage in billion cubic metres (BCM).
- To compare the performance of different regression algorithms.
- To identify the best-performing model based on evaluation metrics.

## 3. Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Scikit-learn** — Machine learning algorithms and evaluation metrics
- **Google Colab** — Cloud-based Python development environment
- **CSV** — Dataset and output file format

## 4. Dataset Description

The project uses a reservoir dataset stored in `Dataset.csv`.

The dataset contains information about reservoirs, including their locations, capacities, water levels, and storage measurements.

### Important Features

| Feature | Description |
|---|---|
| `LIVE CAP. AT FRL (BCM)` | Live storage capacity at Full Reservoir Level |
| `THIS SEASON LEVEL (MTS)` | Water level recorded for the current season |
| `LAST YEAR LIVE STORAGE (BCM)` | Live storage recorded during the previous year |
| `Normal Storage (BCM)` | Normal storage value for the reservoir |

### Target Variable

`THIS YEAR LIVE STORAGE (BCM)`

This is the variable the machine learning models attempt to predict.

**Note:** The dataset contains only 24 reservoir records, so the model results should be interpreted as preliminary.

## 5. Machine Learning Algorithms

### 5.1 Linear Regression

Linear Regression models the relationship between the input features and the target variable using a linear equation.

It serves as the baseline model for predicting reservoir storage.

### 5.2 Polynomial Regression

Polynomial Regression extends linear regression by introducing polynomial combinations of the input features. This allows the model to capture certain nonlinear relationships.

This project uses polynomial features of degree 2.

### 5.3 Ridge Regression

Ridge Regression is a regularized version of linear regression. It adds an L2 penalty to the model's coefficients to help control model complexity and reduce overfitting.

### 5.4 K-Nearest Neighbors Regression

KNN Regression predicts a value using the target values of nearby training observations.

This project uses five neighbors (`n_neighbors=5`) and standardizes the input features before training the model.

## 6. Project Workflow

The project follows these steps:

1. Load the reservoir dataset.
2. Select the input features and target variable.
3. Convert the selected columns to numeric values.
4. Handle missing values in the selected features.
5. Divide the data into training and testing sets.
6. Train the four regression models.
7. Generate predictions for the test data.
8. Evaluate the models using RMSE, MAE, and R².
9. Compare model performance.
10. Export model comparison results and predictions to CSV files.

### Train-Test Split

The dataset is divided using the following settings:

- Training data: 80%
- Testing data: 20%
- Random state: 42

Because the dataset is small, the test set contains only five observations.

## 7. Model Evaluation Metrics

The models are evaluated using three metrics.

| Metric | Meaning | Preferred Result |
|---|---|---|
| RMSE | Root Mean Squared Error | Lower |
| MAE | Mean Absolute Error | Lower |
| R² Score | Coefficient of Determination | Higher |

RMSE and MAE measure prediction errors, while R² indicates how well the model explains variation in the target values.

## 8. Results

The following results were obtained using the selected features and the specified train-test split.

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.0702 | 0.0568 | 0.9608 |
| Ridge Regression | 0.0836 | 0.0727 | 0.9444 |
| KNN Regression | 0.2840 | 0.2259 | 0.3579 |
| Polynomial Regression | 0.6912 | 0.4712 | -2.8043 |

### Best-Performing Model

Linear Regression achieved the lowest RMSE and MAE and the highest R² score among the four models on this test split.

Ridge Regression also performed well, with results close to those of Linear Regression.

KNN Regression produced larger prediction errors, while Polynomial Regression performed poorly on the test data.

These results do not guarantee that Linear Regression will perform best on new data. More observations and additional validation are needed to confirm the model's generalization performance.

## 9. Output Files

The project can generate the following CSV files:

### `model_comparison.csv`

Contains the evaluation results for each regression model, including RMSE, MAE, and R².

### `model_predictions.csv`

Contains the actual reservoir storage values and the predictions produced by each regression model.

The prediction file includes columns such as:

- `Original_Row_Index`
- `Actual_Storage_BCM`
- `Linear Regression_Prediction`
- `Polynomial Regression_Prediction`
- `Ridge Regression_Prediction`
- `KNN Regression_Prediction`

These files make it easier to analyze the model predictions and compare their accuracy.

## 10. How to Run the Project

### Step 1: Open Google Colab

Open Google Colab and create a new Python notebook.

### Step 2: Upload the Dataset

Upload `Dataset.csv` to your notebook, or use the notebook's file-upload functionality to select it from your computer.

### Step 3: Install Dependencies

The required libraries are generally available in Google Colab. If necessary, install them with:

```bash
pip install pandas numpy scikit-learn
```

### Step 4: Run the Notebook

Execute the notebook cells in order to load and preprocess the data, train the models, and evaluate their performance.

### Step 5: Generate Output Files

Run the cells that save the model comparison and prediction results to CSV files.

### Step 6: Download the Results

Download `model_comparison.csv` and `model_predictions.csv` from Google Colab for further analysis or inclusion in your project submission.

## 11. Limitations

- The dataset contains only 24 records.
- The test set contains only five observations, making performance estimates sensitive to individual predictions.
- The reported metrics depend on the selected train-test split.
- Polynomial Regression may produce unrealistic negative storage predictions because it does not inherently enforce physical constraints.
- The current implementation predicts storage from selected reservoir features; it does not perform true time-series forecasting.
- The model's performance on new reservoirs or future observations has not been established.

## 12. Future Enhancements

Possible improvements include:

- Collecting more reservoir records from different seasons and years.
- Using cross-validation to obtain more reliable performance estimates.
- Testing additional regression algorithms, such as Random Forest Regression.
- Tuning model hyperparameters.
- Visualizing actual versus predicted storage values.
- Incorporating rainfall, inflow, outflow, and other hydrological variables when available.
- Developing a time-series forecasting model if sufficient dated observations are available for each reservoir.
- Investigating methods that prevent physically invalid negative storage predictions.

## 13. Conclusion

This project demonstrates the application of supervised machine learning regression techniques to reservoir water storage prediction.

Four algorithms—Linear Regression, Polynomial Regression, Ridge Regression, and KNN Regression—were trained and compared using RMSE, MAE, and R².

On the selected test split, Linear Regression achieved the best evaluation results, followed by Ridge Regression.

The project provides a starting point for applying machine learning to reservoir data analysis. However, additional data and validation are required before using the predictions for real-world reservoir management or operational decisions.

---

**Project Type:** Machine Learning / Supervised Learning / Regression

**Programming Language:** Python

**Environment:** Google Colab
