# House Price Prediction using Machine Learning

## Project Overview
This project uses machine learning regression algorithms to predict house prices based on housing features such as area, bedrooms, bathrooms, stories, and available facilities.

## Project Objectives
- Clean and prepare the housing dataset.
- Explore the data using basic visualizations.
- Train and compare different regression models.
- Evaluate models using MAE, MSE, RMSE, and R-squared (R²).
- Predict the price of a sample house.

## Machine Learning Models
1. Linear Regression
2. Decision Tree Regressor
3. Support Vector Regression (SVR)
4. K-Nearest Neighbors (KNN) Regressor

## Visualizations
The project includes these main figures:
- House price distribution
- Area versus house price
- Correlation heatmap
- Actual versus predicted house prices

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Files
- `house_price_prediction.py` — main Python code
- `Housing-selected-columns.csv` — housing dataset
- `requirements.txt` — required Python libraries
- `README.md` — project documentation

## Installation
Install the required libraries with:

```bash
pip install -r requirements.txt
```

## How to Run
1. Place `Housing-selected-columns.csv` in the project folder or update the dataset path in the Python code.
2. Open PowerShell in the project folder.
3. Run:

```bash
python house_price_prediction.py
```

## Model Evaluation
The models are compared using:
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Indicates how well the model explains variation in house prices

The model with the lowest RMSE is selected for the sample prediction.

## Notes
- Update the dataset path if the CSV is stored in a different location.
- The sample house values in the code are examples and can be changed.
- Model results depend on the dataset and train-test split.

## Author
Add your name here.
