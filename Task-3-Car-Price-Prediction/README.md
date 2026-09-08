#  Car Price Prediction with Machine Learning

A complete, beginner/intermediate machine learning project that predicts used car selling prices from vehicle characteristics using regression models. Built as a portfolio project demonstrating an end-to-end data science workflow — from raw data to a deployable model.

##  Overview

This project explores, cleans, and models a real-world used-car dataset to predict `Selling_Price` based on features such as manufacturing year, present (showroom) price, kilometers driven, fuel type, seller type, transmission, and ownership history.

A key part of this project was **critical data inspection**: the raw dataset turned out to mix car and motorcycle listings under a single `Car_Name` column. This was detected during EDA (confirmed via the clean separation in `Present_Price` scale between the two groups) and the dataset was scoped to cars only (198 records after cleaning) before modeling.

##  Dataset

- **Source:** `car_data.csv` (301 rows, 9 columns — 200 cars + 101 two-wheelers before filtering)
- **Target variable:** `Selling_Price` (in INR Lakhs)
- **Original features:** `Car_Name`, `Year`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`
- **Engineered features:** `Brand` (derived from model name), `Vehicle_Age` (derived from `Year`)

##  Workflow

1. **Data Inspection** — shape, dtypes, missing values, duplicates, descriptive statistics
2. **Data Cleaning** — duplicate removal, outlier investigation (no blind deletion), vehicle-type scoping
3. **Exploratory Data Analysis** — price distributions, brand comparisons, correlation analysis
4. **Feature Engineering** — `Brand` extraction, `Vehicle_Age` calculation, categorical encoding
5. **Preprocessing** — `ColumnTransformer` + `Pipeline` (scaling + one-hot encoding), leakage-safe train/test split
6. **Modeling** — Linear Regression, Random Forest, Gradient Boosting
7. **Evaluation** — MAE, MSE, RMSE, R² comparison across models
8. **Interpretation** — feature importance analysis and real-world application discussion

##  Results

| Model | MAE (Lakh) | RMSE (Lakh) | R² |
|---|---|---|---|
| **Random Forest** ⭐ | **1.05** | **2.51** | **0.81** |
| Linear Regression | 1.45 | 2.79 | 0.77 |
| Gradient Boosting | 1.12 | 2.80 | 0.77 |

**Best model:** Random Forest Regressor, selected by RMSE and cross-checked against MAE and R².

### Key Insights
- `Present_Price` (showroom price of the new model) is the single strongest predictor of resale value.
- Vehicle age and mileage both reduce predicted price, consistent with normal depreciation.
- Toyota commands the highest average resale price in the dataset; Maruti the lowest, reflecting brand/segment positioning.
- Diesel and automatic-transmission cars sell for a higher median price than petrol/manual cars.
- Tree-based ensembles outperform Linear Regression, indicating non-linear relationships in the pricing data.

## 🛠️ Tech Stack

- **Python 3** — pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Machine Learning** — scikit-learn (`Pipeline`, `ColumnTransformer`, `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`)

##  Repository Structure

```
├── car_data.csv                    # Raw dataset
├── Car_Price_Prediction.ipynb      # Full analysis notebook
└── README.md                       # Project documentation
```

##  How to Run

```bash
git clone <your-repo-url>
cd car-price-prediction
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Car_Price_Prediction.ipynb
```

##  Limitations

- Small dataset (198 cars) limits generalization
- No condition, accident-history, or maintenance-history data
- No regional/location information
- Limited brand coverage (Maruti, Toyota, Hyundai, Honda only)
- Historical snapshot — would need periodic retraining for live market use


---

*This project was built as part of a Data Science internship portfolio.*
