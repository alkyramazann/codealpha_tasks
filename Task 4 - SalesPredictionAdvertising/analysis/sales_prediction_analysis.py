"""
==============================================================
SALES PREDICTION USING PYTHON
Dataset: Advertising.csv (TV, Radio, Newspaper spend vs Sales)
==============================================================
This script performs a complete, adaptive analysis based on the
ACTUAL structure of the attached dataset. No columns, dates,
platforms, or segments are invented.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
CHART_DIR = "../charts/"
RANDOM_STATE = 42

# ==============================================================
# 2. LOAD DATASET
# ==============================================================
df = pd.read_csv("../data/Advertising.csv")

print("=" * 70)
print("STEP 3: DATASET INSPECTION")
print("=" * 70)

# The first column is an unnamed row index exported from the source file
if df.columns[0].strip() == "" or "Unnamed" in df.columns[0]:
    df = df.drop(columns=[df.columns[0]])

print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print("\nFirst 10 rows:")
print(df.head(10).to_string())
print("\nData types:")
print(df.dtypes)
print("\nDescriptive statistics:")
print(df.describe().to_string())
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# --------------------------------------------------------------
# Dataset understanding note
# --------------------------------------------------------------
"""
WHAT THE DATASET REPRESENTS
----------------------------
This is the classic "Advertising" dataset: 200 observations (e.g. markets
or campaigns) recording advertising spend (in thousands of dollars) across
three media channels -- TV, Radio, Newspaper -- and the resulting Sales
(in thousands of units).

Columns actually present: TV, Radio, Newspaper, Sales.
NOT present: any date/time column, any explicit "target segment" column,
and any explicit "platform" column in the sense of an e-commerce/ad
platform label. However, TV / Radio / Newspaper themselves function as
the three advertising CHANNELS in this dataset, so the "platform
comparison" requested in the brief is answered by comparing these three
spend columns' relationship with Sales.

BUSINESS QUESTIONS THIS DATASET CAN ANSWER
- How does advertising spend on each channel relate to sales?
- Which channel is the most efficient use of an advertising budget?
- Can sales be predicted from a given media spend allocation?
- What advertising mix maximizes sales for a fixed budget?
"""

# ==============================================================
# 4. DATA CLEANING
# ==============================================================
print("\n" + "=" * 70)
print("STEP 4: DATA CLEANING")
print("=" * 70)

# Missing values
missing_total = df.isnull().sum().sum()
print(f"\nTotal missing values found: {missing_total}")
if missing_total > 0:
    df = df.dropna()
    print("Rows with missing values dropped (dataset is small and dense; "
          "no reliable pattern to impute from).")
else:
    print("No missing values present -> no imputation needed.")

# Duplicates
dupes = df.duplicated().sum()
print(f"\nDuplicate rows found: {dupes}")
if dupes > 0:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows (identical spend/sales "
          "rows are not plausible as distinct real observations).")
else:
    print("No duplicate rows -> nothing removed.")

# Data types
for col in ["TV", "Radio", "Newspaper", "Sales"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("\nConfirmed TV, Radio, Newspaper, Sales are numeric:")
print(df.dtypes)

# Invalid values
print("\nChecking for negative/impossible values:")
for col in df.columns:
    n_neg = (df[col] < 0).sum()
    print(f"  {col}: {n_neg} negative values")

# Outlier check (IQR method, informational only -- not auto-removed)
print("\nOutlier check (IQR method):")
for col in df.columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"  {col}: {len(outliers)} potential outliers (kept -- plausible "
          f"high-spend/high-sales campaigns, not data errors)")

# No date column -> no datetime conversion possible/needed
has_date_column = False
print("\nNo date/time column exists in this dataset. Time-series feature "
      "engineering and trend/seasonality analysis are therefore skipped, "
      "as instructed when sufficient temporal information is absent.")

print(f"\nFinal cleaned shape: {df.shape}")

# ==============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ==============================================================
print("\n" + "=" * 70)
print("STEP 5: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

corr = df.corr()
print("\nCorrelation matrix:")
print(corr.round(3).to_string())

print("\nCorrelation of each channel with Sales:")
print(corr["Sales"].drop("Sales").sort_values(ascending=False).round(3).to_string())

# No target-segment / platform label columns exist -> those specific
# breakdowns (Section 4 sub-bullets) cannot be produced; channel-level
# comparison (TV vs Radio vs Newspaper) is used instead, see Step 12/13.

# ==============================================================
# 6. FEATURE ENGINEERING
# ==============================================================
print("\n" + "=" * 70)
print("STEP 6: FEATURE ENGINEERING")
print("=" * 70)

df["Total_Ad_Spend"] = df["TV"] + df["Radio"] + df["Newspaper"]
df["TV_Share"] = df["TV"] / df["Total_Ad_Spend"]
df["Radio_Share"] = df["Radio"] / df["Total_Ad_Spend"]
df["Newspaper_Share"] = df["Newspaper"] / df["Total_Ad_Spend"]

print("Created features: Total_Ad_Spend, TV_Share, Radio_Share, "
      "Newspaper_Share.")
print("No categorical Target Segment / Platform / Product / Region column "
      "exists in this dataset, so one-hot encoding is not applicable here.")
print(df[["Total_Ad_Spend", "TV_Share", "Radio_Share", "Newspaper_Share"]].head())

# ==============================================================
# 7. FEATURE SELECTION
# ==============================================================
print("\n" + "=" * 70)
print("STEP 7: FEATURE SELECTION")
print("=" * 70)

# Total_Ad_Spend is a linear combination of TV+Radio+Newspaper (perfect
# multicollinearity), and the *_Share features are derived ratios of the
# same three numbers. Including all of them alongside the base channels
# would add redundant, collinear information without new predictive
# signal for a small (200-row) dataset. The three raw channel-spend
# columns already fully capture the same information and are also the
# most business-interpretable (a manager can act on "$ spent per
# channel", not on a share ratio). No ID-like columns exist.
selected_features = ["TV", "Radio", "Newspaper"]
target = "Sales"
print(f"Selected features: {selected_features}")
print("Reason: they are the only genuinely independent numeric predictors; "
      "Total_Ad_Spend and the *_Share features are mathematically derived "
      "from them (multicollinear) and were excluded from the model inputs, "
      "though Total_Ad_Spend is used later for the efficiency analysis.")

# ==============================================================
# 8. CHOOSE PREDICTION APPROACH
# ==============================================================
print("\n" + "=" * 70)
print("STEP 8: PREDICTION APPROACH")
print("=" * 70)
print("No date/time column exists and there is no indication these 200 rows "
      "are sequential observations of the same market over time. "
      "Time-series forecasting is therefore not appropriate here. "
      "-> Approach selected: REGRESSION (predicting Sales from advertising "
      "spend across TV, Radio, Newspaper).")

# ==============================================================
# 9. TRAIN/TEST SPLIT
# ==============================================================
X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)} "
      f"(80/20 split, random_state={RANDOM_STATE})")

# ==============================================================
# 10-11. MODEL TRAINING & EVALUATION
# ==============================================================
print("\n" + "=" * 70)
print("STEP 10-11: MODEL TRAINING & EVALUATION")
print("=" * 70)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

results_df = pd.DataFrame(results).T.round(3)
print("\nModel Comparison Table:")
print(results_df.to_string())

best_model_name = results_df["R2"].idxmax()
best_model = models[best_model_name]
best_preds = predictions[best_model_name]
print(f"\nBest model based on R2: {best_model_name}")

# ==============================================================
# 12. ACTUAL VS PREDICTED
# ==============================================================
print("\n" + "=" * 70)
print("STEP 12: ACTUAL VS PREDICTED SALES")
print("=" * 70)

comparison = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": best_preds.round(2),
})
comparison["Error"] = (comparison["Actual_Sales"] - comparison["Predicted_Sales"]).round(2)
comparison = comparison.reset_index(drop=True)
print("\nSample predictions (first 10 test rows):")
print(comparison.head(10).to_string())
comparison.to_csv("../reports/actual_vs_predicted.csv", index=False)

# ==============================================================
# 13. ADVERTISING IMPACT ANALYSIS
# ==============================================================
print("\n" + "=" * 70)
print("STEP 13: ADVERTISING IMPACT ANALYSIS")
print("=" * 70)

channel_corr = corr["Sales"].drop("Sales")
print("\nCorrelation with Sales by channel:")
print(channel_corr.round(3).to_string())

# Simple efficiency metric: sales generated per $1,000 spent per channel,
# estimated via each channel's simple linear regression slope (sales
# units per unit of spend), since raw spend levels differ hugely across
# channels.
efficiency = {}
for ch in ["TV", "Radio", "Newspaper"]:
    slope = np.polyfit(df[ch], df["Sales"], 1)[0]
    efficiency[ch] = slope
efficiency_series = pd.Series(efficiency).sort_values(ascending=False)
print("\nEstimated Sales lift per additional $1,000 spend, by channel "
      "(simple linear slope, all else ignored):")
print(efficiency_series.round(4).to_string())
print(f"\nMost efficient channel by this measure: {efficiency_series.idxmax()}")
print("\nNote: correlation/slope does not prove causation -- this dataset "
      "does not include experimental (randomized) spend assignment, so "
      "these results describe association, not a guaranteed causal effect.")

# No Target Segment or Platform label column exists, so a breakdown of
# the relationship "by segment/platform" cannot be produced beyond the
# channel-level comparison above.

# ==============================================================
# 14. FEATURE IMPORTANCE (tree-based model)
# ==============================================================
print("\n" + "=" * 70)
print("STEP 14: FEATURE IMPORTANCE (Random Forest)")
print("=" * 70)

rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=selected_features)
importances = importances.sort_values(ascending=False)
print(importances.round(4).to_string())

# ==============================================================
# 15. VISUALIZATIONS
# ==============================================================
print("\n" + "=" * 70)
print("STEP 15: VISUALIZATIONS")
print("=" * 70)

# Chart 1: Advertising Spend vs Sales (per channel scatter, TV as primary)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, ch in zip(axes, ["TV", "Radio", "Newspaper"]):
    ax.scatter(df[ch], df["Sales"], alpha=0.6, color="#2E6F95", edgecolor="white")
    ax.set_title(f"{ch} Spend vs Sales")
    ax.set_xlabel(f"{ch} Advertising Spend ($ thousands)")
    ax.set_ylabel("Sales (thousands of units)")
fig.suptitle("Chart 1: Advertising Spend vs Sales by Channel", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CHART_DIR + "chart1_spend_vs_sales.png", bbox_inches="tight")
plt.close()

# Chart 2: "Platform" (channel) comparison -- total spend and avg sales lift
fig, ax = plt.subplots(figsize=(7, 5))
totals = df[["TV", "Radio", "Newspaper"]].sum()
ax.bar(totals.index, totals.values, color=["#2E6F95", "#E07A5F", "#81B29A"])
ax.set_title("Chart 2: Total Advertising Spend by Channel (200 campaigns)", fontweight="bold")
ax.set_xlabel("Advertising Channel")
ax.set_ylabel("Total Spend ($ thousands)")
plt.tight_layout()
plt.savefig(CHART_DIR + "chart2_spend_by_channel.png", bbox_inches="tight")
plt.close()

# Chart 3: Correlation heatmap (used in place of target-segment chart,
# since no segment column exists in this dataset)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Chart 3: Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig(CHART_DIR + "chart3_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# Chart 4: Sales distribution (used in place of a time-trend chart,
# since no date column exists in this dataset)
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(df["Sales"], bins=20, kde=True, color="#3D5A80", ax=ax)
ax.set_title("Chart 4: Distribution of Sales", fontweight="bold")
ax.set_xlabel("Sales (thousands of units)")
ax.set_ylabel("Number of Campaigns")
plt.tight_layout()
plt.savefig(CHART_DIR + "chart4_sales_distribution.png", bbox_inches="tight")
plt.close()

# Chart 5: Actual vs Predicted Sales
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(comparison["Actual_Sales"], comparison["Predicted_Sales"],
           alpha=0.7, color="#2E6F95", edgecolor="white")
lims = [min(comparison["Actual_Sales"].min(), comparison["Predicted_Sales"].min()) - 1,
        max(comparison["Actual_Sales"].max(), comparison["Predicted_Sales"].max()) + 1]
ax.plot(lims, lims, "--", color="gray", label="Perfect Prediction")
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title(f"Chart 5: Actual vs Predicted Sales ({best_model_name})", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(CHART_DIR + "chart5_actual_vs_predicted.png", bbox_inches="tight")
plt.close()

# Chart 6: Feature importance
fig, ax = plt.subplots(figsize=(6.5, 4))
importances.plot(kind="barh", color="#81B29A", ax=ax)
ax.set_title("Chart 6: Feature Importance (Random Forest)", fontweight="bold")
ax.set_xlabel("Importance")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(CHART_DIR + "chart6_feature_importance.png", bbox_inches="tight")
plt.close()

print("Saved 6 charts to the charts/ folder.")

# ==============================================================
# Save results table for the report
# ==============================================================
results_df.to_csv("../reports/model_comparison.csv")
print("\nModel comparison table saved to reports/model_comparison.csv")
print("\nDONE.")
