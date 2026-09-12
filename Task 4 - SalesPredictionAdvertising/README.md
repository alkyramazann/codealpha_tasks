# Sales Prediction using Python — Advertising Dataset

A Data Science / Marketing Analytics project predicting sales from advertising
spend, built for an internship portfolio / GitHub submission.

## 1. Dataset

**Source file:** `data/Advertising.csv` (the only data source used — nothing invented or substituted)

| Property | Value |
|---|---|
| Rows | 200 |
| Columns | TV, Radio, Newspaper, Sales |
| Missing values | 0 |
| Duplicate rows | 0 |
| Date/time column | None present |
| Target segment column | None present |
| Platform/channel column | None present (TV/Radio/Newspaper themselves are the channels) |

Each row represents one advertising campaign/market: how much was spent
(in $ thousands) on TV, Radio, and Newspaper advertising, and the resulting
Sales (in thousands of units).

Because there is **no date column** and **no segment/platform label column**,
every part of this project that depends on those (time-series forecasting,
segment breakdowns) has been explicitly skipped rather than invented — as
instructed. The three spend columns (TV, Radio, Newspaper) are treated as
the "channels" for the advertising-effectiveness comparison the brief asks for.

## 2. Approach Selected: Regression

With no chronological structure in the data, **regression** — not time-series
forecasting — is the appropriate approach: Sales is predicted from TV, Radio,
and Newspaper spend.

- **Train/test split:** 80% / 20% (160 / 40 rows), `random_state=42`
- **Features used:** TV, Radio, Newspaper (raw spend). `Total_Ad_Spend` and
  per-channel *share* features were engineered but excluded from the model
  inputs since they are mathematically derived from the three spend columns
  (perfect multicollinearity) — see the script for the full reasoning.

## 3. Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.461 | 1.782 | 0.899 |
| Random Forest | 0.613 | 0.740 | **0.983** |
| Gradient Boosting | 0.619 | 0.730 | 0.983 |

**Best model: Random Forest Regressor** (R² = 0.983, meaning the model
explains ~98.3% of the variance in Sales on unseen test data; on average
predictions are off by about 0.61 thousand units of sales — MAE).

The large gap between Linear Regression and the tree-based models indicates
the true relationship between advertising spend and sales is **non-linear**
(likely diminishing returns at higher spend levels, especially for TV).

## 4. Advertising Impact Analysis

Correlation with Sales:

| Channel | Correlation with Sales | Random Forest Feature Importance |
|---|---:|---:|
| TV | 0.782 | 0.625 |
| Radio | 0.576 | 0.362 |
| Newspaper | 0.228 | 0.013 |

Estimated sales lift per additional $1,000 spent (simple linear slope, all
else held aside): **Radio 0.20**, Newspaper 0.055, TV 0.048 units of sales
per $1,000.

- **TV spend has the strongest overall correlation with Sales** and is by far
  the most important predictive feature in the model — it drives the bulk of
  sales variation across campaigns.
- **Radio shows the steepest per-dollar marginal return** in this simple
  single-variable estimate, suggesting it may be an efficient channel at the
  margin, though it explains less of total sales variance than TV overall.
- **Newspaper has the weakest relationship with Sales** by every measure
  (lowest correlation, lowest feature importance, lowest model reliance).
- **Correlation is not causation:** this dataset records observed spend and
  outcomes, not a randomized experiment, so these results describe
  association, not a guaranteed causal effect of spend on sales.

## 5. Business Insights

1. TV advertising spend has the strongest relationship with sales (r = 0.78) and is the single most important driver in the predictive model (62.5% of Random Forest feature importance).
2. Radio advertising also correlates meaningfully with sales (r = 0.58) and shows the highest estimated sales-per-dollar slope of the three channels.
3. Newspaper advertising has a weak relationship with sales (r = 0.23) and contributes almost nothing to the model (1.3% importance) — it is the least influential channel in this dataset.
4. The relationship between spend and sales is non-linear: tree-based models (Random Forest, Gradient Boosting) achieve R² ≈ 0.98 versus 0.90 for plain Linear Regression, indicating diminishing or variable returns rather than a straight-line effect.
5. The best model (Random Forest) predicts sales with an average error of about 0.61 thousand units on data it has never seen, which is small relative to the sales range in the dataset (1.6 to 27.0 thousand units).
6. TV and Radio spend are only weakly correlated with each other (r = 0.055) and with Newspaper (r ≈ 0.06–0.35), meaning each channel's spend varies fairly independently across campaigns in this dataset — the model can attribute effects to each channel somewhat separately.
7. No date or segment/platform information exists in this dataset, so time-trend and segment-specific conclusions cannot be drawn; all findings here are cross-campaign associations, not longitudinal trends.

## 6. Actionable Marketing Recommendations

*What the data shows* vs. *what a business could consider doing*:

1. **Data shows:** TV spend is the strongest and most reliable predictor of sales. → **Consider:** when allocating a fixed budget, weighting it toward TV is well-supported by this dataset's patterns, though not guaranteed to be causal.
2. **Data shows:** Radio has a comparatively strong per-dollar association with sales despite typically lower total spend. → **Consider:** testing incremental Radio budget increases (e.g., a small controlled experiment) to see if the association holds when spend is deliberately varied.
3. **Data shows:** Newspaper spend has the weakest relationship with sales in this dataset. → **Consider:** reviewing whether continued Newspaper investment is justified, or trialing a reduced-Newspaper / reallocated-budget scenario.
4. **Data shows:** the spend–sales relationship is non-linear (returns likely diminish at higher spend). → **Consider:** avoiding purely linear extrapolation when planning large budget increases; use the trained non-linear model (or a similar one) to simulate different allocations before committing budget.
5. **Data shows:** the model achieves strong but not perfect accuracy (R² = 0.98, not 1.0). → **Consider:** treating model output as a decision-support estimate, not a guarantee, and validating any real reallocation with a small live test before scaling up.

## 7. Real-World Application

A model like this could support:
- **Marketing budget allocation** — simulating expected sales for different TV/Radio/Newspaper spend splits before committing budget.
- **Campaign optimization** — flagging which channel a marginal dollar is most likely to be well spent on, based on historical patterns.
- **Revenue/demand planning** — using predicted sales as an input to inventory or staffing forecasts.

**Limitations:**
- Only 200 observations and 3 predictor variables — no data on competitor activity, pricing, seasonality, economic conditions, or online/digital channels that likely also affect sales.
- No causal design (not a randomized experiment), so "spend more on X → more sales" is a hypothesis this data supports, not a proven fact.
- No time dimension, so the model cannot account for trends, seasonality, or how these relationships might change over time.
- Model accuracy, while high here, is estimated on a small 40-row test set, so results should be validated on more data before being used for high-stakes budget decisions.

## 8. Conclusion

This project analyzed the 200-row Advertising dataset (TV, Radio, Newspaper
spend and Sales) using a **regression approach**, since no date or segment
columns exist to support time-series or segmented analysis. Three models were
trained and compared; the **Random Forest Regressor performed best** (R² =
0.983, MAE ≈ 0.61), substantially outperforming plain Linear Regression
(R² = 0.899), which points to a non-linear relationship between spend and
sales. **TV spend is the strongest driver of sales**, followed by Radio, with
Newspaper contributing very little. The main business takeaway is that, based
on this historical data, advertising budget appears best concentrated in TV
and Radio, with Newspaper's contribution worth re-evaluating — while
recognizing these are associative, not proven causal, findings from a
relatively small, non-experimental dataset.

## Project Structure

```
SalesPrediction_Advertising/
├── data/
│   └── Advertising.csv              # original dataset (unmodified)
├── analysis/
│   └── sales_prediction_analysis.py # full end-to-end analysis script
├── charts/
│   ├── chart1_spend_vs_sales.png
│   ├── chart2_spend_by_channel.png
│   ├── chart3_correlation_heatmap.png
│   ├── chart4_sales_distribution.png
│   ├── chart5_actual_vs_predicted.png
│   └── chart6_feature_importance.png
├── reports/
│   ├── model_comparison.csv
│   └── actual_vs_predicted.csv
└── README.md
```

Run with: `cd analysis && python3 sales_prediction_analysis.py`
