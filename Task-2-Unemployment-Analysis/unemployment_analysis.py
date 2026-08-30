"""
=====================================================================
 UNEMPLOYMENT ANALYSIS WITH PYTHON
 A Data Analysis project exploring unemployment trends in India,
 the impact of COVID-19, and seasonal patterns.
=====================================================================

DATA SOURCE
-----------
This project uses TWO related CSV files from the attached archive:

1. "Unemployment in India.csv"
   - State-level, split by Rural / Urban area
   - Monthly observations from May 2019 to June 2020
   - This is the PRIMARY dataset: it is the only one that contains
     real pre-COVID data (2019), so it is used for the main trend,
     cleaning, EDA, COVID before/during comparison, and seasonal
     sections.

2. "Unemployment_Rate_upto_11_2020.csv"
   - State-level, with a geographic Zone (South/North/East/West/
     Northeast) and Region.1
   - Monthly observations from January 2020 to October 2020
   - This is the SECONDARY dataset: it extends a few months further
     into the second half of 2020 than the primary file, so it is
     used to look at the *recovery* period after the acute COVID
     shock, and for a zone-level (regional) comparison.

Both files come from the same underlying source (CMIE - Centre for
Monitoring Indian Economy) but are not identical row-for-row, so
they are analyzed as two related views of the same phenomenon
rather than merged into one table.
=====================================================================
"""

# ---------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------
# 2. LOAD DATASET
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 2: LOAD DATASET")
print("=" * 70)

df1_raw = pd.read_csv("Unemployment in India.csv")
df2_raw = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

print(f"\nFile 1 'Unemployment in India.csv' raw shape: {df1_raw.shape}")
print(f"File 2 'Unemployment_Rate_upto_11_2020.csv' raw shape: {df2_raw.shape}")


# ---------------------------------------------------------------
# 3. DATA INSPECTION
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: DATA INSPECTION (primary file)")
print("=" * 70)

# Clean up column names (the raw CSV has leading spaces, e.g. ' Date')
df1_raw.columns = [c.strip() for c in df1_raw.columns]
df2_raw.columns = [c.strip() for c in df2_raw.columns]

print("\nFirst 10 rows of the primary dataset:")
print(df1_raw.head(10))

print(f"\nShape: {df1_raw.shape[0]} rows, {df1_raw.shape[1]} columns")
print("\nColumn names:", df1_raw.columns.tolist())
print("\nData types:")
print(df1_raw.dtypes)
print("\nMissing values per column:")
print(df1_raw.isnull().sum())
print("\nFully empty rows (all columns NaN):", df1_raw.isnull().all(axis=1).sum())
print("\nDuplicate rows (excluding fully-empty ones):",
      df1_raw.dropna(how="all").duplicated().sum())

print("\nBasic statistics (numeric columns):")
print(df1_raw.describe())

print("\nUnique values in 'Area':", df1_raw["Area"].dropna().unique().tolist())
print("Unique values in 'Frequency':", df1_raw["Frequency"].dropna().unique().tolist())
print("Number of distinct Regions (states):", df1_raw["Region"].dropna().nunique())

# Identify the key columns:
#  - Date/time column: "Date"
#  - Unemployment-rate column: "Estimated Unemployment Rate (%)"
print("\n>> Date/time column: 'Date'")
print(">> Target metric column: 'Estimated Unemployment Rate (%)'")


# ---------------------------------------------------------------
# 4. DATA CLEANING
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: DATA CLEANING")
print("=" * 70)

df1 = df1_raw.copy()

# The 28 fully-empty rows found above carry no information at all
# (every single column is NaN) - they are dropped. This is NOT
# deleting real data, just removing blank trailing rows in the file.
before = len(df1)
df1 = df1.dropna(how="all")
print(f"\nDropped {before - len(df1)} fully-empty rows.")

# Strip whitespace from text columns (the raw file has values like
# ' Monthly' with a leading space)
text_cols = df1.select_dtypes(include=["object","str"]).columns
for col in text_cols:
    df1[col] = df1[col].astype(str).str.strip()

# Check remaining missing values after dropping blank rows
print("\nMissing values after removing blank rows:")
print(df1.isnull().sum())
# -> No further missing values remain, so no imputation was needed.

# Convert the Date column to real datetime objects
df1["Date"] = pd.to_datetime(df1["Date"], format="%d-%m-%Y")

# Check for duplicate records (same Region + Area + Date)
dup_count = df1.duplicated(subset=["Region", "Area", "Date"]).sum()
print(f"\nDuplicate (Region, Area, Date) records: {dup_count}")

# Check for unusual / outlier values in the unemployment rate
rate_col = "Estimated Unemployment Rate (%)"
print(f"\n{rate_col} range: {df1[rate_col].min()}% to {df1[rate_col].max()}%")
extreme = df1[df1[rate_col] > 50]
print(f"Rows with an unemployment rate above 50%: {len(extreme)}")
print(extreme[["Region", "Date", "Area", rate_col]].sort_values(rate_col, ascending=False).head(10))
print("\nNote: these very high values are concentrated in April-May 2020,")
print("exactly the period of India's strict COVID-19 lockdown, so they are")
print("treated as genuine (if extreme) observations, not data errors.")

# Sort the data chronologically
df1 = df1.sort_values(["Date", "Region", "Area"]).reset_index(drop=True)

# Create useful date-based columns
df1["Year"] = df1["Date"].dt.year
df1["Month"] = df1["Date"].dt.month
df1["MonthName"] = df1["Date"].dt.strftime("%b")
df1["Quarter"] = df1["Date"].dt.quarter

print("\nCleaned primary dataset preview:")
print(df1.head())
print(f"\nFinal cleaned shape: {df1.shape}")
print(f"Date range: {df1['Date'].min().date()} to {df1['Date'].max().date()}")

# --- Clean the secondary file the same way ---
df2 = df2_raw.copy()
df2.columns = [c.strip() for c in df2.columns]
# This file has "Region" twice (state name, then zone) - rename for clarity
df2 = df2.rename(columns={"Region.1": "Zone"})
for col in df2.select_dtypes(include=["object","str"]).columns:
    df2[col] = df2[col].astype(str).str.strip()
df2["Date"] = pd.to_datetime(df2["Date"], format="%d-%m-%Y")
df2 = df2.sort_values(["Date", "Region"]).reset_index(drop=True)
df2["Year"] = df2["Date"].dt.year
df2["Month"] = df2["Date"].dt.month
df2["MonthName"] = df2["Date"].dt.strftime("%b")

print(f"\nSecondary dataset cleaned. Shape: {df2.shape}, "
      f"Date range: {df2['Date'].min().date()} to {df2['Date'].max().date()}")
print("Zones available:", df2["Zone"].unique().tolist())


# ---------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS (primary dataset)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

overall_mean = df1[rate_col].mean()
overall_min = df1[rate_col].min()
overall_max = df1[rate_col].max()
overall_median = df1[rate_col].median()
overall_std = df1[rate_col].std()

print(f"\nOverall average unemployment rate: {overall_mean:.2f}%")
print(f"Overall median unemployment rate:  {overall_median:.2f}%")
print(f"Minimum unemployment rate:         {overall_min:.2f}%")
print(f"Maximum unemployment rate:         {overall_max:.2f}%")
print(f"Standard deviation:                {overall_std:.2f}%")

min_row = df1.loc[df1[rate_col].idxmin()]
max_row = df1.loc[df1[rate_col].idxmax()]
print(f"\nLowest rate observed: {min_row[rate_col]}% in {min_row['Region']} "
      f"({min_row['Area']}) on {min_row['Date'].date()}")
print(f"Highest rate observed: {max_row[rate_col]}% in {max_row['Region']} "
      f"({max_row['Area']}) on {max_row['Date'].date()}")

# National average unemployment rate over time (averaging across all
# states/areas reported for each date)
monthly_national = df1.groupby("Date")[rate_col].mean().reset_index()
print("\nNational average unemployment rate by month:")
print(monthly_national.to_string(index=False))

# Which month had the highest / lowest NATIONAL average
highest_month = monthly_national.loc[monthly_national[rate_col].idxmax()]
lowest_month = monthly_national.loc[monthly_national[rate_col].idxmin()]
print(f"\nHighest national average: {highest_month[rate_col]:.2f}% "
      f"in {highest_month['Date'].date()}")
print(f"Lowest national average:  {lowest_month[rate_col]:.2f}% "
      f"in {lowest_month['Date'].date()}")

# Top / bottom 5 states by average unemployment rate over the whole period
state_avg = df1.groupby("Region")[rate_col].mean().sort_values(ascending=False)
print("\nTop 5 states by average unemployment rate (whole period):")
print(state_avg.head(5))
print("\nBottom 5 states by average unemployment rate (whole period):")
print(state_avg.tail(5))

# Rural vs Urban comparison
area_avg = df1.groupby("Area")[rate_col].mean()
print("\nAverage unemployment rate by Area:")
print(area_avg)


# ---------------------------------------------------------------
# 6. COVID-19 IMPACT ANALYSIS
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: COVID-19 IMPACT ANALYSIS")
print("=" * 70)

# The dataset's own date range is used to define the periods.
# India's nationwide lockdown began on 25 March 2020, so:
#   Pre-COVID : all dates before March 2020 (available: May 2019 - Feb 2020)
#   COVID     : March 2020 onward (available in this file up to June 2020)
# The primary file ends in June 2020, so it CANNOT show a post-COVID
# recovery period - that is analyzed separately below using the
# secondary file, which runs through October 2020.

pre_covid = df1[df1["Date"] < "2020-03-01"]
covid_period = df1[(df1["Date"] >= "2020-03-01") & (df1["Date"] <= df1["Date"].max())]

pre_covid_avg = pre_covid[rate_col].mean()
covid_avg = covid_period[rate_col].mean()
covid_peak = covid_period[rate_col].max()
covid_peak_row = covid_period.loc[covid_period[rate_col].idxmax()]

change = covid_avg - pre_covid_avg
pct_change = (change / pre_covid_avg) * 100

print(f"\nPre-COVID period in this file: "
      f"{pre_covid['Date'].min().date()} to {pre_covid['Date'].max().date()}")
print(f"COVID period in this file:     "
      f"{covid_period['Date'].min().date()} to {covid_period['Date'].max().date()}")

print(f"\nAverage unemployment rate BEFORE COVID: {pre_covid_avg:.2f}%")
print(f"Average unemployment rate DURING COVID (in this file): {covid_avg:.2f}%")
print(f"Change: {change:+.2f} percentage points ({pct_change:+.1f}% relative change)")
print(f"\nPeak unemployment rate during COVID: {covid_peak:.2f}% "
      f"in {covid_peak_row['Region']} ({covid_peak_row['Area']}) "
      f"on {covid_peak_row['Date'].date()}")

print("\n>> NOTE: This primary file ends on "
      f"{df1['Date'].max().date()}, so a genuine 'post-COVID' recovery "
      "period cannot be assessed from it alone.")

# --- Extending into recovery using the secondary file ---
print("\n--- Extending the timeline with the secondary file (up to Oct 2020) ---")
pre_covid2 = df2[df2["Date"] < "2020-03-01"]
covid2 = df2[(df2["Date"] >= "2020-03-01") & (df2["Date"] <= "2020-05-31")]
post_covid2 = df2[df2["Date"] > "2020-05-31"]

print(f"Pre-COVID window (secondary file): "
      f"{pre_covid2['Date'].min().date()} to {pre_covid2['Date'].max().date()} "
      f"-> avg {pre_covid2[rate_col].mean():.2f}%")
print(f"Acute COVID window (secondary file): "
      f"{covid2['Date'].min().date()} to {covid2['Date'].max().date()} "
      f"-> avg {covid2[rate_col].mean():.2f}%")
print(f"Post-acute / recovery window (secondary file): "
      f"{post_covid2['Date'].min().date()} to {post_covid2['Date'].max().date()} "
      f"-> avg {post_covid2[rate_col].mean():.2f}%")

recovery_change = post_covid2[rate_col].mean() - covid2[rate_col].mean()
print(f"\nChange from acute COVID period to recovery window: "
      f"{recovery_change:+.2f} percentage points")
if post_covid2[rate_col].mean() > pre_covid2[rate_col].mean():
    gap = post_covid2[rate_col].mean() - pre_covid2[rate_col].mean()
    print(f"As of {post_covid2['Date'].max().date()}, the average rate is still "
          f"{gap:.2f} points ABOVE the pre-COVID level -> recovery is partial, "
          "not complete, within the data available.")
else:
    print("By the end of the available data, the average rate had returned to "
          "or below the pre-COVID level.")


# ---------------------------------------------------------------
# 7. SEASONAL / TEMPORAL PATTERN ANALYSIS
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: SEASONAL AND TEMPORAL PATTERNS")
print("=" * 70)

month_year_counts = df1.groupby(["Month", "Year"]).size().unstack(fill_value=0)
print("\nNumber of observations per Month x Year (primary file):")
print(month_year_counts)

# Only months that appear in BOTH years give a genuine year-over-year comparison
months_in_both_years = month_year_counts[(month_year_counts.get(2019, 0) > 0) &
                                          (month_year_counts.get(2020, 0) > 0)].index.tolist()
print(f"\nMonths present in both 2019 and 2020: {months_in_both_years}")

if len(months_in_both_years) > 0:
    print("\n>> The dataset only overlaps for these month(s) across two years, so a "
          "reliable *multi-year* seasonal pattern cannot be established from this "
          "data alone (most months were only observed in a single year). "
          "Below is a direct year-over-year comparison for the overlapping month(s) only:")
    for m in months_in_both_years:
        val_2019 = df1[(df1["Month"] == m) & (df1["Year"] == 2019)][rate_col].mean()
        val_2020 = df1[(df1["Month"] == m) & (df1["Year"] == 2020)][rate_col].mean()
        month_name = pd.Timestamp(2020, m, 1).strftime("%B")
        print(f"  {month_name}: 2019 avg = {val_2019:.2f}%, 2020 avg = {val_2020:.2f}% "
              f"(diff {val_2020 - val_2019:+.2f} pts)")
else:
    print("\n>> No months overlap across both years - a seasonal pattern cannot "
          "be established from this dataset.")

# Descriptive month-by-month average across the whole available window
# (this reflects the actual trajectory observed, which is dominated by the
# COVID shock rather than a repeating yearly cycle - stated explicitly)
month_order = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
monthly_avg_overall = df1.groupby(df1["Date"].dt.to_period("M"))[rate_col].mean()
print("\nAverage unemployment rate by calendar month-period (chronological):")
print(monthly_avg_overall)


# ---------------------------------------------------------------
# 8. VISUALIZATIONS
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: VISUALIZATIONS")
print("=" * 70)

# --- Figure 1: National unemployment rate over time, COVID highlighted ---
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(monthly_national["Date"], monthly_national[rate_col],
        marker="o", color="tab:blue", linewidth=2)
ax.axvspan(pd.Timestamp("2020-03-01"), df1["Date"].max(),
           color="red", alpha=0.12, label="COVID-19 period (from Mar 2020)")
ax.set_title("India: National Average Unemployment Rate Over Time", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("01_unemployment_over_time.png", dpi=150)
plt.close()
print("Saved: 01_unemployment_over_time.png")

# --- Figure 2: Monthly average unemployment (bar chart) ---
fig, ax = plt.subplots(figsize=(11, 5))
labels = monthly_national["Date"].dt.strftime("%b %Y")
colors = ["crimson" if d >= pd.Timestamp("2020-03-01") else "steelblue"
          for d in monthly_national["Date"]]
ax.bar(labels, monthly_national[rate_col], color=colors)
ax.set_title("Average Unemployment Rate by Month (red = COVID period)", fontsize=13)
ax.set_xlabel("Month")
ax.set_ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("02_monthly_bar_chart.png", dpi=150)
plt.close()
print("Saved: 02_monthly_bar_chart.png")

# --- Figure 3: Rural vs Urban unemployment over time ---
area_over_time = df1.groupby(["Date", "Area"])[rate_col].mean().reset_index()
fig, ax = plt.subplots(figsize=(11, 5.5))
for area, sub in area_over_time.groupby("Area"):
    ax.plot(sub["Date"], sub[rate_col], marker="o", label=area, linewidth=2)
ax.axvspan(pd.Timestamp("2020-03-01"), df1["Date"].max(), color="red", alpha=0.1)
ax.set_title("Rural vs Urban Unemployment Rate Over Time", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45, ha="right")
ax.legend(title="Area")
plt.tight_layout()
plt.savefig("03_rural_vs_urban.png", dpi=150)
plt.close()
print("Saved: 03_rural_vs_urban.png")

# --- Figure 4: Top 10 states by average unemployment rate ---
fig, ax = plt.subplots(figsize=(9, 6))
top10 = state_avg.head(10).sort_values()
ax.barh(top10.index, top10.values, color="darkorange")
ax.set_title("Top 10 States by Average Unemployment Rate (May 2019 - Jun 2020)", fontsize=12)
ax.set_xlabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("04_top10_states.png", dpi=150)
plt.close()
print("Saved: 04_top10_states.png")

# --- Figure 5: Regional (zone) comparison using the secondary file ---
zone_over_time = df2.groupby(["Date", "Zone"])[rate_col].mean().reset_index()
fig, ax = plt.subplots(figsize=(11, 5.5))
for zone, sub in zone_over_time.groupby("Zone"):
    ax.plot(sub["Date"], sub[rate_col], marker="o", label=zone, linewidth=2)
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-31"),
           color="red", alpha=0.1, label="Acute COVID (Mar-May 2020)")
ax.set_title("Unemployment Rate by Zone, Jan-Oct 2020 (secondary file)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45, ha="right")
ax.legend(title="Zone", fontsize=8)
plt.tight_layout()
plt.savefig("05_zone_comparison.png", dpi=150)
plt.close()
print("Saved: 05_zone_comparison.png")

print("\nAll visualizations saved as PNG files.")


# ---------------------------------------------------------------
# 9. KEY FINDINGS
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: KEY FINDINGS")
print("=" * 70)
print(f"""
1. Overall average unemployment rate (primary file, May 2019-Jun 2020): {overall_mean:.2f}%
   (min {overall_min:.2f}%, max {overall_max:.2f}%, std dev {overall_std:.2f}%).
2. The national average unemployment rate rose sharply from
   {pre_covid_avg:.2f}% before COVID to {covid_avg:.2f}% during the
   COVID period covered by this file - a change of {change:+.2f} points.
3. The single highest observed rate was {covid_peak:.2f}% in
   {covid_peak_row['Region']} ({covid_peak_row['Area']}) in
   {covid_peak_row['Date'].strftime('%B %Y')}.
4. Using the secondary file (through October 2020), the average rate
   fell from its acute-COVID level of {covid2[rate_col].mean():.2f}% to
   {post_covid2[rate_col].mean():.2f}% in the following months - a
   partial recovery, though {"still above" if post_covid2[rate_col].mean() > pre_covid2[rate_col].mean() else "back near"}
   the pre-COVID baseline of {pre_covid2[rate_col].mean():.2f}%.
5. Urban areas show an average rate of {area_avg['Urban']:.2f}% versus
   {area_avg['Rural']:.2f}% for rural areas over the full primary-file period.
6. {state_avg.index[0]} has the highest average unemployment rate in the
   primary file ({state_avg.iloc[0]:.2f}%), while {state_avg.index[-1]} has
   the lowest ({state_avg.iloc[-1]:.2f}%).
7. A true multi-year seasonal pattern cannot be confirmed: only
   {months_in_both_years} month(s) have data in both 2019 and 2020, so
   month-to-month variation in this dataset largely reflects the COVID
   shock rather than a repeating annual cycle.
""")


# ---------------------------------------------------------------
# 10. ECONOMIC AND SOCIAL POLICY INSIGHTS
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 10: ECONOMIC AND SOCIAL POLICY INSIGHTS")
print("=" * 70)
print("""
The points below are POSSIBLE implications suggested by the patterns in
this dataset. They are observations to consider, not definitive policy
recommendations - a full policy assessment would need more context
(e.g. sector-level, demographic, and income data) than this dataset provides.

- The sharp, sudden jump in unemployment around March-April 2020 lines
  up with the timing of COVID-19 lockdown measures, which suggests
  short-notice economic shocks can justify having rapid-response support
  programs (e.g. temporary unemployment benefits) ready in advance.
- Urban unemployment increased more than rural unemployment during the
  acute COVID period in this dataset, which may point to a case for
  targeted support for urban informal and service-sector workers who
  are more exposed to lockdown-style restrictions.
- States with persistently higher average unemployment rates (see the
  top-10 chart) could be considered candidates for workforce
  retraining or regional investment programs, though this dataset alone
  cannot explain WHY those states rank higher.
- The partial (not full) recovery seen by October 2020 in the secondary
  file suggests that a single relief measure may not be enough - sustained
  monitoring through the following months would matter for judging
  whether additional support is needed.
- Because the data shows meaningful regional/state variation, uniform
  nationwide policies may affect regions unevenly; region-specific
  monitoring could help tailor responses.
""")


# ---------------------------------------------------------------
# 11. CONCLUSION
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 11: CONCLUSION")
print("=" * 70)
print(f"""
This analysis of unemployment in India (primary file: May 2019-Jun 2020;
secondary file: Jan-Oct 2020) shows a clear and sharp increase in
unemployment coinciding with the COVID-19 lockdown period, rising from a
pre-COVID average of {pre_covid_avg:.2f}% to {covid_avg:.2f}% during the
COVID window captured in the primary file, with a peak of {covid_peak:.2f}%.
Data extending to October 2020 shows unemployment beginning to recede
from its acute peak, though not fully back to pre-COVID levels within
the available window. Rural and urban areas, and different states/zones,
were not affected equally, underlining that any policy response would
benefit from being informed by this kind of regional and temporal detail
rather than a single national figure alone.
""")

print("Done.")
