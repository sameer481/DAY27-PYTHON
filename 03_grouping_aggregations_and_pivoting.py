"""
================================================================================
Day 27: Pandas Deep Dive - Part 3: Grouping, Aggregations, Pivots & Reshaping
================================================================================
The ability to slice, dice, aggregate, reshape, and calculate window metrics
over high-dimensional tabular data is the hallmark of modern data engineering
and business intelligence. Pandas implements Hadley Wickham's foundational
Split-Apply-Combine paradigm with extraordinary flexibility.

Theoretical Deep-Dive:
----------------------
1. The Split-Apply-Combine Paradigm:
   - Split: Divides the DataFrame into discrete subsets based on one or more keys.
     Pandas uses an internal hash-table grouping algorithm (O(N) time complexity)
     to map unique key combinations to row index arrays without copying underlying data.
   - Apply: Executes a user function or optimized Cython engine across each subset.
     The three functional archetypes of Apply are:
       a. Aggregation (.agg()): Reduces each group (N rows) down to a single scalar value.
       b. Transformation (.transform()): Returns a result of the EXACT same shape (N rows)
          as the parent group (e.g. z-scoring, cumulative sums, imputing group medians).
       c. Filtration (.filter()): Evaluates a group-level boolean predicate, either
          retaining all rows in the group or dropping the entire group.
   - Combine: Reassembles the discrete group results into a unified Series or DataFrame.

2. Tidy Data & Reshaping Mechanics:
   - Tidy Data (Codd / Wickham standard):
     1. Each variable forms a column.
     2. Each observation forms a row.
     3. Each type of observational unit forms a table.
   - Pivoting (.pivot() & .pivot_table()): Transforms long (tidy) data into wide format
     for cross-tabulation, reporting, and matrix math.
   - Melting (pd.melt()): Inverts a wide table back into long format by unpivoting
     feature columns into variable-value pairs.
   - Stacking (.stack()) & Unstacking (.unstack()): Pivots innermost or specified levels
     between column headers and MultiIndex row indices.

3. Window Functions (Rolling, Expanding, Exponentially Weighted):
   - Rolling Window: Computes statistics over a sliding window of fixed width W.
   - Expanding Window: Computes cumulative statistics from the beginning of time t=0 up to t.
   - Exponentially Weighted Moving Average (EWMA / .ewm()): Weights recent observations
     exponentially higher than past observations (crucial for volatility modeling and signals).

Contents:
---------
1. The Split-Apply-Combine Engine (GroupBy Internals & Iteration)
2. Named Aggregations & Multi-Column Custom Metrics
3. Group Transformations vs. Group Filtration (.transform & .filter)
4. Multi-Dimensional Pivoting, Margins (Subtotals) & Cross-Tabulations
5. Tidy Data Reshaping: pd.melt, .stack, and .unstack
6. Window Analytics: Rolling Windows, Expanding Windows & EWMA
================================================================================
"""

import numpy as np
import pandas as pd


def print_section(title: str) -> None:
    """Helper utility for formatted console section banners."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. THE SPLIT-APPLY-COMBINE ENGINE (GROUPBY INTERNALS & ITERATION)
# ==============================================================================
print_section("1. The Split-Apply-Combine Engine (GroupBy Internals & Iteration)")

# Global enterprise sales transactions dataset
sales_data = pd.DataFrame({
    "Region": ["EMEA", "APAC", "AMER", "EMEA", "AMER", "APAC", "AMER", "EMEA"],
    "Product": ["Cloud", "SaaS", "Hardware", "SaaS", "Cloud", "Cloud", "Hardware", "Cloud"],
    "SalesRep": ["Klaus", "Mei", "Jack", "Sarah", "Emily", "Kenji", "Carlos", "Fatima"],
    "DealSizeUSD": [120_000, 45_000, 310_000, 85_000, 220_000, 95_000, 180_000, 140_000],
    "DiscountPct": [0.10, 0.05, 0.15, 0.08, 0.12, 0.00, 0.20, 0.05]
})

print("--- Enterprise Sales Deals ---")
print(sales_data)

# GroupBy creates a lazy DataFrameGroupBy object without evaluating immediately
grouped_region = sales_data.groupby("Region")
print(f"\nGroupBy Object Type: {type(grouped_region).__name__}")
print(f"Number of Unique Groups: {grouped_region.ngroups}")
print("Group Indices Mapping (Row indices per group):")
for name, indices in grouped_region.groups.items():
    print(f"  {name}: {indices.tolist()}")

# Extracting a single group without scanning the entire DataFrame
print("\n--- Direct Extraction of APAC Deals (.get_group('APAC')) ---")
print(grouped_region.get_group("APAC"))


# ==============================================================================
# 2. NAMED AGGREGATIONS & MULTI-COLUMN CUSTOM METRICS
# ==============================================================================
print_section("2. Named Aggregations & Multi-Column Custom Metrics")

# Modern Pandas Named Aggregation: Clean, explicit, and assigns custom column names
# Syntax: new_col_name=(source_col, aggfunc)
regional_performance = sales_data.groupby("Region").agg(
    TotalRevenueUSD=("DealSizeUSD", "sum"),
    MeanDealSizeUSD=("DealSizeUSD", "mean"),
    MaxDealUSD=("DealSizeUSD", "max"),
    AvgDiscount=("DiscountPct", "mean"),
    DealCount=("SalesRep", "count")
).round(2)

print("--- Regional Sales Performance Summary (Named Aggregations) ---")
print(regional_performance)

# Multi-level Grouping (Region + Product) with custom Lambda and numpy ufuncs
multi_grouped = sales_data.groupby(["Region", "Product"], observed=False).agg(
    RevenueSum=("DealSizeUSD", "sum"),
    RevenueStd=("DealSizeUSD", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0)
).round(2)

print("\n--- Multi-Key Grouping (Region x Product) ---")
print(multi_grouped)


# ==============================================================================
# 3. GROUP TRANSFORMATIONS VS. GROUP FILTRATION (.transform & .filter)
# ==============================================================================
print_section("3. Group Transformations vs. Group Filtration (.transform & .filter)")

# Transformation: Returns a Series of the EXACT same length as the original DataFrame.
# Ideal for computing group-relative benchmarks, z-scores, and deviations from group mean.
sales_data["RegionMeanDeal"] = sales_data.groupby("Region")["DealSizeUSD"].transform("mean")
sales_data["DealVsRegionMeanDiff"] = sales_data["DealSizeUSD"] - sales_data["RegionMeanDeal"]
sales_data["DealZScoreInRegion"] = sales_data.groupby("Region")["DealSizeUSD"].transform(
    lambda s: (s - s.mean()) / (s.std(ddof=1) if len(s) > 1 else 1.0)
)

print("--- DataFrame Augmented with Group Transformations (.transform) ---")
print(sales_data[["Region", "SalesRep", "DealSizeUSD", "RegionMeanDeal", "DealVsRegionMeanDiff", "DealZScoreInRegion"]].round(2))

# Filtration: Keeps or drops entire groups based on a collective predicate.
# Example: Retain only regions that generated more than $300,000 in total sales!
high_performing_regions = sales_data.groupby("Region").filter(
    lambda group: group["DealSizeUSD"].sum() > 300_000
)

print("\n--- Filtered: Only Regions with Aggregate Revenue > $300k (.filter) ---")
print(high_performing_regions[["Region", "SalesRep", "DealSizeUSD"]])


# ==============================================================================
# 4. MULTI-DIMENSIONAL PIVOTING, MARGINS & CROSS-TABULATIONS
# ==============================================================================
print_section("4. Multi-Dimensional Pivoting, Margins & Cross-Tabulations")

# Creating a richer dataset for business reporting
expanded_deals = pd.DataFrame({
    "Quarter": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2", "Q1", "Q2", "Q1", "Q2"],
    "Industry": ["Finance", "Tech", "Healthcare", "Finance", "Tech", "Healthcare", "Finance", "Healthcare", "Tech", "Finance"],
    "Tier": ["Enterprise", "SMB", "Enterprise", "SMB", "Enterprise", "SMB", "SMB", "Enterprise", "SMB", "Enterprise"],
    "ContractValue": [450_000, 75_000, 320_000, 110_000, 580_000, 90_000, 85_000, 410_000, 95_000, 520_000]
})

print("--- Expanded Deals Records ---")
print(expanded_deals.head(6))

# Pivot Table with Margins (Subtotals across All rows and columns)
pivot_report = pd.pivot_table(
    data=expanded_deals,
    values="ContractValue",
    index="Industry",
    columns="Tier",
    aggfunc=["sum", "mean"],
    margins=True,
    margins_name="TOTAL",
    fill_value=0
)

print("\n--- Multi-Level Pivot Table with Margins (Subtotals) ---")
print(pivot_report.round(0))

# Cross-tabulation (pd.crosstab) for frequency distributions & joint probabilities
crosstab_pct = pd.crosstab(
    index=expanded_deals["Industry"],
    columns=expanded_deals["Tier"],
    normalize="all"  # Joint probability table summing to 1.0
) * 100

print("\n--- Joint Probability Distribution Cross-Tabulation (% of All Deals) ---")
print(crosstab_pct.round(2))


# ==============================================================================
# 5. TIDY DATA RESHAPING: pd.melt, .stack, AND .unstack
# ==============================================================================
print_section("5. Tidy Data Reshaping: pd.melt, .stack, and .unstack")

# Wide Financial Table (Untidy: Quarters are spread across columns)
wide_financials = pd.DataFrame({
    "Company": ["Apex Corp", "Zenith LLC", "Quantum Inc"],
    "Q1_Revenue": [140.5, 95.2, 210.8],
    "Q2_Revenue": [155.0, 99.4, 225.1],
    "Q3_Revenue": [162.3, 104.1, 240.0],
    "Q4_Revenue": [180.7, 115.8, 265.4]
})

print("--- Wide Format Financials (Untidy) ---")
print(wide_financials)

# Melting wide table into tidy long format (Variables in columns, observations in rows)
tidy_financials = pd.melt(
    wide_financials,
    id_vars=["Company"],
    value_vars=["Q1_Revenue", "Q2_Revenue", "Q3_Revenue", "Q4_Revenue"],
    var_name="Period",
    value_name="RevenueUSD_M"
)

# Clean up 'Period' string: 'Q1_Revenue' -> 'Q1'
tidy_financials["Period"] = tidy_financials["Period"].str.replace("_Revenue", "")
print("\n--- Melted Tidy Long Format (pd.melt) ---")
print(tidy_financials.head(8))

# Demonstrating .stack() and .unstack() on MultiIndex
indexed_financials = wide_financials.set_index("Company")
stacked_series = indexed_financials.stack()
print("\n--- Stacked into MultiIndex Series (.stack) ---")
print(stacked_series.head(6))

unstacked_df = stacked_series.unstack()
print("\n--- Unstacked back into Original DataFrame (.unstack) ---")
print(unstacked_df)


# ==============================================================================
# 6. WINDOW ANALYTICS: ROLLING WINDOWS, EXPANDING WINDOWS & EWMA
# ==============================================================================
print_section("6. Window Analytics: Rolling Windows, Expanding Windows & EWMA")

# Simulating daily stock price observations over 20 trading sessions
rng = np.random.default_rng(42)
daily_shocks = rng.normal(loc=0.5, scale=2.5, size=20)
base_price = 100.0
price_path = base_price + np.cumsum(daily_shocks)

ts_df = pd.DataFrame({
    "Date": pd.date_range("2026-01-01", periods=20, freq="B"),
    "ClosePrice": price_path
}).set_index("Date")

# 1. Rolling 5-day Moving Average and Volatility
ts_df["MA_5D"] = ts_df["ClosePrice"].rolling(window=5, min_periods=1).mean()
ts_df["Vol_5D"] = ts_df["ClosePrice"].rolling(window=5, min_periods=2).std()

# 2. Expanding Window: Historical Maximum up to each day
ts_df["ExpandingPeak"] = ts_df["ClosePrice"].expanding(min_periods=1).max()
ts_df["DrawdownFromPeak"] = ts_df["ClosePrice"] - ts_df["ExpandingPeak"]

# 3. Exponentially Weighted Moving Average (EWMA with 5-day decay span)
ts_df["EWMA_Span5"] = ts_df["ClosePrice"].ewm(span=5, adjust=False).mean()

print("--- Financial Window Metrics (Rolling, Expanding, EWMA) ---")
print(ts_df.tail(10).round(2))

print(f"\nCompleted Part 3 successfully! Ready for Part 4.")
