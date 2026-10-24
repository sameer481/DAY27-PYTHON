"""
================================================================================
Day 27: Pandas Deep Dive - Part 5: Performance Playbook & Production Systems
================================================================================
Mastering Pandas requires knowing not just what is syntactically possible,
but what is computationally optimal. A poorly written Pandas loop can take
hours, while an idiomatic vectorized implementation completes in milliseconds.

This module delivers:
1. The Definitive Pandas Performance Hierarchy Benchmark (100x speedup proof)
2. Chunked Out-of-Core Processing for Massive Datasets (Memory Streaming)
3. Three Complete End-to-End Production Systems Built Strictly from Scratch:
   - Project 1: E-Commerce Cohort Retention & RFM Customer Segmentation Engine
   - Project 2: Algorithmic Trading Strategy Backtester & Risk Performance Engine
   - Project 3: Healthcare Longitudinal Clinical Trial ETL & Efficacy Pipeline
================================================================================
"""

import io
import time
import numpy as np
import pandas as pd


def print_section(title: str) -> None:
    """Helper utility for formatted console section banners."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. THE DEFINITIVE PANDAS PERFORMANCE HIERARCHY BENCHMARK
# ==============================================================================
print_section("1. The Definitive Pandas Performance Hierarchy Benchmark")

# We will benchmark 4 common approaches to computing a custom business metric:
# Target: Adjusted Taxed Revenue = Price * Quantity * (1.0 - Discount) * TaxRate

N_RECORDS = 100_000
rng = np.random.default_rng(42)

benchmark_df = pd.DataFrame({
    "Price": rng.uniform(10.0, 500.0, size=N_RECORDS),
    "Quantity": rng.integers(1, 10, size=N_RECORDS),
    "Discount": rng.choice([0.0, 0.05, 0.10, 0.15], size=N_RECORDS),
    "TaxRate": 0.08
})

print(f"Benchmarking on {N_RECORDS:,} records...\n")

# Approach A: df.iterrows() (ANTI-PATTERN! Creates a heavy Series object per row)
# We test only on first 5,000 rows and extrapolate to prevent freezing the CPU
sub_n = 5_000
sub_df = benchmark_df.iloc[:sub_n].copy()

t0 = time.perf_counter()
res_iterrows = []
for _, row in sub_df.iterrows():
    val = row["Price"] * row["Quantity"] * (1.0 - row["Discount"]) * (1.0 + row["TaxRate"])
    res_iterrows.append(val)
t_iterrows_sub = time.perf_counter() - t0
t_iterrows_extrapolated = t_iterrows_sub * (N_RECORDS / sub_n)

# Approach B: df.itertuples() (Lightweight namedtuple C-iteration)
t0 = time.perf_counter()
res_itertuples = []
for row in benchmark_df.itertuples(index=False):
    val = row.Price * row.Quantity * (1.0 - row.Discount) * (1.0 + row.TaxRate)
    res_itertuples.append(val)
t_itertuples = time.perf_counter() - t0

# Approach C: df.apply() with Lambda
t0 = time.perf_counter()
res_apply = benchmark_df.apply(
    lambda r: r["Price"] * r["Quantity"] * (1.0 - r["Discount"]) * (1.0 + r["TaxRate"]),
    axis=1
)
t_apply = time.perf_counter() - t0

# Approach D: Pure Vectorized Pandas / NumPy SIMD (Idiomatic Best Practice)
t0 = time.perf_counter()
res_vectorized = (
    benchmark_df["Price"]
    * benchmark_df["Quantity"]
    * (1.0 - benchmark_df["Discount"])
    * (1.0 + benchmark_df["TaxRate"])
)
t_vectorized = time.perf_counter() - t0

print(f"{'Methodology':<35} | {'Execution Time':<15} | {'Relative Speedup':<15}")
print("-" * 72)
print(f"{'1. df.iterrows() [Extrapolated]':<35} | {t_iterrows_extrapolated * 1000:>10.2f} ms | {'1.0x (Baseline)':<15}")
print(f"{'2. df.apply(axis=1)':<35} | {t_apply * 1000:>10.2f} ms | {t_iterrows_extrapolated / t_apply:>10.1f}x faster")
print(f"{'3. df.itertuples()':<35} | {t_itertuples * 1000:>10.2f} ms | {t_iterrows_extrapolated / t_itertuples:>10.1f}x faster")
print(f"{'4. Vectorized SIMD Pandas':<35} | {t_vectorized * 1000:>10.2f} ms | {t_iterrows_extrapolated / t_vectorized:>10.1f}x faster")

print(f"\n>> Vectorized Pandas is ~{t_iterrows_extrapolated / t_vectorized:.0f}x faster than iterrows()!")


# ==============================================================================
# 2. CHUNKED OUT-OF-CORE STREAMING ENGINE
# ==============================================================================
print_section("2. Chunked Out-of-Core Streaming Engine (Memory Efficient)")

# Simulating reading a 10GB+ CSV file that exceeds physical RAM
# In production, use `pd.read_csv(filepath, chunksize=N)`
csv_buffer = io.StringIO()
csv_buffer.write("Region,DealSizeUSD,Status\n")
for _ in range(50_000):
    reg = rng.choice(["NA", "EMEA", "APAC", "LATAM"])
    amt = rng.integers(100, 5000)
    st = rng.choice(["Won", "Lost", "Pending"])
    csv_buffer.write(f"{reg},{amt},{st}\n")
csv_buffer.seek(0)

# Process in chunks of 10,000 rows without ever holding entire dataset in RAM
chunk_size = 10_000
running_regional_revenue = {}
total_won_deals = 0

for chunk_idx, chunk in enumerate(pd.read_csv(csv_buffer, chunksize=chunk_size)):
    # Clean and filter on-the-fly
    won_chunk = chunk[chunk["Status"] == "Won"]
    total_won_deals += len(won_chunk)

    # Accumulate aggregate metrics
    group_sum = won_chunk.groupby("Region")["DealSizeUSD"].sum()
    for region, revenue in group_sum.items():
        running_regional_revenue[region] = running_regional_revenue.get(region, 0) + revenue

print(f"Processed 50,000 rows in {chunk_idx + 1} discrete streaming chunks.")
print(f"Total 'Won' Deals Extracted: {total_won_deals:,}")
print("Accumulated Regional Revenue from Out-of-Core Streaming:")
for reg, rev in sorted(running_regional_revenue.items()):
    print(f"  {reg:<6}: ${rev:,.2f}")


# ==============================================================================
# 3. PRODUCTION SYSTEM 1: E-COMMERCE COHORT RETENTION & RFM ENGINE
# ==============================================================================
print_section("3. Production System 1: E-Commerce Cohort Retention & RFM Engine")

# Generating 12 months of customer transaction events
n_events = 25_000
customer_pool = [f"CUST_{i:04d}" for i in range(1, 1201)]
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-12-31")

random_timestamps = start_date + pd.to_timedelta(
    rng.integers(0, int((end_date - start_date).total_seconds()), size=n_events),
    unit="s"
)

raw_ecommerce = pd.DataFrame({
    "OrderID": range(100_000, 100_000 + n_events),
    "CustomerID": rng.choice(customer_pool, size=n_events),
    "OrderDate": random_timestamps,
    "OrderTotalUSD": rng.exponential(scale=75.0, size=n_events) + 15.0
})

# --- A. COHORT RETENTION MATRIX ---
# 1. Truncate transaction date to monthly period
raw_ecommerce["OrderMonth"] = raw_ecommerce["OrderDate"].dt.to_period("M")

# 2. Identify the first purchase month (Cohort Month) for each customer
customer_first_month = raw_ecommerce.groupby("CustomerID")["OrderMonth"].min()
raw_ecommerce["CohortMonth"] = raw_ecommerce["CustomerID"].map(customer_first_month)

# 3. Calculate Cohort Index (integer distance in months since acquisition)
raw_ecommerce["CohortIndex"] = (
    (raw_ecommerce["OrderMonth"].dt.year - raw_ecommerce["CohortMonth"].dt.year) * 12
    + (raw_ecommerce["OrderMonth"].dt.month - raw_ecommerce["CohortMonth"].dt.month)
)

# 4. Construct Cohort Pivot Table
cohort_data = raw_ecommerce.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index="CohortMonth", columns="CohortIndex", values="CustomerID")

# 5. Calculate Retention Percentage relative to Month 0 base
cohort_sizes = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100.0

print("--- Monthly Customer Cohort Retention Rate (%) Matrix ---")
print(retention_matrix.iloc[:6, :7].round(1))

# --- B. RFM (RECENCY, FREQUENCY, MONETARY) SEGMENTATION ---
snapshot_date = raw_ecommerce["OrderDate"].max() + pd.Timedelta(days=1)

rfm_table = raw_ecommerce.groupby("CustomerID").agg(
    RecencyDays=("OrderDate", lambda dates: (snapshot_date - dates.max()).days),
    Frequency=("OrderID", "count"),
    MonetaryValue=("OrderTotalUSD", "sum")
)

# Assign RFM quantiles (1 to 4 scores)
rfm_table["R_Score"] = pd.qcut(rfm_table["RecencyDays"], q=4, labels=[4, 3, 2, 1])  # Recent is better -> higher score
rfm_table["F_Score"] = pd.qcut(rfm_table["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
rfm_table["M_Score"] = pd.qcut(rfm_table["MonetaryValue"], q=4, labels=[1, 2, 3, 4])

# RFM Segment Classification
def assign_segment(row) -> str:
    r, f = int(row["R_Score"]), int(row["F_Score"])
    if r >= 3 and f >= 3:
        return "Champions / Core Loyal"
    elif r >= 3 and f < 3:
        return "Promising New"
    elif r < 2 and f >= 3:
        return "At-Risk / Needs Nurturing"
    else:
        return "Hibernating / Lost"

rfm_table["Segment"] = rfm_table.apply(assign_segment, axis=1)

print("\n--- Customer Persona RFM Segment Distribution ---")
segment_summary = rfm_table.groupby("Segment").agg(
    CustomerCount=("RecencyDays", "count"),
    AvgRecency=("RecencyDays", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("MonetaryValue", "mean")
).round(1)
print(segment_summary)


# ==============================================================================
# 4. PRODUCTION SYSTEM 2: ALGORITHMIC TRADING QUANTITATIVE BACKTESTER
# ==============================================================================
print_section("4. Production System 2: Algorithmic Trading Quantitative Backtester")

# Generate 252 business days of simulated asset pricing
n_bars = 252
dates = pd.bdate_range("2025-01-01", periods=n_bars)
daily_innovations = rng.normal(loc=0.0006, scale=0.018, size=n_bars)
prices = 100.0 * np.cumprod(1.0 + daily_innovations)

market_data = pd.DataFrame({"Close": prices}, index=dates)

# Strategy: Dual Moving Average Crossover (Fast 10D SMA vs Slow 30D SMA) with Volatility filter
market_data["SMA_Fast"] = market_data["Close"].rolling(window=10).mean()
market_data["SMA_Slow"] = market_data["Close"].rolling(window=30).mean()
market_data["Vol_20D"] = market_data["Close"].pct_change().rolling(window=20).std()

# Signal Generation: 1 = Long, 0 = Cash
# Enter Long when Fast SMA crosses above Slow SMA; Exit to cash when Fast SMA drops below Slow SMA
market_data["Signal"] = np.where(market_data["SMA_Fast"] > market_data["SMA_Slow"], 1, 0)
# Shift position by 1 day to prevent lookahead bias (execute at next day's open/close)
market_data["Position"] = market_data["Signal"].shift(1).fillna(0)

# Backtest Performance Metrics
market_data["AssetReturn"] = market_data["Close"].pct_change().fillna(0)
market_data["StrategyReturn"] = market_data["Position"] * market_data["AssetReturn"]

# Cumulative Equity Curves
market_data["BuyHoldEquity"] = (1.0 + market_data["AssetReturn"]).cumprod()
market_data["StrategyEquity"] = (1.0 + market_data["StrategyReturn"]).cumprod()

# Performance KPIs
annualized_factor = 252
strat_cagr = (market_data["StrategyEquity"].iloc[-1] ** (annualized_factor / n_bars)) - 1.0
bh_cagr = (market_data["BuyHoldEquity"].iloc[-1] ** (annualized_factor / n_bars)) - 1.0

strat_vol = market_data["StrategyReturn"].std() * np.sqrt(annualized_factor)
strat_sharpe = (market_data["StrategyReturn"].mean() * annualized_factor) / (strat_vol if strat_vol > 0 else 1.0)

# Peak and Max Drawdown
market_data["Peak"] = market_data["StrategyEquity"].cummax()
market_data["Drawdown"] = (market_data["StrategyEquity"] - market_data["Peak"]) / market_data["Peak"]
strat_max_dd = market_data["Drawdown"].min()

print("--- Algorithmic Strategy Backtest Results (SMA Crossover) ---")
print(f"  Asset Final Return:        {(market_data['BuyHoldEquity'].iloc[-1] - 1.0) * 100:.2f}%")
print(f"  Strategy Final Return:     {(market_data['StrategyEquity'].iloc[-1] - 1.0) * 100:.2f}%")
print(f"  Strategy Annualized CAGR:  {strat_cagr * 100:.2f}%")
print(f"  Annualized Volatility:     {strat_vol * 100:.2f}%")
print(f"  Annualized Sharpe Ratio:   {strat_sharpe:.2f}")
print(f"  Maximum Drawdown:          {strat_max_dd * 100:.2f}%")


# ==============================================================================
# 5. PRODUCTION SYSTEM 3: HEALTHCARE CLINICAL TRIAL ETL & EFFICACY PIPELINE
# ==============================================================================
print_section("5. Production System 3: Healthcare Clinical Trial ETL & Efficacy Pipeline")

# Real-world clinical trials track biomarker response across scheduled patient visit timelines
patients = [f"PT_{i:03d}" for i in range(1, 41)]
arms = ["Treatment_DrugX", "Placebo_Control"]
visits = ["Baseline", "Week_04", "Week_12", "Week_24"]

clinical_records = []
for pt in patients:
    arm = rng.choice(arms)
    base_systolic = rng.normal(loc=150.0, scale=10.0)  # Baseline stage 1/2 hypertension
    base_biomarker = rng.normal(loc=85.0, scale=12.0)

    for v_idx, visit in enumerate(visits):
        # 10% probability of a missed visit
        if v_idx > 0 and rng.uniform() < 0.10:
            continue

        # Simulating therapeutic treatment effect for DrugX vs Placebo
        if arm == "Treatment_DrugX":
            bp_delta = -4.5 * v_idx + rng.normal(0, 3.0)
            bio_delta = -6.0 * v_idx + rng.normal(0, 4.0)
        else:
            bp_delta = -0.5 * v_idx + rng.normal(0, 3.5)
            bio_delta = 0.2 * v_idx + rng.normal(0, 4.0)

        clinical_records.append({
            "PatientID": pt,
            "Arm": arm,
            "Visit": visit,
            "SystolicBP": base_systolic + bp_delta,
            "Biomarker_mg_dL": base_biomarker + bio_delta,
            "AdverseEvent": rng.choice([0, 1], p=[0.92, 0.08])
        })

trial_df = pd.DataFrame(clinical_records)
print(f"Ingested {len(trial_df)} Clinical Trial Observations across {len(patients)} Patients.")

# 1. Pivot Vitals by Visit Timeline
vitals_pivot = trial_df.pivot_table(
    index=["Arm", "PatientID"],
    columns="Visit",
    values=["SystolicBP", "Biomarker_mg_dL"],
    aggfunc="first"
)

# Flatten MultiIndex columns into clean single strings
vitals_pivot.columns = [f"{col[0]}_{col[1]}" for col in vitals_pivot.columns]

# 2. Calculate Efficacy: Delta from Baseline to Week 24
vitals_pivot["BP_Delta_W24"] = vitals_pivot["SystolicBP_Week_24"] - vitals_pivot["SystolicBP_Baseline"]
vitals_pivot["Bio_Delta_W24"] = vitals_pivot["Biomarker_mg_dL_Week_24"] - vitals_pivot["Biomarker_mg_dL_Baseline"]

# 3. Aggregate Statistical Efficacy by Clinical Trial Arm
efficacy_summary = vitals_pivot.groupby("Arm").agg(
    PatientsCompletedW24=("BP_Delta_W24", "count"),
    MeanSystolicReduction=("BP_Delta_W24", "mean"),
    SystolicReductionStd=("BP_Delta_W24", "std"),
    MeanBiomarkerReduction=("Bio_Delta_W24", "mean"),
    BiomarkerReductionStd=("Bio_Delta_W24", "std")
).round(2)

print("\n--- Clinical Trial Primary Endpoint Efficacy Summary (Delta to Baseline) ---")
print(efficacy_summary)

print(f"\nCompleted Part 5 successfully! Production systems operational.")
