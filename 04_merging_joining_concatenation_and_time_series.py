"""
================================================================================
Day 27: Pandas Deep Dive - Part 4: Merging, Joining, Concat & Time Series
================================================================================
Modern data engineering pipelines assemble fragmented information across
relational databases, distributed streaming logs, and high-frequency sensor or
financial feeds. Pandas provides relational algebra (`merge`, `join`), fast
array concatenation (`concat`), specialized asynchronous alignment (`merge_asof`),
and an industry-standard Time Series manipulation toolkit.

Theoretical Deep-Dive:
----------------------
1. Relational Joins & The Merge Validation Engine:
   - Relational joins match records using hash-table joins over key columns.
   - Join Types:
     - `inner`: Intersection of keys.
     - `left` / `right`: Preserves left or right table, filling mismatches with NaN/NA.
     - `outer`: Union of all keys.
     - `cross`: Cartesian product of both tables.
   - The Merge Validation Parameter (`validate`):
     A common production nightmare is an unintended many-to-many Cartesian explosion,
     multiplying 10,000 rows into 100,000,000 rows. By passing `validate="one_to_many"`
     or `"one_to_one"`, Pandas asserts key uniqueness before executing the join!
   - Diagnostic Merge Indicators (`indicator=True`):
     Appends a `_merge` Categorical column detailing whether a row came from `left_only`,
     `right_only`, or `both`.

2. Asynchronous Matching via `pd.merge_asof()`:
   - In quantitative finance and IoT telemetry, timestamps from disparate sources
     rarely match down to the millisecond.
   - Standard relational joins fail because timestamps differ by milliseconds.
   - `pd.merge_asof()` executes an ordered merge by matching the nearest or last known
     preceding record (`direction="backward"`) within an optional `tolerance` time window.

3. Time Series Architecture:
   - `DatetimeIndex`: An immutable array of nanosecond-precision 64-bit timestamps.
   - Timezone Mechanics:
     - Timezone-naive: Represents local wall-clock time without UTC offset.
     - Timezone-aware: Anchored to UTC with historical offset transition tables (handling
       Daylight Saving Time (DST) gracefully).
   - Frequency Resampling (.resample()):
     - Downsampling: Aggregating higher frequencies into lower frequencies (e.g. 1-second ticks into 5-minute OHLC bars).
     - Upsampling: Projecting lower frequencies into higher frequencies, requiring forward-fill or spline interpolation.

Contents:
---------
1. Relational Joins (Inner, Left, Right, Outer, Diagnostic Indicators & Validation)
2. High-Frequency Asynchronous Matching with `pd.merge_asof()` (Quote-to-Trade Alignment)
3. Concatenation along Axes & Preventing DataFrame Fragmentation (`pd.concat`)
4. Time Series Foundations: DatetimeIndex, Frequency Aliases & Timezone Conversions
5. Resampling Engine: Downsampling to OHLC Financial Candlesticks & Upsampling
6. Financial Engineering: Shifts, Percentage Returns, Volatility & Rolling Drawdowns
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
# 1. RELATIONAL JOINS (INDICATORS, VALIDATION & INTEGRITY CHECKS)
# ==============================================================================
print_section("1. Relational Joins (Indicators, Validation & Integrity Checks)")

customers = pd.DataFrame({
    "CustomerID": [101, 102, 103, 104, 105],
    "CustomerName": ["Alpha Corp", "Beta Labs", "Gamma LLC", "Delta Ltd", "Epsilon Inc"],
    "Country": ["USA", "Germany", "Japan", "UK", "Canada"]
})

orders = pd.DataFrame({
    "OrderID": [9001, 9002, 9003, 9004, 9005],
    "CustomerID": [101, 102, 101, 106, 103],  # Note: 106 does not exist in customers!
    "OrderTotalUSD": [450.0, 1200.0, 890.0, 310.0, 2150.0]
})

print("--- Customers Table ---")
print(customers)
print("\n--- Orders Table ---")
print(orders)

# Left Join with Merge Indicator to audit pipeline reconciliation
merged_diagnostic = pd.merge(
    customers,
    orders,
    on="CustomerID",
    how="outer",
    indicator=True
)

print("\n--- Full Outer Join with Diagnostic Indicator Column (_merge) ---")
print(merged_diagnostic)

print("\n--- Distribution of Match Statuses ---")
print(merged_diagnostic["_merge"].value_counts())

# Safe Production Join with `validate='one_to_many'`
# CustomerID is unique in customers (1), but can appear multiple times in orders (m)
safe_order_report = pd.merge(
    customers,
    orders,
    on="CustomerID",
    how="inner",
    validate="one_to_many"  # Will raise MergeError if customers table has duplicates!
)
print("\n--- Validated 1:M Inner Join (Customer Order Details) ---")
print(safe_order_report[["OrderID", "CustomerName", "Country", "OrderTotalUSD"]])


# ==============================================================================
# 2. HIGH-FREQUENCY ASYNCHRONOUS MATCHING WITH pd.merge_asof()
# ==============================================================================
print_section("2. High-Frequency Asynchronous Matching with pd.merge_asof()")

# In financial markets, trades execute against the prevailing Bid/Ask quotes.
# Quotes update continuously, while trades occur at distinct irregular timestamps.

quotes = pd.DataFrame({
    "Timestamp": pd.to_datetime([
        "2026-03-01 09:30:00.010",
        "2026-03-01 09:30:00.050",
        "2026-03-01 09:30:00.120",
        "2026-03-01 09:30:00.250",
        "2026-03-01 09:30:00.400"
    ]),
    "Symbol": ["NVDA", "NVDA", "NVDA", "NVDA", "NVDA"],
    "BidPrice": [495.10, 495.15, 495.20, 495.18, 495.25],
    "AskPrice": [495.15, 495.20, 495.25, 495.22, 495.30]
})

trades = pd.DataFrame({
    "Timestamp": pd.to_datetime([
        "2026-03-01 09:30:00.045",  # Occurs before second quote -> matches 09:30:00.010 quote
        "2026-03-01 09:30:00.130",  # Matches 09:30:00.120 quote
        "2026-03-01 09:30:00.290",  # Matches 09:30:00.250 quote
    ]),
    "Symbol": ["NVDA", "NVDA", "NVDA"],
    "TradePrice": [495.15, 495.22, 495.20],
    "Shares": [200, 500, 150]
})

print("--- Real-Time Limit Order Book (Quotes) ---")
print(quotes)
print("\n--- Executed Market Trades ---")
print(trades)

# As-Of Merge: Finds the last valid quote prior to each executed trade
asof_aligned = pd.merge_asof(
    trades,
    quotes,
    on="Timestamp",
    by="Symbol",
    direction="backward",
    tolerance=pd.Timedelta("100ms")  # Reject quotes older than 100ms
)

# Calculate execution slippage against prevailing ask price
asof_aligned["EffectiveSpread"] = (asof_aligned["TradePrice"] - asof_aligned["BidPrice"]) / (asof_aligned["AskPrice"] - asof_aligned["BidPrice"])
print("\n--- Asynchronous AsOf Aligned Trades & Quotes (with Slippage) ---")
print(asof_aligned)


# ==============================================================================
# 3. CONCATENATION ALONG AXES & PREVENTING FRAGMENTATION (pd.concat)
# ==============================================================================
print_section("3. Concatenation along Axes & Preventing Fragmentation (pd.concat)")

# Scenario: Partitioned server logs arriving from multiple geographical nodes
us_east_logs = pd.DataFrame({
    "ServerID": ["srv-e1", "srv-e2"],
    "CPU_Load": [0.65, 0.78],
    "ActiveSessions": [1420, 2150]
})

eu_central_logs = pd.DataFrame({
    "ServerID": ["srv-c1", "srv-c2"],
    "CPU_Load": [0.42, 0.88],
    "ActiveSessions": [890, 2800]
})

# Row-wise concatenation with Hierarchical MultiIndex keys
cluster_fleet = pd.concat(
    [us_east_logs, eu_central_logs],
    keys=["US-East", "EU-Central"],
    names=["Datacenter", "NodeIdx"]
)

print("--- Hierarchical Multi-Datacenter Server Fleet (Concat Axis 0) ---")
print(cluster_fleet)

# Column-wise concatenation (Axis 1) with index alignment
technical_indicators = pd.DataFrame({
    "RSI_14": [45.2, 58.1, 72.4, 61.0],
    "MACD": [0.12, 0.45, 0.88, 0.32]
}, index=pd.date_range("2026-03-01", periods=4, freq="D"))

sentiment_scores = pd.DataFrame({
    "NewsSentiment": [0.65, 0.70, 0.85, 0.40]
}, index=pd.date_range("2026-03-01", periods=4, freq="D"))

feature_matrix = pd.concat([technical_indicators, sentiment_scores], axis=1)
print("\n--- Multi-Modal Feature Matrix (Concat Axis 1) ---")
print(feature_matrix)


# ==============================================================================
# 4. TIME SERIES FOUNDATIONS: DATETIMEINDEX & TIMEZONE CONVERSIONS
# ==============================================================================
print_section("4. Time Series Foundations: DatetimeIndex & Timezone Conversions")

# Generating business calendar dates (skipping weekends)
trading_calendar = pd.bdate_range(start="2026-01-01", end="2026-01-15", name="TradingDate")
print(f"Generated {len(trading_calendar)} Business Days:")
print(trading_calendar[:5])

# Timezone Localization & Conversion
# Step 1: Create timezone-naive timestamps
naive_timestamps = pd.Series(
    data=[100, 105, 110],
    index=pd.date_range("2026-06-01 12:00:00", periods=3, freq="h", name="WallClock")
)
print("\n--- Timezone-Naive Series ---")
print(naive_timestamps)

# Step 2: Localize to UTC
utc_aware = naive_timestamps.tz_localize("UTC")
print("\n--- UTC Localized Series ---")
print(utc_aware)

# Step 3: Convert from UTC to New York and Tokyo timezones
ny_time = utc_aware.tz_convert("America/New_York")
tokyo_time = utc_aware.tz_convert("Asia/Tokyo")

tz_comparison = pd.DataFrame({
    "UTC": utc_aware.index,
    "America/New_York": ny_time.index,
    "Asia/Tokyo": tokyo_time.index
})
print("\n--- Cross-Continent Timezone Conversions ---")
print(tz_comparison)


# ==============================================================================
# 5. RESAMPLING ENGINE: DOWNSAMPLING TO FINANCIAL OHLC & UPSAMPLING
# ==============================================================================
print_section("5. Resampling Engine: Downsampling to Financial OHLC & Upsampling")

# Simulating 1-minute high-frequency price ticks
rng = np.random.default_rng(42)
minute_index = pd.date_range("2026-03-02 09:30", periods=30, freq="min")
synthetic_ticks = 150.0 + np.cumsum(rng.normal(0.0, 0.75, size=30))

tick_series = pd.Series(synthetic_ticks, index=minute_index, name="TickPrice")

# Downsampling 1-minute ticks into 5-minute OHLC (Open, High, Low, Close) bars
ohlc_5min = tick_series.resample("5min").ohlc()
ohlc_5min["Volume"] = rng.integers(1000, 10000, size=len(ohlc_5min))

print("--- Downsampled 5-Minute Financial OHLC Candlestick Bars ---")
print(ohlc_5min.round(2))

# Upsampling: Taking low-frequency daily data and projecting into hourly data
daily_macro = pd.Series([3.5, 3.8], index=pd.date_range("2026-03-01", periods=2, freq="D"))
hourly_interpolated = daily_macro.resample("4h").interpolate(method="time")
print("\n--- Upsampling Daily Macro Rate to 4-Hour Frequency via Time Interpolation ---")
print(hourly_interpolated)


# ==============================================================================
# 6. FINANCIAL ENGINEERING: SHIFTS, PERCENTAGE RETURNS & ROLLING DRAWDOWNS
# ==============================================================================
print_section("6. Financial Engineering: Shifts, Returns & Rolling Drawdowns")

# Generating 60 days of synthetic equity prices
days = 60
daily_returns_sim = rng.normal(loc=0.0008, scale=0.015, size=days)  # ~20% annualized vol
equity_curve = 1000.0 * np.cumprod(1.0 + daily_returns_sim)

portfolio = pd.DataFrame({
    "Date": pd.date_range("2026-01-01", periods=days, freq="B"),
    "PortfolioValue": equity_curve
}).set_index("Date")

# 1. Daily percentage returns and log returns
portfolio["DailyReturn"] = portfolio["PortfolioValue"].pct_change()
portfolio["LogReturn"] = np.log(portfolio["PortfolioValue"] / portfolio["PortfolioValue"].shift(1))

# 2. Cumulative return factor
portfolio["CumulativeReturn"] = (portfolio["PortfolioValue"] / portfolio["PortfolioValue"].iloc[0]) - 1.0

# 3. Peak-to-Date and Drawdown Analysis
portfolio["PeakValue"] = portfolio["PortfolioValue"].cummax()
portfolio["DrawdownPct"] = (portfolio["PortfolioValue"] - portfolio["PeakValue"]) / portfolio["PeakValue"]

# 4. Rolling 20-day Annualized Volatility (252 trading days/year)
portfolio["RollingVol20D"] = portfolio["DailyReturn"].rolling(window=20).std() * np.sqrt(252)

max_drawdown = portfolio["DrawdownPct"].min()
total_gain = portfolio["CumulativeReturn"].iloc[-1]

print("--- Financial Risk & Return Analytics (Tail 10 Days) ---")
print(portfolio[["PortfolioValue", "DailyReturn", "CumulativeReturn", "DrawdownPct", "RollingVol20D"]].tail(10).round(4))
print(f"\nPortfolio Performance Summary:")
print(f"  Final Cumulative Return: {total_gain * 100:.2f}%")
print(f"  Maximum Peak-to-Trough Drawdown: {max_drawdown * 100:.2f}%")

print(f"\nCompleted Part 4 successfully! Ready for Part 5.")
