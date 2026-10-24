"""
================================================================================
Day 27: Pandas Deep Dive - Part 2: Data Cleaning, Missing Data & Transforms
================================================================================
Data cleaning and preprocessing constitute 80% of real-world data science,
analytics engineering, and machine learning workflows. Production datasets are
fraught with missing values, malformed strings, erroneous sensor readings,
duplicated logs, and non-standard data types.

Theoretical Deep-Dive:
----------------------
1. The Evolution of Missing Values in Pandas:
   - `np.nan`: An IEEE 754 floating-point "Not a Number". Because it is a float,
     introducing even a single `np.nan` into an integer array historically forced
     the entire column to convert into float64, losing integer semantics and
     increasing memory footprint!
   - `None`: Python's native null object. When placed in a Series, it often forces
     an unoptimized `object` dtype.
   - `pd.NA`: Introduced in Pandas 1.0+ to represent missing values across all
     nullable dtypes (Int64, Float64, boolean, string) under Kleene's three-valued logic.
     In Kleene logic:
       True | pd.NA  --> True
       False & pd.NA --> False
       True & pd.NA  --> <NA>
   - `pd.NaT`: Not-a-Time, the missing value sentinel specifically for datetime/timedelta.

2. Imputation Strategies:
   - Mean / Median / Mode Imputation: Simple, but reduces variance and distorts distributions.
   - Forward / Backward Fill (.ffill() / .bfill()): Carries forward the last known valid
     observation. In production financial or sensor streaming, forward filling must always
     specify a `limit` parameter to prevent propagating stale data indefinitely.
   - Interpolation: Estimates missing values along a linear, polynomial, or time continuum.

3. Outlier Engineering:
   - Tukey's Interquartile Range (IQR) Rule:
     IQR = Q3 - Q1
     Valid Bounds = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
     Robust against skewed distributions and non-Gaussian data.
   - Z-score Rule:
     z = (x - mean) / std
     Flag outliers where |z| > 3 (assuming near-normal distribution).
   - Winsorization / Clipping (.clip()): Rather than deleting outlier rows (which loses
     other valid feature columns), outliers are capped at the percentile threshold boundaries.

Contents:
---------
1. Anatomy of Missing Data Sentinels & Three-Valued Logic
2. Detection, Dropping & Multi-Strategy Imputation (.dropna, .fillna, .interpolate)
3. Deduplication & Record Disambiguation (.duplicated, .drop_duplicates)
4. High-Performance Vectorized String Cleaning (.str Accessor & Regex Extraction)
5. Categorical Engineering & Coercive Type Conversions (to_numeric, to_datetime)
6. Robust Outlier Detection, Filtering & Winsorization (.clip)
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
# 1. ANATOMY OF MISSING DATA SENTINELS & THREE-VALUED LOGIC
# ==============================================================================
print_section("1. Anatomy of Missing Data Sentinels & Three-Valued Logic")

# Comparing np.nan vs pd.NA behavior
print(f"np.nan == np.nan: {np.nan == np.nan} (By IEEE 754 spec, NaN is never equal to itself!)")
print(f"pd.NA == pd.NA:   {pd.NA == pd.NA} (Evaluates to <NA> under 3-valued logic)")

# Nullable integer type vs classic integer conversion
classic_with_nan = pd.Series([1, 2, np.nan, 4])
nullable_int = pd.Series([1, 2, pd.NA, 4], dtype="Int64")  # Note capital 'I'

print("\n--- Classic Series with np.nan (Forces float64) ---")
print(classic_with_nan)

print("\n--- Modern Nullable Integer Series with pd.NA (Preserves Int64!) ---")
print(nullable_int)

# Kleene logic truth table demonstration
bool_s = pd.Series([True, False, pd.NA], dtype="boolean")
print("\n--- Kleene 3-Valued Logic Table ---")
print("True OR pd.NA:  ", True | pd.NA)
print("False OR pd.NA: ", False | pd.NA)
print("True AND pd.NA: ", True & pd.NA)
print("False AND pd.NA:", False & pd.NA)


# ==============================================================================
# 2. DETECTION, DROPPING & MULTI-STRATEGY IMPUTATION
# ==============================================================================
print_section("2. Detection, Dropping & Multi-Strategy Imputation")

sensor_df = pd.DataFrame({
    "Timestamp": pd.date_range("2026-03-01 08:00", periods=7, freq="min"),
    "Temperature": [21.4, 21.6, np.nan, np.nan, 22.1, 22.4, np.nan],
    "Humidity": [45.0, np.nan, 46.5, np.nan, 47.2, np.nan, 48.0],
    "Pressure": [1013.2, 1013.1, 1012.9, 1013.0, np.nan, 1013.3, 1013.5],
    "SensorStatus": ["OK", "OK", "WARN", np.nan, "OK", "OK", "OK"]
})

print("--- Raw Sensor DataFrame with Missing Observations ---")
print(sensor_df)

# Missingness audit
print("\n--- Missing Value Count & Percentage per Column ---")
missing_summary = pd.DataFrame({
    "MissingCount": sensor_df.isna().sum(),
    "MissingPct": (sensor_df.isna().mean() * 100).round(1)
})
print(missing_summary)

# Dropping Rows with thresholds
# thresh=4 requires at least 4 non-null values to keep the row
clean_dropped = sensor_df.dropna(thresh=4)
print("\n--- Dropped Rows with Fewer than 4 Valid Features (thresh=4) ---")
print(clean_dropped)

# Targeted Imputation Strategies:
# 1. Forward Fill with Limit: Carry forward last reading, but max 1 step
# 2. Linear Interpolation for continuous physical temperature
# 3. Categorical fillna with 'UNKNOWN' sentinel
imputed_df = sensor_df.copy()
imputed_df["Temperature"] = imputed_df["Temperature"].interpolate(method="linear")
imputed_df["Humidity"] = imputed_df["Humidity"].ffill(limit=1).bfill(limit=1)
imputed_df["Pressure"] = imputed_df["Pressure"].fillna(imputed_df["Pressure"].median())
imputed_df["SensorStatus"] = imputed_df["SensorStatus"].fillna("UNKNOWN")

print("\n--- Imputed Sensor DataFrame (Interpolation + Constrained Ffill + Median) ---")
print(imputed_df)


# ==============================================================================
# 3. DEDUPLICATION & RECORD DISAMBIGUATION
# ==============================================================================
print_section("3. Deduplication & Record Disambiguation")

transactions = pd.DataFrame({
    "OrderID": [501, 502, 503, 502, 504, 501],
    "Customer": ["Alice", "Bob", "Charlie", "Bob", "David", "Alice"],
    "Amount": [150.0, 75.0, 320.0, 75.0, 410.0, 150.0],
    "LogTimestamp": [
        "2026-03-01 10:00:01",
        "2026-03-01 10:05:22",
        "2026-03-01 10:12:00",
        "2026-03-01 10:05:25",  # Duplicate retry log!
        "2026-03-01 10:15:30",
        "2026-03-01 10:00:01"   # Exact duplicate row!
    ]
})

print("--- Raw Transactions Log with Duplicates ---")
print(transactions)

# Identifying duplicate rows across subset of keys
dup_mask = transactions.duplicated(subset=["OrderID"], keep=False)
print("\n--- All Instances of Duplicated OrderIDs (keep=False) ---")
print(transactions[dup_mask])

# Deduplicating while keeping the latest log entry
deduped_df = transactions.drop_duplicates(subset=["OrderID"], keep="last").reset_index(drop=True)
print("\n--- Deduplicated Orders (Keeping Last Recorded Entry) ---")
print(deduped_df)


# ==============================================================================
# 4. HIGH-PERFORMANCE VECTORIZED STRING CLEANING (.str ACCESSOR & REGEX)
# ==============================================================================
print_section("4. High-Performance Vectorized String Cleaning (.str Accessor & Regex)")

dirty_crm_data = pd.DataFrame({
    "RawContact": [
        "  JOHNATHAN DOE  <j.doe@apex-cloud.io>  ",
        "DR. SARAH CONNOR, PHD <sarah.c@cyberdyne.org> ",
        "  Alex 'Mercer' Rivera <a_rivera99@gmail.com> ",
        "Invalid Entry Without Email 999-555-1234",
        "MARIA GARCIA-LOPEZ <m.garcia@techcorp.es> "
    ],
    "Location": [
        "San Francisco, CA 94107",
        "Austin, TX 78701",
        "New York, NY 10001",
        "Unknown Location",
        "Seattle, WA 98101"
    ]
})

print("--- Dirty CRM Contact Records ---")
print(dirty_crm_data)

# 1. Regex named capture groups for parsing email addresses and clean names
contact_regex = r"(?P<Name>[^<]+)<(?P<Email>[^>]+)>"
parsed_contacts = dirty_crm_data["RawContact"].str.extract(contact_regex)

# Clean parsed names: strip whitespace and standardize title-casing
parsed_contacts["Name"] = parsed_contacts["Name"].str.strip().str.title()
parsed_contacts["Email"] = parsed_contacts["Email"].str.strip().str.lower()

# 2. Parsing Location string into City, State, and Zip code
loc_regex = r"(?P<City>[^,]+),\s*(?P<State>[A-Z]{2})\s*(?P<Zip>\d{5})"
parsed_locations = dirty_crm_data["Location"].str.extract(loc_regex)

# Combine parsed features
clean_crm = pd.concat([parsed_contacts, parsed_locations], axis=1)
print("\n--- Cleaned & Structured CRM Features via Vectorized Regex ---")
print(clean_crm)

# String containment and pattern matching
corporate_emails = clean_crm[clean_crm["Email"].str.contains(r"\.(io|org|es)$", regex=True, na=False)]
print("\n--- Corporate / Non-Commercial Emails (.io, .org, .es) ---")
print(corporate_emails[["Name", "Email"]])


# ==============================================================================
# 5. CATEGORICAL ENGINEERING & COERCIVE TYPE CONVERSIONS
# ==============================================================================
print_section("5. Categorical Engineering & Coercive Type Conversions")

raw_invoices = pd.DataFrame({
    "InvoiceNo": ["INV-001", "INV-002", "INV-003", "INV-004", "INV-005"],
    "GrossRevenue": ["$1,450.50", "$3,200.00", "CORRUPTED_ENTRY", "$890.25", "$4,120.80"],
    "PriorityLevel": ["High", "Low", "Critical", "Medium", "High"],
    "IssuedDate": ["2026/01/15", "15-Feb-2026", "2026-03-01", "INVALID_DATE", "2026-04-10"]
})

print("--- Raw Invoices with Dirty String Types ---")
print(raw_invoices)

# 1. Clean monetary strings and coerce to numeric
# pd.to_numeric with errors='coerce' replaces unparseable strings with np.nan
clean_revenue = (
    raw_invoices["GrossRevenue"]
    .str.replace(r"[\$,]", "", regex=True)
)
raw_invoices["GrossRevenue"] = pd.to_numeric(clean_revenue, errors="coerce")

# 2. Coercive Datetime Parsing
raw_invoices["IssuedDate"] = pd.to_datetime(raw_invoices["IssuedDate"], format="mixed", errors="coerce")

# 3. Ordered Categorical Encoding
# Allows logical comparisons: PriorityLevel > 'Medium'
priority_order = ["Low", "Medium", "High", "Critical"]
raw_invoices["PriorityLevel"] = pd.Categorical(
    raw_invoices["PriorityLevel"],
    categories=priority_order,
    ordered=True
)

print("\n--- Cleaned Invoices with Robust Types ---")
print(raw_invoices)
print("\nData Types:")
print(raw_invoices.dtypes)

# Querying with ordered categorical
urgent_invoices = raw_invoices[raw_invoices["PriorityLevel"] >= "High"]
print("\n--- Filtered: PriorityLevel >= 'High' (Leveraging Ordered Category) ---")
print(urgent_invoices)


# ==============================================================================
# 6. ROBUST OUTLIER DETECTION, FILTERING & WINSORIZATION (.clip)
# ==============================================================================
print_section("6. Robust Outlier Detection, Filtering & Winsorization (.clip)")

# Simulating server latency measurements in milliseconds (with extreme anomalous spikes)
rng = np.random.default_rng(42)
normal_latencies = rng.normal(loc=45.0, scale=8.0, size=1000)
anomalous_spikes = np.array([250.0, 480.0, 920.0, 5.0, 2.0])  # Network timeouts / glitches
latency_series = pd.Series(np.concatenate([normal_latencies, anomalous_spikes]), name="LatencyMs")

print(f"Total Latency Observations: {len(latency_series)}")
print(f"Mean: {latency_series.mean():.2f} ms, Std: {latency_series.std():.2f} ms")
print(f"Min:  {latency_series.min():.2f} ms, Max: {latency_series.max():.2f} ms")

# Method A: Tukey's Interquartile Range (IQR) Rule
q25 = latency_series.quantile(0.25)
q75 = latency_series.quantile(0.75)
iqr = q75 - q25
lower_bound_iqr = q25 - 1.5 * iqr
upper_bound_iqr = q75 + 1.5 * iqr

outliers_iqr = latency_series[(latency_series < lower_bound_iqr) | (latency_series > upper_bound_iqr)]
print(f"\n--- Tukey's IQR Method ---")
print(f"Q1 (25%): {q25:.2f} ms, Q3 (75%): {q75:.2f} ms, IQR: {iqr:.2f} ms")
print(f"Valid Acceptance Range: [{lower_bound_iqr:.2f} ms, {upper_bound_iqr:.2f} ms]")
print(f"Detected {len(outliers_iqr)} Outliers:")
print(outliers_iqr.values)

# Method B: Winsorization (Clipping)
# Instead of discarding records, clamp extremes to the 1st and 99th percentiles
p01 = latency_series.quantile(0.01)
p99 = latency_series.quantile(0.99)

winsorized_latencies = latency_series.clip(lower=p01, upper=p99)
print(f"\n--- Winsorization via .clip(lower=P01, upper=P99) ---")
print(f"Lower Bound (P01): {p01:.2f} ms | Upper Bound (P99): {p99:.2f} ms")
print(f"Original Max:   {latency_series.max():.2f} ms")
print(f"Winsorized Max: {winsorized_latencies.max():.2f} ms")
print(f"Post-clip Mean: {winsorized_latencies.mean():.2f} ms, Post-clip Std: {winsorized_latencies.std():.2f} ms")

print(f"\nCompleted Part 2 successfully! Ready for Part 3.")
