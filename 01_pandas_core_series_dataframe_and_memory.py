"""
================================================================================
Day 27: Pandas Deep Dive - Part 1: Core Series, DataFrame & Memory Architecture
================================================================================
Pandas (Python Data Analysis Library) is the foundational open-source data
manipulation and analysis engine for the Python ecosystem. Built directly on top
of NumPy, it introduces labeled, relational, and heterogeneous data structures:
`Series` (1-dimensional) and `DataFrame` (2-dimensional).

Theoretical Deep-Dive:
----------------------
1. The Internal Architecture: BlockManager vs. ArrayManager vs. PyArrow:
   Historically, Pandas stores DataFrame columns using the `BlockManager`.
   - The BlockManager consolidates columns of the same NumPy dtype into
     shared 2D NumPy arrays (blocks).
     For example, three float64 columns are stored as one 2D float64 ndarray.
   - While efficient for certain 2D operations, BlockManager operations
     (like inserting or reordering columns) frequently trigger full memory
     copies or consolidation cascades.
   - Pandas 2.0+ introduces native Apache Arrow integration (`dtype="string[pyarrow]"`
     or Arrow-backed engines) and the experimental `ArrayManager`, storing
     each column as an independent 1D array.

2. Series & DataFrame Internals:
   - `Series`: A 1D labeled array consisting of a NumPy/Arrow data buffer and
     an `Index` object mapping labels to row positions (via an internal C hash table).
   - `DataFrame`: A 2D tabular structure mapping column labels (Index) to 1D Series,
     all sharing a common row Index.
   - `Index`: An immutable array-like object supporting O(1) label lookup, set
     operations (union, intersection), and hierarchical navigation (MultiIndex).

3. Views vs. Copies & Copy-on-Write (CoW):
   - Traditional Pandas often left users confused about whether an operation
     returned a view (pointing to existing memory) or a copy (new allocation).
   - The infamous `SettingWithCopyWarning` occurs when attempting "chained indexing":
     `df['col'][0] = 99`  <-- Chained access! Python evaluates df['col'] first,
     which may return a temporary copy, discarding your assignment!
   - Best practice: Always use `.loc[row_indexer, col_indexer]` for deterministic assignment.
   - Pandas 2.0+ features Copy-on-Write (`pd.options.mode.copy_on_write = True`),
     guaranteeing that modifications never mutate unrelated objects without an explicit copy.

4. Deep Memory Optimization:
   - Python objects (`dtype='object'`) in Pandas store arrays of 64-bit pointers
     pointing to scattered PyObjects. This leads to 5x-10x memory bloat and CPU cache misses.
   - Categorical dtypes (`pd.Categorical`) encode strings into integer codes
     with a small dictionary of unique categories, often reducing memory footprint by 80-95%.
   - Downcasting integers (int64 -> int8/int16) and floats (float64 -> float32)
     dramatically slashes RAM usage for large datasets.

Contents:
---------
1. Series and DataFrame Construction from Diverse Sources
2. The Index Hierarchy: RangeIndex, Index, DatetimeIndex & MultiIndex
3. High-Speed Indexing & Selection: .loc, .iloc, .at, .iat, and Callable Lambdas
4. Boolean Masking, Compound Logic & SQL-Style `.query()` Engine
5. The SettingWithCopyWarning & Copy-on-Write (CoW) Deep Dive
6. Deep Memory Profiling & Production-Grade Downcasting Engine
================================================================================
"""

import sys
import time
import numpy as np
import pandas as pd


def print_section(title: str) -> None:
    """Helper utility for formatted console section banners."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. SERIES AND DATAFRAME CONSTRUCTION FROM DIVERSE SOURCES
# ==============================================================================
print_section("1. Series and DataFrame Construction from Diverse Sources")

# A Series combines raw values with a labeled index
temperatures = pd.Series(
    data=[28.5, 32.1, 24.8, 19.4, 26.7],
    index=["Tokyo", "Dubai", "London", "Reykjavik", "Sydney"],
    name="AvgTemperatureC",
    dtype=np.float64
)

print("--- Pandas Series with Custom Index ---")
print(temperatures)
print(f"Index Type:    {type(temperatures.index).__name__}")
print(f"Data Values:   {temperatures.values} (type: {type(temperatures.values).__name__})")
print(f"Shape:         {temperatures.shape}, Size: {temperatures.size}")

# Constructing DataFrames from various Python data structures
# Method A: Dict of lists / arrays (column-oriented)
data_dict = {
    "TransactionID": [1001, 1002, 1003, 1004],
    "Customer": ["Alice", "Bob", "Charlie", "David"],
    "AmountUSD": [249.99, 89.50, 1250.00, 45.20],
    "IsFraud": [False, False, True, False]
}
df_from_dict = pd.DataFrame(data_dict)

# Method B: List of dicts / records (row-oriented, common in JSON APIs)
records = [
    {"Symbol": "AAPL", "Price": 182.52, "Volume": 48_210_000},
    {"Symbol": "GOOGL", "Price": 141.80, "Volume": 22_450_000},
    {"Symbol": "NVDA", "Price": 495.22, "Volume": 51_330_000},
]
df_from_records = pd.DataFrame.from_records(records, index="Symbol")

# Method C: 2D NumPy array with specified row/column labels
matrix_data = np.random.default_rng(42).standard_normal(size=(4, 3))
df_from_numpy = pd.DataFrame(
    data=matrix_data,
    index=pd.date_range("2026-01-01", periods=4, freq="D", name="Date"),
    columns=["Metric_Alpha", "Metric_Beta", "Metric_Gamma"]
)

print("\n--- DataFrame from Records (Stock Quotes) ---")
print(df_from_records)

print("\n--- DataFrame from 2D NumPy Array with DatetimeIndex ---")
print(df_from_numpy)


# ==============================================================================
# 2. THE INDEX HIERARCHY: RANGEINDEX, INDEX, DATETIMEINDEX & MULTIINDEX
# ==============================================================================
print_section("2. The Index Hierarchy: RangeIndex, Index, DatetimeIndex & MultiIndex")

# RangeIndex is zero-memory overhead representation of range(start, stop, step)
standard_df = pd.DataFrame({"Value": range(1_000_000)})
print(f"Standard Index Type: {type(standard_df.index).__name__}")
print(f"RangeIndex Memory:   {standard_df.index.memory_usage()} bytes (virtually zero!)")

# Hierarchical MultiIndex: Modeling multi-dimensional data in 2D tables
arrays = [
    ["North America", "North America", "Europe", "Europe", "Asia", "Asia"],
    ["USA", "Canada", "Germany", "UK", "Japan", "Singapore"],
]
index_tuples = list(zip(*arrays))
multi_idx = pd.MultiIndex.from_tuples(index_tuples, names=["Continent", "Country"])

regional_sales = pd.DataFrame(
    data={
        "Revenue_M": [845.2, 120.4, 430.1, 310.5, 520.8, 185.3],
        "EBITDA_Margin": [0.28, 0.22, 0.25, 0.20, 0.31, 0.35]
    },
    index=multi_idx
)

print("\n--- MultiIndex DataFrame (Continent -> Country) ---")
print(regional_sales)

# Cross-section extraction with .xs()
print("\n--- Cross-Section (.xs) for Europe ---")
print(regional_sales.xs(key="Europe", level="Continent"))

# Unstacking level 1 into columns
print("\n--- Unstacking Level 'Country' to Columns ---")
print(regional_sales.unstack(level="Country")["Revenue_M"])


# ==============================================================================
# 3. HIGH-SPEED INDEXING & SELECTION: .loc, .iloc, .at, .iat, AND CALLABLES
# ==============================================================================
print_section("3. High-Speed Indexing: .loc, .iloc, .at, .iat, and Callable Lambdas")

# Sample employee dataset
employees = pd.DataFrame({
    "EmployeeID": [101, 102, 103, 104, 105],
    "Name": ["Elena Rostova", "Marcus Vance", "Sara Chen", "Jamal Davis", "Priya Nair"],
    "Department": ["Engineering", "Product", "Engineering", "Marketing", "Data Science"],
    "Salary": [145_000, 130_000, 160_000, 95_000, 150_000],
    "ExperienceYears": [6, 8, 10, 3, 7]
}).set_index("EmployeeID")

print("--- Employees DataFrame ---")
print(employees)

# 1. Label-based selection (.loc) - includes both start AND end labels
print("\n--- .loc[102:104, ['Name', 'Salary']] (Label Slice: Inclusive of End) ---")
print(employees.loc[102:104, ["Name", "Salary"]])

# 2. Integer position-based selection (.iloc) - standard Python 0-indexed [start:stop)
print("\n--- .iloc[0:3, 0:2] (Position Slice: Half-open [0, 3)) ---")
print(employees.iloc[0:3, 0:2])

# 3. High-Performance Scalar Access: .at[] vs .loc[]
# .at[] bypasses overhead and directly accesses the single scalar value
t0 = time.perf_counter()
for _ in range(50_000):
    val = employees.loc[103, "Salary"]
loc_time = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(50_000):
    val = employees.at[103, "Salary"]
at_time = time.perf_counter() - t0

print(f"\nScalar access benchmark (50,000 lookups):")
print(f"  .loc[] time: {loc_time * 1000:.2f} ms")
print(f"  .at[]  time: {at_time * 1000:.2f} ms")
print(f"  Speedup Factor: {loc_time / at_time:.1f}x faster scalar access with .at[]!")

# 4. Callable Indexing (Using Lambdas inside .loc)
# Indispensable when chaining multiple method calls!
high_earning_engineers = (
    employees
    .loc[lambda df: (df["Department"] == "Engineering") & (df["Salary"] >= 150_000)]
)
print("\n--- Callable Lambda Indexing (Engineering & Salary >= 150k) ---")
print(high_earning_engineers)


# ==============================================================================
# 4. BOOLEAN MASKING, COMPOUND LOGIC & SQL-STYLE .query() ENGINE
# ==============================================================================
print_section("4. Boolean Masking, Compound Logic & SQL-Style .query() Engine")

# Bitwise operators in Pandas: & (AND), | (OR), ~ (NOT)
# Note: Parentheses are MANDATORY due to Python operator precedence rules!
mask = (employees["Salary"] > 120_000) & (employees["ExperienceYears"] >= 7)
print("--- Boolean Mask Result: (Salary > 120k & Experience >= 7) ---")
print(employees[mask])

# The .isin() method for membership testing
tech_roles = employees[employees["Department"].isin(["Engineering", "Data Science"])]
print("\n--- Using .isin(['Engineering', 'Data Science']) ---")
print(tech_roles[["Name", "Department", "Salary"]])

# The .query() expression parser (evaluated in C/numexpr)
min_salary_threshold = 135_000
target_dept = "Engineering"

# Referencing local variables using @ symbol inside the expression string
queried_df = employees.query("Department == @target_dept and Salary >= @min_salary_threshold")
print(f"\n--- .query('Department == @target_dept and Salary >= @min_salary_threshold') ---")
print(queried_df)


# ==============================================================================
# 5. THE SETTINGWITHCOPYWARNING & COPY-ON-WRITE (CoW) DEEP DIVE
# ==============================================================================
print_section("5. The SettingWithCopyWarning & Copy-on-Write (CoW) Deep Dive")

sample_df = pd.DataFrame({
    "Asset": ["Gold", "Silver", "Crude Oil", "Copper"],
    "InventoryTons": [120, 450, 8900, 620]
})

print("Original Asset DataFrame:")
print(sample_df)

# What causes SettingWithCopyWarning?
# Chained indexing: df[subset][col] = value
subset_df = sample_df[sample_df["InventoryTons"] > 500]

print("\n--- Demonstrating Safe In-Place Modification vs Chained Indexing ---")
# BAD: Chained assignment (May warn or fail to modify parent)
# subset_df['InventoryTons'] = 9000  # <--- SettingWithCopyWarning!

# GOOD 1: Explicit copy when you genuinely want an independent DataFrame
independent_copy = sample_df[sample_df["InventoryTons"] > 500].copy()
independent_copy.loc[:, "InventoryTons"] = 9999
print("Independent Copy modified cleanly without touching parent:")
print(independent_copy)

# GOOD 2: Single-step .loc on the original DataFrame to modify parent
sample_df.loc[sample_df["InventoryTons"] > 500, "InventoryTons"] += 100
print("\nOriginal DataFrame modified deterministically via .loc:")
print(sample_df)

# Copy-On-Write (CoW) in Modern Pandas:
# With CoW enabled, Pandas delays copying until an object is actually modified,
# preventing unintended side-effects and optimizing memory throughput.
print("\nCopy-on-Write Option Status:")
print(f"pd.options.mode.copy_on_write: {pd.options.mode.copy_on_write}")


# ==============================================================================
# 6. DEEP MEMORY PROFILING & PRODUCTION-GRADE DOWNCASTING ENGINE
# ==============================================================================
print_section("6. Deep Memory Profiling & Production-Grade Downcasting Engine")

def generate_bloated_dataset(n_rows: int = 200_000) -> pd.DataFrame:
    """Generates a representative large dataset with unoptimized dtypes."""
    rng = np.random.default_rng(42)
    categories = ["Automotive", "Electronics", "Groceries", "Apparel", "Home & Garden"]
    payment_types = ["Credit Card", "PayPal", "Debit Card", "Crypto"]

    return pd.DataFrame({
        "TransactionID": np.arange(1, n_rows + 1, dtype=np.int64),  # Standard int64
        "Quantity": rng.integers(1, 20, size=n_rows, dtype=np.int64),  # Fits in int8!
        "UnitPrice": rng.uniform(5.0, 500.0, size=n_rows),  # Standard float64 -> float32
        "Category": rng.choice(categories, size=n_rows),  # Python strings -> Category
        "PaymentMethod": rng.choice(payment_types, size=n_rows),  # Low-cardinality -> Category
        "IsReturned": rng.integers(0, 2, size=n_rows, dtype=np.int64)  # Fits in bool/int8
    })


def optimize_dataframe_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Production-grade automated memory optimizer:
    - Downcasts integers to smallest lossless int subtype (int8, int16, int32).
    - Downcasts floats from float64 to float32.
    - Converts object columns with low cardinality (<50% unique) to Categoricals.
    - Utilizes deep memory inspection for precise byte calculations.
    """
    start_mem_bytes = df.memory_usage(deep=True).sum()
    df_opt = df.copy()

    for col in df_opt.columns:
        col_type = df_opt[col].dtype

        # Integer Downcasting
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df_opt[col].min()
            c_max = df_opt[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df_opt[col] = df_opt[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df_opt[col] = df_opt[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df_opt[col] = df_opt[col].astype(np.int32)

        # Float Downcasting
        elif pd.api.types.is_float_dtype(col_type):
            df_opt[col] = df_opt[col].astype(np.float32)

        # Object / String column optimization
        elif col_type == object:
            num_unique = df_opt[col].nunique()
            num_total = len(df_opt[col])
            # If unique values are fewer than 50% of total rows, convert to category
            if num_unique / num_total < 0.50:
                df_opt[col] = df_opt[col].astype("category")

    end_mem_bytes = df_opt.memory_usage(deep=True).sum()
    savings_pct = 100 * (start_mem_bytes - end_mem_bytes) / start_mem_bytes

    if verbose:
        start_mb = start_mem_bytes / (1024 * 1024)
        end_mb = end_mem_bytes / (1024 * 1024)
        print(f"Memory Footprint Before: {start_mb:.2f} MB")
        print(f"Memory Footprint After:  {end_mb:.2f} MB")
        print(f"Total Memory Slashed:    {savings_pct:.1f}% reduction!")

    return df_opt


raw_df = generate_bloated_dataset(n_rows=250_000)
print(f"Generated Raw DataFrame with {len(raw_df):,} rows.")
print("\n--- Original Data Types & Memory Usage ---")
print(raw_df.dtypes)

print("\n--- Running Optimization Engine ---")
optimized_df = optimize_dataframe_memory(raw_df, verbose=True)

print("\n--- Optimized Data Types ---")
print(optimized_df.dtypes)
print(f"\nCompleted Part 1 successfully! Ready for Part 2.")
