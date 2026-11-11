
# ================================================================
# OASIS INFOBYTE - DATA ANALYTICS INTERNSHIP
# LEVEL 1 - TASK 1
# EXPLORATORY DATA ANALYSIS ON RETAIL SALES DATA
# ================================================================

# -------------------- 1. IMPORT LIBRARIES ------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# -------------------- 2. PROJECT CONFIGURATION -------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "retail_sales_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

OUTPUT_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)

CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"

CHART_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 120

print("=" * 65)
print("       OASIS INFOBYTE - DATA ANALYTICS INTERNSHIP")
print("             LEVEL 1 - TASK 1")
print("       EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# -------------------- 3. LOAD DATASET ----------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please keep the CSV file in the project folder."
    )

df = pd.read_csv(DATA_PATH, sep="\t")

print("\nDATASET LOADED SUCCESSFULLY!")
print("\nFirst Five Records:")
print(df.head().to_string(index=False))

# -------------------- 4. DATASET INSPECTION ----------------------

print("\n" + "=" * 65)
print("                    DATASET INSPECTION")
print("=" * 65)

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nDataset Information:")
df.info()

# -------------------- 5. DATA CLEANING ---------------------------

print("\n" + "=" * 65)
print("                     DATA CLEANING")
print("=" * 65)

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove duplicate records
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates().copy()

# Check required columns
required_columns = [
    "Date",
    "Gender",
    "Age",
    "Product Category",
    "Quantity",
    "Price per Unit",
    "Total Amount"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

# Convert date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert numerical columns
numeric_columns = [
    "Age",
    "Quantity",
    "Price per Unit",
    "Total Amount"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing essential values
df = df.dropna(subset=required_columns).copy()

# Validate numeric values
df = df[
    (df["Age"] >= 0) &
    (df["Quantity"] > 0) &
    (df["Price per Unit"] >= 0) &
    (df["Total Amount"] >= 0)
].copy()

print("Duplicate rows removed:", duplicates_before)
print("Cleaned dataset shape:", df.shape)
print("Data cleaning completed successfully!")

# -------------------- 6. FEATURE ENGINEERING --------------------

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month Name"] = df["Date"].dt.strftime("%b")
df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)

age_bins = [0, 18, 25, 35, 45, 55, 65, float("inf")]

age_labels = [
    "Below 18",
    "18-25",
    "26-35",
    "36-45",
    "46-55",
    "56-65",
    "Above 65"
]

df["Age Group"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)

# -------------------- 7. DESCRIPTIVE STATISTICS ------------------

print("\n" + "=" * 65)
print("                  DESCRIPTIVE STATISTICS")
print("=" * 65)

statistics = df[numeric_columns].agg(
    ["mean", "median", "std", "min", "max"]
).T

statistics.columns = [
    "Mean",
    "Median",
    "Standard Deviation",
    "Minimum",
    "Maximum"
]

print(statistics.round(2))

statistics.to_csv(
    REPORT_DIR / "descriptive_statistics.csv"
)

# -------------------- 8. MONTHLY SALES ANALYSIS ------------------

monthly_sales = (
    df.groupby(df["Date"].dt.to_period("M"))["Total Amount"]
    .sum()
)

monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(figsize=(14, 6))

sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values,
    marker="o",
    linewidth=2.5
)

plt.title(
    "Monthly Retail Sales Trend",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(CHART_DIR / "monthly_sales_trend.png")
plt.show()
plt.close()

# -------------------- 9. QUARTERLY SALES ANALYSIS ----------------

quarterly_sales = (
    df.groupby("Quarter")["Total Amount"]
    .sum()
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=quarterly_sales.index,
    y=quarterly_sales.values
)

plt.title(
    "Quarterly Retail Sales Performance",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Quarter")
plt.ylabel("Total Revenue")
plt.tight_layout()

plt.savefig(CHART_DIR / "quarterly_sales_trend.png")
plt.show()
plt.close()

# -------------------- 10. CUSTOMER AGE ANALYSIS ------------------

age_distribution = (
    df["Age Group"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=age_distribution.index.astype(str),
    y=age_distribution.values
)

plt.title(
    "Customer Distribution by Age Group",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Age Group")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(CHART_DIR / "customer_age_distribution.png")
plt.show()
plt.close()

# -------------------- 11. GENDER ANALYSIS ------------------------

gender_count = df["Gender"].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(
    gender_count.values,
    labels=gender_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Customer Gender Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(CHART_DIR / "gender_distribution.png")
plt.show()
plt.close()

# -------------------- 12. TOP 10 PRODUCTS/CATEGORIES -------------

product_column = (
    "Product"
    if "Product" in df.columns
    else "Product Category"
)

top_products = (
    df.groupby(product_column)["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title(
    "Top 10 Best-Selling Products / Categories",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Quantity Sold")
plt.ylabel("Product / Category")
plt.tight_layout()

plt.savefig(CHART_DIR / "top_10_products.png")
plt.show()
plt.close()

# -------------------- 13. CATEGORY REVENUE ANALYSIS --------------

category_revenue = (
    df.groupby("Product Category")["Total Amount"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 6))

sns.barplot(
    x=category_revenue.index,
    y=category_revenue.values
)

plt.title(
    "Revenue by Product Category",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(CHART_DIR / "category_revenue.png")
plt.show()
plt.close()

# -------------------- 14. CORRELATION ANALYSIS -------------------

correlation = df.select_dtypes(
    include=np.number
).corr()

plt.figure(figsize=(11, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    "Correlation Between Numerical Variables",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(CHART_DIR / "correlation_heatmap.png")
plt.show()
plt.close()

# -------------------- 15. AGE GROUP REVENUE ANALYSIS -------------

age_revenue = (
    df.groupby("Age Group", observed=False)["Total Amount"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=age_revenue,
    x="Age Group",
    y="Total Amount"
)

plt.title(
    "Revenue Contribution by Customer Age Group",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Age Group")
plt.ylabel("Total Revenue")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(CHART_DIR / "age_group_revenue.png")
plt.show()
plt.close()

# -------------------- 16. BUSINESS INSIGHTS ---------------------

total_revenue = df["Total Amount"].sum()
total_transactions = len(df)
average_transaction = df["Total Amount"].mean()

best_category = category_revenue.idxmax()
best_product = top_products.index[0]
best_month = monthly_sales.idxmax()
best_age_group = age_distribution.idxmax()
best_gender = gender_count.idxmax()

print("\n" + "=" * 65)
print("                 RETAIL SALES BUSINESS INSIGHTS")
print("=" * 65)

print(f"\nTotal Revenue: {total_revenue:,.2f}")
print(f"Total Transactions: {total_transactions:,}")
print(f"Average Transaction Value: {average_transaction:,.2f}")

print("\n1. BEST REVENUE CATEGORY")
print(f"{best_category}: {category_revenue.max():,.2f}")

print("\n2. BEST-SELLING PRODUCT / CATEGORY")
print(f"{best_product}: {top_products.max()} units")

print("\n3. HIGHEST SALES MONTH")
print(best_month.strftime("%B %Y"))

print("\n4. LARGEST CUSTOMER AGE GROUP")
print(best_age_group)

print("\n5. GENDER WITH MOST TRANSACTIONS")
print(best_gender)

# -------------------- 17. BUSINESS RECOMMENDATIONS --------------

recommendations = [
    (
        f"Focus marketing campaigns on {best_category}, "
        "the highest-revenue product category."
    ),
    (
        f"Plan inventory carefully for {best_product}, "
        "which recorded the highest quantity sold."
    ),
    (
        f"Study the purchasing behaviour of the {best_age_group} "
        "age group to develop targeted customer campaigns."
    ),
    (
        f"Investigate the factors contributing to higher sales "
        f"during {best_month.strftime('%B %Y')}."
    )
]

print("\n" + "=" * 65)
print("              BUSINESS RECOMMENDATIONS")
print("=" * 65)

for i, recommendation in enumerate(recommendations, 1):
    print(f"\n{i}. {recommendation}")

# Save recommendations
recommendation_df = pd.DataFrame({
    "Recommendation": recommendations
})

recommendation_df.to_csv(
    REPORT_DIR / "business_recommendations.csv",
    index=False
)

# -------------------- 18. SAVE CLEANED DATA ----------------------

df.to_csv(
    OUTPUT_DIR / "cleaned_retail_sales.csv",
    index=False
)

# Save category analysis
category_revenue.to_csv(
    REPORT_DIR / "category_revenue.csv"
)

# Save monthly sales
monthly_sales.to_csv(
    REPORT_DIR / "monthly_sales.csv"
)

# Save age analysis
age_revenue.to_csv(
    REPORT_DIR / "age_group_revenue.csv",
    index=False
)

# -------------------- 19. FINAL SUMMARY -------------------------

print("\n" + "=" * 65)
print("                  PROJECT COMPLETED!")
print("=" * 65)

print("\nGenerated Files:")
print("\nCharts:", CHART_DIR)
print("Reports:", REPORT_DIR)
print("Cleaned Dataset:", OUTPUT_DIR / "cleaned_retail_sales.csv")

print("\nTotal Charts Generated: 8")
print("Data Cleaning: Completed")
print("Statistical Analysis: Completed")
print("Business Insights: Generated")
print("Recommendations: Generated")

print("\nOASIS INFOBYTE - LEVEL 1 - TASK 1")
print("Exploratory Data Analysis Completed Successfully!")

print("=" * 65)
