import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ecommerce_100_products.csv")

# Basic information
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

# Remove duplicates
df = df.drop_duplicates()

# Q1
print(df["Category"].value_counts())

# Q2
print(df.groupby("Category")["Revenue"].sum().sort_values(ascending=False))

# Q3
print(df.nlargest(10, "Units_Sold"))

# Q4
print(df.nlargest(10, "Rating"))

# Q5
print(df["Price"].corr(df["Rating"]))

# Q6
print(df["Discount_Percent"].corr(df["Units_Sold"]))

# Q7
print(df.groupby("Category")["Rating"].mean())

# Q8
print(df.groupby("Category")["Price"].mean())

# Q9
print(df.nlargest(10, "Revenue"))

# Q10
print(
    df[
        (df["Stock"] < 100) &
        (df["Units_Sold"] > df["Units_Sold"].median())
    ]
)

# Graph 1
sns.countplot(data=df, x="Category")
plt.title("Number of Products by Category")
plt.xticks(rotation=30)
plt.show()

# Graph 2
revenue = df.groupby("Category", as_index=False)["Revenue"].sum()
sns.barplot(data=revenue, x="Category", y="Revenue")
plt.title("Total Revenue by Category")
plt.xticks(rotation=30)
plt.show()

# Graph 3
top10 = df.nlargest(10, "Units_Sold")
sns.barplot(data=top10, x="Units_Sold", y="Product_Name")
plt.title("Top 10 Products by Units Sold")
plt.show()

# Graph 4
sns.histplot(data=df, x="Rating", bins=8, kde=True)
plt.title("Distribution of Product Ratings")
plt.show()

# Graph 5
sns.scatterplot(
    data=df,
    x="Price",
    y="Rating",
    hue="Category"
)
plt.title("Price vs Product Rating")
plt.show()

# Graph 6
sns.scatterplot(
    data=df,
    x="Discount_Percent",
    y="Units_Sold",
    hue="Category"
)
plt.title("Discount Percentage vs Units Sold")
plt.show()

# Graph 7
rating = df.groupby("Category", as_index=False)["Rating"].mean()
sns.barplot(data=rating, x="Category", y="Rating")
plt.title("Average Product Rating by Category")
plt.ylim(0, 5)
plt.show()

# Graph 8
price = df.groupby("Category", as_index=False)["Price"].mean()
sns.barplot(data=price, x="Category", y="Price")
plt.title("Average Product Price by Category")
plt.xticks(rotation=30)
plt.show()

# Graph 9
top_revenue = df.nlargest(10, "Revenue")
sns.barplot(
    data=top_revenue,
    x="Revenue",
    y="Product_Name"
)
plt.title("Top 10 Products by Revenue")
plt.show()

# Graph 10
numeric_cols = [
    "Price",
    "Discount_Percent",
    "Final_Price",
    "Rating",
    "Reviews",
    "Units_Sold",
    "Revenue",
    "Stock"
]

corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()