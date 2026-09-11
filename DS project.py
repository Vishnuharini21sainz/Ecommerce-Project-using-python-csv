import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("ecommerce_100_products.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
df = df.drop_duplicates()
print(df["Category"].value_counts())
print(df.groupby("Category")["Revenue"].sum().sort_values(ascending=False))
print(df.nlargest(10, "Units_Sold"))
print(df.nlargest(10, "Rating"))
print(df["Price"].corr(df["Rating"]))
print(df["Discount_Percent"].corr(df["Units_Sold"]))
print(df.groupby("Category")["Rating"].mean())
print(df.groupby("Category")["Price"].mean())
print(df.nlargest(10, "Revenue"))
print(
    df[
        (df["Stock"] < 100) &
        (df["Units_Sold"] > df["Units_Sold"].median())
    ]
)
sns.countplot(data=df, x="Category")
plt.title("Number of Products by Category")
plt.xticks(rotation=30)
plt.show()
revenue = df.groupby("Category", as_index=False)["Revenue"].sum()
sns.barplot(data=revenue, x="Category", y="Revenue")
plt.title("Total Revenue by Category")
plt.xticks(rotation=30)
plt.show()
top10 = df.nlargest(10, "Units_Sold")
sns.barplot(data=top10, x="Units_Sold", y="Product_Name")
plt.title("Top 10 Products by Units Sold")
plt.show()
sns.histplot(data=df, x="Rating", bins=8, kde=True)
plt.title("Distribution of Product Ratings")
plt.show()
sns.scatterplot(
    data=df,
    x="Price",
    y="Rating",
    hue="Category"
)
plt.title("Price vs Product Rating")
plt.show()
sns.scatterplot(
    data=df,
    x="Discount_Percent",
    y="Units_Sold",
    hue="Category"
)
plt.title("Discount Percentage vs Units Sold")
plt.show()
rating = df.groupby("Category", as_index=False)["Rating"].mean()
sns.barplot(data=rating, x="Category", y="Rating")
plt.title("Average Product Rating by Category")
plt.ylim(0, 5)
plt.show()
price = df.groupby("Category", as_index=False)["Price"].mean()
sns.barplot(data=price, x="Category", y="Price")
plt.title("Average Product Price by Category")
plt.xticks(rotation=30)
plt.show()
top_revenue = df.nlargest(10, "Revenue")
sns.barplot(
    data=top_revenue,
    x="Revenue",
    y="Product_Name"
)
plt.title("Top 10 Products by Revenue")
plt.show()
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
