# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10ykBHpciz1eYN5kYr2Jg8BBg6Eev4zt5
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install streamlit

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_excel('data/sales.xlsx')

# Display the first few rows to understand the structure
df.head()

# Renaming columns for easier reference
df.columns = ['SalesID', 'CustomerID', 'SalesRepID', 'ProductID', 'SalesQuantity', 'SalesAmount', 'Year', 'Month', 'Day']

# 1. Descriptive Analysis: Calculate total annual sales for each product
annual_sales = df.groupby(['Year', 'ProductID'])['SalesAmount'].sum().reset_index()

# 2. Calculate average sales per customer
avg_sales_per_customer = df.groupby('CustomerID')['SalesAmount'].mean().reset_index()

# 3. Identify the top 3 sales representatives based on total sales
top_sales_reps = df.groupby('SalesRepID')['SalesAmount'].sum().nlargest(3).reset_index()

# Displaying the results

import matplotlib.pyplot as plt
import seaborn as sns

# Setting up visualizations
plt.figure(figsize=(12, 6))
sns.barplot(data=annual_sales, x='ProductID', y='SalesAmount', hue='Year')
plt.title('Total Annual Sales for Each Product')
plt.xlabel('Product ID')
plt.ylabel('Total Sales Amount (in US$)')
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# Visualization for average sales per customer
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_sales_per_customer, x='CustomerID', y='SalesAmount', palette='viridis')
plt.title('Average Sales per Customer')
plt.xlabel('Customer ID')
plt.ylabel('Average Sales Amount (in US$)')
plt.tight_layout()
plt.show()

# Visualization for top 3 sales representatives by total sales
plt.figure(figsize=(8, 5))
sns.barplot(data=top_sales_reps, x='SalesRepID', y='SalesAmount', palette='magma')
plt.title('Top 3 Sales Representatives by Total Sales')
plt.xlabel('SalesRep ID')
plt.ylabel('Total Sales Amount (in US$)')
plt.tight_layout()
plt.show()

# Re-plot with annotations
plt.figure(figsize=(12, 6))
ax1 = sns.barplot(data=annual_sales, x='ProductID', y='SalesAmount', hue='Year')
plt.title('Total Annual Sales for Each Product')
plt.xlabel('Product ID')
plt.ylabel('Total Sales Amount (in US$)')
plt.legend(title='Year')

# Add annotations
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():,.0f}",
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()

# Visualization for average sales per customer with annotations
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(data=avg_sales_per_customer, x='CustomerID', y='SalesAmount', palette='viridis')
plt.title('Average Sales per Customer')
plt.xlabel('Customer ID')
plt.ylabel('Average Sales Amount (in US$)')

# Add annotations
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():,.0f}",
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()

# Visualization for top 3 sales representatives by total sales with annotations
plt.figure(figsize=(8, 5))
ax3 = sns.barplot(data=top_sales_reps, x='SalesRepID', y='SalesAmount', palette='magma')
plt.title('Top 3 Sales Representatives by Total Sales')
plt.xlabel('SalesRep ID')
plt.ylabel('Total Sales Amount (in US$)')

# Add annotations
for p in ax3.patches:
    ax3.annotate(f"{p.get_height():,.0f}",
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()

# Analyzing monthly sales trends
monthly_sales = df.groupby(['Year', 'Month'])['SalesAmount'].sum().reset_index()

# Convert 'Month' to a categorical type with proper month order for plotting
months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_sales['Month'] = pd.Categorical(monthly_sales['Month'], categories=months_order, ordered=True)

# Sort by Year and Month for correct plotting order
monthly_sales = monthly_sales.sort_values(['Year', 'Month'])

# Plotting monthly sales trends
plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_sales, x='Month', y='SalesAmount', hue='Year', marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount (in US$)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Identifying the highest sales month for each year
highest_monthly_sales = monthly_sales.loc[monthly_sales.groupby('Year')['SalesAmount'].idxmax()]

# Plotting with highlights for the highest months
plt.figure(figsize=(14, 7))
ax = sns.lineplot(data=monthly_sales, x='Month', y='SalesAmount', hue='Year', marker='o')
plt.title('Monthly Sales Trends with Highlighted Highest Sales Months')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount (in US$)')
plt.xticks(rotation=45)

# Highlight highest monthly sales months
for _, row in highest_monthly_sales.iterrows():
    ax.annotate(f"{row['Month']}",
                xy=(row['Month'], row['SalesAmount']),
                xytext=(row['Month'], row['SalesAmount'] + 50000),
                arrowprops=dict(arrowstyle="->", color='red'),
                fontsize=12, color='red')

plt.tight_layout()
plt.show()

# Analyzing top products in each month by total sales
top_products_monthly = df.groupby(['Year', 'Month', 'ProductID'])['SalesAmount'].sum().reset_index()

# Identifying the top product for each month of each year
top_products_monthly = top_products_monthly.loc[top_products_monthly.groupby(['Year', 'Month'])['SalesAmount'].idxmax()]

# Plotting top products for each month with total sales amount
plt.figure(figsize=(14, 8))
sns.barplot(data=top_products_monthly, x='Month', y='SalesAmount', hue='ProductID', dodge=True)
plt.title('Top Products in Each Month by Total Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount (in US$)')
plt.xticks(rotation=45)
plt.legend(title='Product ID')
plt.tight_layout()
plt.show()

# Fixing the month column to be numeric for datetime conversion
month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df['Month'] = df['Month'].map(month_mapping)

# Creating a 'Quarter' column
df['Quarter'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.to_period('Q')

# Grouping sales data by quarter and product
quarterly_sales = df.groupby(['Quarter', 'ProductID'])['SalesAmount'].sum().reset_index()
# Ensure the 'Quarter' column is of string type for seaborn compatibility
quarterly_sales['Quarter'] = quarterly_sales['Quarter'].astype(str)

# Plotting product sales trends per quarter
plt.figure(figsize=(16, 8))
sns.lineplot(data=quarterly_sales, x='Quarter', y='SalesAmount', hue='ProductID', marker='o')
plt.title('Product Sales Trends per Quarter')
plt.xlabel('Quarter')
plt.ylabel('Total Sales Amount (in US$)')
plt.xticks(rotation=45)
plt.legend(title='Product ID')
plt.tight_layout()
plt.show()

# Data yang sudah disimpan di sales_df
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
df['Month'] = df['Month'].map(month_mapping)
df['Quarter'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.to_period('Q')

# Sidebar options
st.sidebar.header('Dashboard Options')
analysis_type = st.sidebar.selectbox('Choose analysis type:', ['Annual Product Sales', 'Top Sales Representatives', 'Monthly Sales Trends', 'Quarterly Product Trends'])

# Analysis: Total Annual Product Sales
if analysis_type == 'Annual Product Sales':
    annual_sales = df.groupby(['Year', 'ProductID'])['SalesAmount'].sum().reset_index()
    st.subheader('Total Annual Sales for Each Product')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=annual_sales, x='ProductID', y='SalesAmount', hue='Year', ax=ax)
    ax.set_title('Total Annual Sales for Each Product')
    ax.set_xlabel('Product ID')
    ax.set_ylabel('Total Sales Amount (in US$)')
    st.pyplot(fig)

# Analysis: Top 3 Sales Representatives
elif analysis_type == 'Top Sales Representatives':
    top_sales_reps = df.groupby('SalesRepID')['SalesAmount'].sum().nlargest(3).reset_index()
    st.subheader('Top 3 Sales Representatives by Total Sales')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_sales_reps, x='SalesRepID', y='SalesAmount', palette='magma', ax=ax)
    ax.set_title('Top 3 Sales Representatives by Total Sales')
    ax.set_xlabel('SalesRep ID')
    ax.set_ylabel('Total Sales Amount (in US$)')
    st.pyplot(fig)

# Analysis: Monthly Sales Trends
elif analysis_type == 'Monthly Sales Trends':
    monthly_sales = df.groupby(['Year', 'Month'])['SalesAmount'].sum().reset_index()
    monthly_sales['Month'] = pd.Categorical(monthly_sales['Month'], categories=range(1, 13), ordered=True)
    monthly_sales = monthly_sales.sort_values(['Year', 'Month'])
    st.subheader('Monthly Sales Trends')
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=monthly_sales, x='Month', y='SalesAmount', hue='Year', marker='o', ax=ax)
    ax.set_title('Monthly Sales Trends')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales Amount (in US$)')
    st.pyplot(fig)

# Analysis: Quarterly Product Trends
elif analysis_type == 'Quarterly Product Trends':
    quarterly_sales = df.groupby(['Quarter', 'ProductID'])['SalesAmount'].sum().reset_index()
    quarterly_sales['Quarter'] = quarterly_sales['Quarter'].astype(str)
    st.subheader('Product Sales Trends per Quarter')
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=quarterly_sales, x='Quarter', y='SalesAmount', hue='ProductID', marker='o', ax=ax)
    ax.set_title('Product Sales Trends per Quarter')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Total Sales Amount (in US$)')
    st.pyplot(fig)

# Explanation and insights
st.write("""
### Analysis Insights
- **Annual Sales**: This helps identify which products have the most revenue potential across years.
- **Top Sales Representatives**: Highlighting the top performers aids in recognizing and replicating successful strategies.
- **Monthly Trends**: Seasonal peaks and troughs indicate customer buying behavior, assisting in inventory and promotion planning.
- **Quarterly Trends**: Shows longer-term patterns for strategic adjustments and forecasts.
""")