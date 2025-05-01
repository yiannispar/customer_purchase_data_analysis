###############
# script to analyze retail data
# author: I. Paraskevas
# created: April 2025
###############

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

df = pd.read_csv('retail_data.csv')
os.makedirs("plots", exist_ok=True)

## Q1
print(f"Number of entries in dataset: {len(df)}")
num_unique_customers = df['CustomerID'].nunique()
print(f"Number of unique customers: {num_unique_customers}")


## Q2
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) # convert to datetime object
min_date = df['InvoiceDate'].min()
max_date = df['InvoiceDate'].max()
print(f"Date range: {min_date} to {max_date}")


## Q3
descriptions_to_remove = ['lost', 'damage', 'wet']
condition = df['Description'].str.contains('|'.join(descriptions_to_remove), case=False)
df = df[~condition]

# make sure these entries have been removed
unique_descriptions = df['Description'].unique()
for desc in unique_descriptions:
    if any(phrase.lower() in desc.lower() for phrase in descriptions_to_remove):
        print(f"{desc} found but should have been removed!")
 
rows_removed = condition.sum()
print(f"Number of rows removed: {rows_removed}")    

# InvoiceNo and missing CustomerID
invoice_counts = df.groupby(df['CustomerID'].isna())['InvoiceNo'].nunique()
print(f"Unique invoices WITH CustomerID: {invoice_counts.get(False, 0)}")
print(f"Unique invoices WITHOUT CustomerID: {invoice_counts.get(True, 0)}")
total_invoices = invoice_counts.sum()
print(f"Percentage of invoices without CustomerID: {invoice_counts.get(True, 0)/total_invoices:.1%}")
        
## Q4/Q5
df['PurchaseDate'] = df['InvoiceDate'].dt.date # date only (w/o time)
df_temp1 = df.groupby('CustomerID')['PurchaseDate'].nunique().reset_index(name='Purchase_days_total')

df['Purchase_amount_total'] = df['Quantity'] * df['UnitPrice']
df_temp2 = df.groupby('CustomerID')['Purchase_amount_total'].sum().reset_index(name='Purchase_amount_total')

df_temp3 = df.groupby('CustomerID')['Purchase_amount_total'].mean().reset_index(name='Purchase_amount_avg')

reference_date = pd.to_datetime('2012-01-01')
df_temp4 = df.groupby('CustomerID')['InvoiceDate'].max()  # Get last purchase date per customer
df_temp4 = (reference_date - df_temp4).dt.days.reset_index(name='Recency')

df_temp5 = df.groupby('CustomerID')['Country'].first().reset_index()

# merge all tables
df_merged_temp = df_temp1.merge(df_temp2, on='CustomerID', how='outer').merge(df_temp3, on='CustomerID', how='outer').merge(df_temp4, on='CustomerID', how='outer').merge(df_temp5, on='CustomerID', how='outer')

# ensure that tables match in lengths 
rows_match = len(df_temp1) == len(df_temp2) == len(df_temp3) == len(df_temp4) == len(df_temp5) == len(df_merged_temp)
# print(f"All rows matched: {rows_match}")


## Q6
df_2012 = pd.read_csv('retail_data_2012.csv')
merged_df = df_merged_temp.merge(df_2012, on='CustomerID', how='outer')

# ensure that tables match in length
rows_match = len(df_2012) == len(df_merged_temp) == len(merged_df)
# print(f"All rows matched: {rows_match}")


## Q7
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# [xmin, xmax, nbins]
features = {
    'Purchase_days_total': [0, 40, 40],
    'Purchase_amount_total': [0, 5000, 50],
    'Purchase_amount_avg': [0, 100, 50],
    'Recency': [0,370, 50]  
}

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2 
    
    sns.histplot(
        data=merged_df,
        x=feature,
        hue='Future_purchase',
        bins=features[feature][2],
        binrange=(features[feature][0], features[feature][1]),
        alpha=0.5,
        ax=axes[row, col],
        palette={0: 'blue', 1: 'green'}
        )
    axes[row, col].set_yscale('log')
    
plt.tight_layout()
plt.savefig('plots/customer_purchase_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

purchase_counts = merged_df['Future_purchase'].value_counts()

print(f"Customers WITHOUT future purchases: {purchase_counts.get(0, 0)}")
print(f"Customers WITH future purchases: {purchase_counts.get(1, 0)}")


## Q10
num_countries = merged_df['Country'].nunique()
print(f"Number of countries: {num_countries}")

country_counts = merged_df['Country'].value_counts()
country_percent = (country_counts / country_counts.sum()) * 100

plt.figure(figsize=(12, 6))
bars = plt.bar(
    country_percent.index, 
    country_percent.values,
    color='skyblue', 
    edgecolor='black'
)
plt.ylim(bottom=0.001, top=110)
plt.yscale('log')

plt.title('Percentage Distribution of Countries', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height/2.4,
        f'{height:.3f}%',  # Show 1 decimal place
        ha='center', 
        va='center',
        fontsize=8,
        rotation=90
    )

plt.tight_layout()
plt.savefig('plots/countries.png', dpi=300, bbox_inches='tight')
plt.close()

## Q11
threshold = 1  # 1% threshold
mask = country_percent < threshold
other_percent = country_percent[mask].sum()
filtered_percent = country_percent[~mask].copy()

filtered_percent['Other'] = other_percent
filtered_percent = filtered_percent.sort_values(ascending=False)

bars = plt.bar(
    filtered_percent.index,
    filtered_percent.values,
    color='skyblue',
    edgecolor='black'
)
plt.ylim(bottom=1, top=125)
plt.yscale('log')

plt.title('Country Distribution (Countries <1% grouped as "Other")', fontsize=14)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.5,
        f'{height:.1f}%',
        ha='center',
        va='bottom',
        fontsize=8
    )
   
plt.tight_layout()
plt.savefig('plots/countries_with_other.png', dpi=300, bbox_inches='tight')
plt.close()

# create new category in dataframe
small_countries = country_counts[country_percent < 1].index
merged_df['Country_processed'] = merged_df['Country'].apply(lambda x: 'Other' if x in small_countries else x)


## Q12
# Group by country and calculate purchase probability
country_prob = merged_df.groupby('Country')['Future_purchase'].mean().mul(100).reset_index(name='Purchase_Probability') # mean because it calculates the probability of future purchases 
country_prob = country_prob.sort_values('Purchase_Probability', ascending=False)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    country_prob['Country'],
    country_prob['Purchase_Probability'],
    color='skyblue',
    edgecolor='black',
    width=0.7 
)

plt.axhline(y=100, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)

plt.title('Probability of Future Purchase by Country', fontsize=14, pad=20)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Purchase Probability', fontsize=12)
plt.ylim(0, 110)

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels

for bar in bars:
    height = bar.get_height()
    if height < 0.1: continue
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height/1.5,
        f'{height:.1f}%',  # Show 1 decimal place
        ha='center', 
        va='center',
        fontsize=8,
        rotation=90
    )

plt.tight_layout()
plt.savefig('plots/purchase_probability.png', dpi=300, bbox_inches='tight')
plt.close()

# assign error to purchase probabilities
# note: Since Future_purchase is a binary variable, the probability estimate per country can be modeled as a binomial proportion
# we can assign an uncertainty using the formula for the standard error of a binomial proportion
# standard error = sqrt( p(1-p)/n ) where p is the mean of Future_purchase and n is the number of customers from that country (count)

country_err = merged_df.groupby("Country")["Future_purchase"].agg(['mean', 'count']) # compute both the mean and the count 
country_err["Standard Error"] = (country_err["mean"] * (1 - country_err["mean"]) / country_err["count"])**0.5

# create new bar chart
country_prob = merged_df.groupby('Country')['Future_purchase'].mean().mul(100).reset_index(name='Purchase_Probability')
country_count = merged_df.groupby('Country')['Future_purchase'].agg(['mean', 'count'])
country_count["Standard_Error"] = (country_count["mean"] * (1 - country_count["mean"]) / country_count["count"])**0.5
country_count["Standard_Error"] *= 100  # scale to percentage

country_prob = country_prob.merge(country_count[["Standard_Error"]], left_on="Country", right_index=True)
country_prob = country_prob.sort_values("Purchase_Probability", ascending=False)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    country_prob['Country'],
    country_prob['Purchase_Probability'],
    yerr=country_prob['Standard_Error'],
    error_kw=dict(ecolor='brown', linewidth=1.5),
    capsize=5,
    color='skyblue',
    edgecolor='black',
    width=0.7
)

plt.axhline(y=100, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)

plt.title('Probability of Future Purchase by Country', fontsize=14, pad=20)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Purchase Probability (%)', fontsize=12)
plt.ylim(0, 110)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plots/purchase_probability_with_error.png', dpi=300, bbox_inches='tight')
plt.close()

# try Wilson Error
# The Wilson score interval is a robust method for calculating confidence intervals (CIs) for binomial proportions (like purchase probabilities). 
# It outperforms standard error methods, especially when:
#   - Sample sizes are small
#   - Probabilities are near 0% or 100%
#   - You need accurate coverage (e.g., 95% CI)

def wilson_bounds(p, n, z=1.96):

    if n == 0:
        return (0, 0)
    
    denominator = 1 + z**2/n
    centre_adj = p + z**2/(2*n)
    adj_std = np.sqrt((p*(1 - p) + z**2/(4*n))/n)
    
    lower = (centre_adj - z*adj_std) / denominator
    upper = (centre_adj + z*adj_std) / denominator
    
    # Enforce bounds before converting to percentage
    lower = max(0, lower)
    upper = min(1, upper)
    
    return (lower * 100, upper * 100)

# Calculate purchase probability
country_prob = merged_df.groupby('Country')['Future_purchase'].mean().mul(100).reset_index(name='Purchase_Probability')
country_count = merged_df.groupby('Country')['Future_purchase'].agg(['mean', 'count'])

# Calculate Wilson bounds
country_count["Wilson_Bounds"] = country_count.apply(
    lambda row: wilson_bounds(row['mean'], row['count']), 
    axis=1
)

country_count["Lower"] = country_count["Wilson_Bounds"].apply(lambda x: x[0])
country_count["Upper"] = country_count["Wilson_Bounds"].apply(lambda x: x[1])

country_prob = country_prob.merge(country_count[["Lower", "Upper"]], left_on="Country", right_index=True)

country_prob["Lower_Error"] = country_prob["Purchase_Probability"] - country_prob["Lower"]
country_prob["Upper_Error"] = country_prob["Upper"] - country_prob["Purchase_Probability"]

country_prob = country_prob.sort_values("Purchase_Probability", ascending=False)
ordered_countries = country_prob['Country'].tolist()

errors = country_prob[["Lower_Error", "Upper_Error"]].values.T

plt.figure(figsize=(12, 6))

bars = plt.bar(
    ordered_countries,
    country_prob['Purchase_Probability'],
    yerr=errors,
    error_kw=dict(ecolor='brown', linewidth=1.5, capsize=5),
    color='skyblue',
    edgecolor='black',
    width=0.7
)

plt.axhline(y=100, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axhline(y=0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)

plt.title('Probability of Future Purchase by Country (Wilson 95% CI)', fontsize=14, pad=20)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Purchase Probability (%)', fontsize=12)
plt.ylim(-5, 105)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plots/purchase_probability_with_wilson_error.png', dpi=300, bbox_inches='tight')
plt.close()

## Q13
# note:To determine whether there is a significant difference in the probability of making a future purchase across different countries, 
# we can test if the proportion of Future_purchase = 1 differs by country using the chi square test of independence. 
# This test evaluates whether the distribution of future purchases is independent of country.
# Null hypothesis: Future purchase is independent of country.
# Alternative hypothesis: Future purchase depends on country.
# We can draw conclusions from the p-value.

from scipy.stats import chi2_contingency

# Create a contingency table: rows = countries, columns = [No Purchase, Purchase]
contingency_table = pd.crosstab(merged_df['Country'], merged_df['Future_purchase'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("======== Chi Square Test ========")
print(f"Chi-square statistic: {chi2:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p:.4f}")

# repeat this test with grouped countries as "Other"
merged_df['Country_processed'] = merged_df['Country'].apply(lambda x: 'Other' if x in small_countries else x)

country_err = merged_df.groupby("Country_processed")["Future_purchase"].agg(['mean', 'count']) # compute both the mean and the count 
country_err["Standard Error"] = (country_err["mean"] * (1 - country_err["mean"]) / country_err["count"])**0.5

# create new bar chart
country_prob = merged_df.groupby('Country_processed')['Future_purchase'].mean().mul(100).reset_index(name='Purchase_Probability')
country_count = merged_df.groupby('Country_processed')['Future_purchase'].agg(['mean', 'count'])
country_count["Standard_Error"] = (country_count["mean"] * (1 - country_count["mean"]) / country_count["count"])**0.5
country_count["Standard_Error"] *= 100  # scale to percentage

country_prob = country_prob.merge(country_count[["Standard_Error"]], left_on="Country_processed", right_index=True)
country_prob = country_prob.sort_values("Purchase_Probability", ascending=False)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    country_prob['Country_processed'],
    country_prob['Purchase_Probability'],
    yerr=country_prob['Standard_Error'],
    error_kw=dict(ecolor='brown', linewidth=1.5),
    capsize=5,
    color='skyblue',
    edgecolor='black',
    width=0.7
)

plt.axhline(y=100, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)

plt.title('Probability of Future Purchase by Country', fontsize=14, pad=20)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Purchase Probability (%)', fontsize=12)
plt.ylim(0, 110)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plots/purchase_probability_with_error_with_other.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a contingency table: rows = countries, columns = [No Purchase, Purchase]
contingency_table = pd.crosstab(merged_df['Country_processed'], merged_df['Future_purchase'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("======== Chi Square Test (with Other) ========")
print(f"Chi-square statistic: {chi2:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p:.4f}")