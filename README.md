# Exploratory Data Analysis (EDA) Script

## Overview

This repository contains exploratory data analysis (EDA) code for the Home Credit Default Risk dataset (application_data.csv). The code is structured to handle missing values, clean the dataset, and perform analysis on various features.

## Libraries Used
pandas: For data manipulation and analysis.
numpy: For numerical operations.
matplotlib: For creating visualizations.
seaborn: For enhancing the visual appeal of plots.

## Data Loading
1. Install the required libraries:

The dataset is loaded using Pandas, and the initial few rows are displayed for a quick overview.

```bash
pip install pandas numpy matplotlib seaborn
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

pd.set_option("display.max_columns", None)

# Load the dataset
a_d = pd.read_csv("application_data.csv")
a_d.head()

```
## Data Cleaning

The code addresses missing values in the dataset by dropping columns with a specified percentage of missing values and filling others with appropriate strategies.
```bash
# Set display options
pd.set_option('display.max_rows', 200)

# Check missing values percentage
a_d.isnull().mean() * 100

# Drop columns with more than 45% missing values
percentage = 45
threshold = int(((100 - percentage) / 100) * a_d.shape[0] + 1)
a = a_d.dropna(axis=1, thresh=threshold)

# Handling missing values in specific columns
# ... (code for handling missing values)

# Display columns with missing values
null_cols = list(a.columns[a.isna().any()])
len(null_cols)
```
## Data Exploration

Explore various aspects of the dataset, including descriptive statistics, boxplots, and visualizations for both categorical and numerical columns.
```bash
# Extract numerical and categorical columns
cat_cols = list(a.columns[a.dtypes == object])
num_cols = list(a.columns[(a.dtypes == np.int64) | (a.dtypes == np.float64)])

# Separate numerical columns into those with and without "FLAG"
num_cols_withoutflag = [col for col in num_cols if not col.startswith("FLAG")]
num_cols_withflag = [col for col in num_cols if col.startswith("FLAG")]

# Explore numerical columns without "FLAG"
for col in num_cols_withoutflag:
    print(a[col].describe())
    plt.figure(figsize=[8, 5])
    sns.boxplot(data=a, x=col)
    plt.show()
    print("_________________")

```
## Target Variable Analysis

Analyze the distribution of the target variable (TARGET) and perform further analysis based on this variable.
```bash
# Analyze the distribution of the target variable (TARGET)
t0 = a[a.TARGET == 0]
t1 = a[a.TARGET == 1]

# Display the distribution of the target variable
a.TARGET.value_counts(normalize=True) * 100

# Plot the distribution of the target variable
sns.countplot(x='TARGET', data=a)
plt.title('Distribution of Target Variable')
plt.show()

```
## Feature Engineering
Create new features, categorize numerical variables, and generate visualizations.
```bash
# Binning the AMT_CREDIT column
a["AMT_CREDIT_Category"] = pd.cut(a.AMT_CREDIT, [0, 200000, 400000, 600000, 800000, 1000000],
                                   labels=["Very low credit", "Low credit", "Medium credit", "High credit", "Very high credit"])

# Display the percentage distribution of the new feature
a["AMT_CREDIT_Category"].value_counts(normalize=True) * 100

# Plot a bar chart for the new feature
a["AMT_CREDIT_Category"].value_counts(normalize=True).plot.bar()
plt.title('Distribution of AMT_CREDIT_Category')
plt.show()

# Binning the YEARS_BIRTH column
a["AGE_Category"] = pd.cut(a.YEARS_BIRTH, [0, 25, 45, 65, 85],
                           labels=["Below 25", "25-45", "45-65", "65-85"])

# Display the percentage distribution of the new feature
a["AGE_Category"].value_counts(normalize=True) * 100

# Plot a pie chart for the new feature
a["AGE_Category"].value_counts(normalize=True).plot.pie(autopct='%1.2f%%')
plt.title('Distribution of AGE_Category')
plt.show()
```
## Categorical Columns Analysis
Explore the distribution of categorical columns for both TARGET=0 and TARGET=1:
```bash
for col in cat_cols:
    print(f"Plotting {col} for Target 0 and 1")
    plt.figure(figsize=[10, 7])
    
    # Plot for Target 0
    plt.subplot(1, 2, 1)
    t0[col].value_counts(normalize=True).plot.bar()
    plt.title("Target 0")
    plt.xlabel(col)
    plt.ylabel("Density")
    
    # Plot for Target 1
    plt.subplot(1, 2, 2)
    t1[col].value_counts(normalize=True).plot.bar()
    plt.title("Target 1")
    plt.xlabel(col)
    plt.ylabel("Density")
    
    plt.show()
    
    print("\n\n__________________________________")
```
## Numeric Columns Analysis
Explore the distribution and relationships of numeric columns:

```bash
plt.figure(figsize=[10, 6])
sns.distplot(t0['AMT_GOODS_PRICE'], label='Target 0', hist=False)
sns.distplot(t1['AMT_GOODS_PRICE'], label='Target 1', hist=False)
plt.legend()
plt.title('Distribution of AMT_GOODS_PRICE for Target 0 and 1')
plt.show()

plt.figure(figsize=[15, 10])
plt.subplot(1, 2, 1)
sns.boxplot(x='WEEKDAY_APPR_PROCESS_START', y='HOUR_APPR_PROCESS_START', data=t0)
plt.subplot(1, 2, 2)
sns.boxplot(x='WEEKDAY_APPR_PROCESS_START', y='HOUR_APPR_PROCESS_START', data=t1)

plt.figure(figsize=[15, 10])
plt.subplot(1, 2, 1)
sns.boxplot(x='AGE_Category', y='AMT_CREDIT', data=t0)
plt.subplot(1, 2, 2)
sns.boxplot(x='AGE_Category', y='AMT_CREDIT', data=t1)

sns.pairplot(t0[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]])

corr_data = a[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "YEARS_EMPLOYED", "YEARS_REGISTRATION", "YEARS_ID_PUBLISH", "YEARS_LAST_PHONE_CHANGE"]]
corr_data.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr_data.corr(), annot=True, cmap="RdYlGn")
```
## Previous Application Data Analysis
Explore the previous_application.csv dataset:

```bash
p = pd.read_csv('previous_application.csv')

# Quality check
percentage = 50
threshold_p = int(((100 - percentage) / 100) * p.shape[0] + 1)
p = p.dropna(axis=1, thresh=threshold_p)

# Analysis on numeric columns
# ... (code for numeric columns analysis)

# Binning the AMT_CREDIT column
p["AMT_CREDIT_Category"] = pd.cut(p.AMT_CREDIT, [0, 200000, 400000, 600000, 800000, 1000000],
                                   labels=["Very low credit", "Low credit", "Medium credit", "High credit", "Very high credit"])

# Plot a bar chart for the new feature
p["AMT_CREDIT_Category"].value_counts(normalize=True).plot.bar()
plt.title('Distribution of AMT_CREDIT_Category')
plt.show()

# Binning the AMT_GOODS_PRICE column
p["AMT_GOODS_PRICE_Category"] = pd.cut(p.AMT_GOODS_PRICE, [0, 0.25, 0.45, 0.65, 0.85, 1],
                                       labels=["Very low price", "Low price", "Medium price", "High price", "Very high price"])

# ... (code for further analysis and visualizations)
```
## Approved, Cancelled, Refused, and Unused Offers
Visualize the distribution of AMT_APPLICATION across different days of the week for each application status:
```bash
plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START', y='AMT_APPLICATION', data=approved)
plt.title("Plot for Approved")
plt.show()

plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START', y='AMT_APPLICATION', data=cancelled)
plt.title("Plot for Cancelled")
plt.show()

plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START', y='AMT_APPLICATION', data=refused)
plt.title("Plot for Refused")
plt.show()

plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START', y='AMT_APPLICATION', data=unused)
plt.title("Plot for Unused")
plt.show()
```
## Scatter Plots and Heatmaps
Visualize relationships and correlations for different application statuses:

```bash
plt.figure(figsize=(15,10))
plt.subplot(1,4,1)
plt.title("Approved")
sns.scatterplot(x="AMT_ANNUITY", y="AMT_GOODS_PRICE", data=approved)

plt.figure(figsize=(15,10))
plt.subplot(1,4,2)
plt.title("Cancelled")
sns.scatterplot(x="AMT_ANNUITY", y="AMT_GOODS_PRICE", data=cancelled)

plt.figure(figsize=(15,10))
plt.subplot(1,4,3)
plt.title("Refused")
sns.scatterplot(x="AMT_ANNUITY", y="AMT_GOODS_PRICE", data=refused)

plt.figure(figsize=(15,10))
plt.subplot(1,4,4)
plt.title("Unused")
sns.scatterplot(x="AMT_ANNUITY", y="AMT_GOODS_PRICE", data=unused)

cor_approved = approved[["DAYS_DECISION", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_GOODS_PRICE", "CNT_PAYMENT"]]
cor_cancelled = cancelled[["DAYS_DECISION", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_GOODS_PRICE", "CNT_PAYMENT"]]
cor_refused = refused[["DAYS_DECISION", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_GOODS_PRICE", "CNT_PAYMENT"]]
cor_unused = unused[["DAYS_DECISION", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_GOODS_PRICE", "CNT_PAYMENT"]]

plt.figure(figsize=[10,10])
sns.heatmap(cor_approved.corr(), annot=True, cmap="Reds")
plt.title("Heatmap for Approved")

plt.figure(figsize=[10,10])
sns.heatmap(cor_cancelled.corr(), annot=True, cmap="Reds")

plt.figure(figsize=[10,10])
sns.heatmap(cor_refused.corr(), annot=True, cmap="Reds")

plt.figure(figsize=[10,10])
sns.heatmap(cor_unused.corr(), annot=True, cmap="Reds")
```
## Merged Data Analysis
Combine data from application_data.csv and previous_application.csv datasets and analyze the relationships:
```bash
merge_df = a.merge(p, on=['SK_ID_CURR'], how='left')

# Drop columns with "FLAG" prefix
for col in merge_df.columns:
    if col.startswith("FLAG"):
        merge_df.drop(columns=col, axis=1, inplace=True)

merge_df.shape

# ... (further analysis and visualizations based on merged data)
```
