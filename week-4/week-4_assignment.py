# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data=pd.read_csv('C:/Users/Bhavya/OneDrive/Documents/celebal_internship/week-4/Students Social Media Addiction.csv')
print(data.head())

# Data analysis
print("DATA DESCRIPTION\n:",data.describe)
print("SHAPE OF THE DATA:",data.shape)
print("SIZE OF THE DATA:",data.size)
print("COLUMNS OF THE DATA:",data.columns)

# Missing values
print(data.isnull().sum())
data.fillna(data.median(numeric_only=True), inplace=True)

# Histogram (Visualization)
data.hist(figsize=(16, 12), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Boxplots to detect outliers
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(16, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=col, data=data)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Correlation matrix
corr = data.corr(numeric_only=True)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Categorical relationships
# Example: Addicted Score vs Avg Daily Usage Hours
sns.countplot(x='Addicted_Score', hue='Avg_Daily_Usage_Hours', data=data)
plt.title("Addiction Level by Daily Usage")
plt.show()


