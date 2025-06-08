# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset
diabetes_data = pd.read_csv('C:/Users/Bhavya/OneDrive/Documents/celebal_internship/week-3/diabetes-dataset.csv')
diabetes_data.head()

# dataset summary
print("Shape of the dataset:", diabetes_data.shape)
print("\nColumns in the dataset:\n", diabetes_data.columns)
print("\nMissing values in each column:\n", diabetes_data.isnull().sum())
print("\nData types:\n", diabetes_data.dtypes)
print("\nSummary Statistics:\n", diabetes_data.describe(include='all'))
diabetes_data.fillna(diabetes_data.median(numeric_only=True), inplace=True)
print("\nMissing values after imputation:\n", diabetes_data.isnull().sum())

# heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(diabetes_data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# histogram of each feature
diabetes_data.hist(bins=20, figsize=(14, 10), edgecolor='black')
plt.suptitle("Distribution of Features", fontsize=16)
plt.show()

# pairplot for selected columns
selected_columns = diabetes_data.columns[:5] 
sns.pairplot(diabetes_data[selected_columns])
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# output vs input plot
sns.boxplot(x='Outcome', y='Glucose', data=diabetes_data)
plt.title("Glucose Levels vs Outcome")
plt.show()
