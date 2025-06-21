# Import the Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Read the datasets
train_df = pd.read_csv('week-5\\train.csv')
test_df = pd.read_csv('week-5\\test.csv')
submission_df = pd.read_csv('week-5\\sample_submission.csv')

# Save and remove target
target = train_df['SalePrice']
train_df.drop('SalePrice', axis=1, inplace=True)

# Mark datasets
train_df['is_train'] = 1
test_df['is_train'] = 0

# Combine for preprocessing
full_data = pd.concat([train_df, test_df], axis=0)

# Remove less informative or mostly missing columns
cols_to_remove = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
full_data.drop(columns=cols_to_remove, inplace=True)

# Fill NA values for categorical features with 'None'
none_fill_cols = [
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType'
]
for col in none_fill_cols:
    full_data[col] = full_data[col].fillna('None')

# Fill NA for numerical values where 0 makes sense
zero_fill_cols = [
    'MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'
]
for col in zero_fill_cols:
    full_data[col].fillna(0, inplace=True)

# Fill LotFrontage by Neighborhood median
full_data['LotFrontage'] = full_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# General fill for remaining missing values
for col in full_data.columns:
    if full_data[col].dtype == 'object':
        full_data[col] = full_data[col].fillna(full_data[col].mode()[0])
    else:
        full_data[col] = full_data[col].fillna(full_data[col].median())

# Convert some numerical columns to categorical
full_data['MSSubClass'] = full_data['MSSubClass'].astype(str)
full_data['MoSold'] = full_data['MoSold'].astype(str)
full_data['YrSold'] = full_data['YrSold'].astype(str)

# Map ratings to numeric values
qual_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
for col in quality_cols:
    full_data[col] = full_data[col].map(qual_dict)

# Feature engineering
full_data['TotalBaths'] = (
    full_data['FullBath'] + 0.5 * full_data['HalfBath'] +
    full_data['BsmtFullBath'] + 0.5 * full_data['BsmtHalfBath']
)
full_data['PorchArea'] = (
    full_data['OpenPorchSF'] + full_data['EnclosedPorch'] +
    full_data['3SsnPorch'] + full_data['ScreenPorch']
)
full_data['TotalSquareFootage'] = (
    full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
)
full_data['Age'] = full_data['YrSold'].astype(int) - full_data['YearBuilt']
full_data['RemodelAge'] = full_data['YrSold'].astype(int) - full_data['YearRemodAdd']
full_data['Remodeled'] = (full_data['YearBuilt'] != full_data['YearRemodAdd']).astype(int)
full_data['HasGarage'] = (full_data['GarageArea'] > 0).astype(int)
full_data['HasFireplace'] = (full_data['Fireplaces'] > 0).astype(int)
full_data['HasPool'] = (full_data['PoolArea'] > 0).astype(int)

# One-hot encoding
full_data = pd.get_dummies(full_data, drop_first=True)

# Split into train and test again
X_train = full_data[full_data['is_train'] == 1].drop('is_train', axis=1)
X_test = full_data[full_data['is_train'] == 0].drop('is_train', axis=1)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, target)

# Evaluate on training data
train_preds = lr_model.predict(X_train_scaled)
train_rmse = np.sqrt(mean_squared_error(target, train_preds))
print(f"Training RMSE: {train_rmse:.4f}")

# Predict on test data and prepare submission
test_preds = lr_model.predict(X_test_scaled)
submission_df['SalePrice'] = test_preds
submission_df.to_csv('refactored_linear_regression_submission.csv', index=False)
print("Saved submission: 'refactored_linear_regression_submission.csv'")
