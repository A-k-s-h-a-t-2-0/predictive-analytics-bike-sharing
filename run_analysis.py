"""
CSE3141: Predictive Analytics - Course Project
Title: Predictive Demand Modeling for Urban Bike Sharing using Generalized Linear Models (Poisson & Negative Binomial Regression)
Course Coordinator / Handout: Manipal University Jaipur, CSE3141
Lectures Covered: Lec 5 - Lec 9 (Data Prep & EDA) -> Lec 10-11 (Feature Eng) -> Lec 19-21 (GLM/Poisson/NegBinomial) -> Lec 30-36 (Validation/Metrics)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------
# 1. DATA INGESTION & OVERVIEW (Lec 5)
# -------------------------------------------------------------
print("="*70)
print("STEP 1: DATA INGESTION (Lecture 5)")
print("="*70)
df = pd.read_csv('data/hour.csv')
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("First 3 rows:\n", df.head(3))

# -------------------------------------------------------------
# 2. DATA PREPROCESSING: CLEANING & OUTLIERS (Lec 6)
# -------------------------------------------------------------
print("\n" + "="*70)
print("STEP 2: PREPROCESSING - MISSING VALUES & OUTLIERS (Lecture 6)")
print("="*70)
missing_counts = df.isnull().sum()
print("Missing values per column:\n", missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "Zero missing values detected!")

# IQR-based outlier detection for 'cnt'
Q1 = df['cnt'].quantile(0.25)
Q3 = df['cnt'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['cnt'] < lower_bound) | (df['cnt'] > upper_bound)]
print(f"\nIQR Outlier Detection for Target 'cnt':")
print(f"Q1 (25%): {Q1:.2f}, Q3 (75%): {Q3:.2f}, IQR: {IQR:.2f}")
print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
print(f"Total Outliers identified via IQR: {len(outliers_iqr)} ({len(outliers_iqr)/len(df)*100:.2f}%)")

# Box Plot for Lecture 6
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['cnt'], color='skyblue')
plt.title("Box Plot of Rental Count (cnt)\n[Outlier Inspection - Lec 6]")
plt.ylabel("Hourly Count")

plt.subplot(1, 2, 2)
sns.boxplot(x='season', y='cnt', data=df, palette='Set2')
plt.title("Box Plot of cnt by Season (1:Spring, 2:Summer, 3:Fall, 4:Winter)")
plt.xlabel("Season")
plt.ylabel("Hourly Count")
plt.tight_layout()
plt.savefig('figures/01_boxplot_outliers_lec6.png', dpi=300)
plt.close()
print("Saved: figures/01_boxplot_outliers_lec6.png")

# -------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA): STATISTICAL METRICS (Lec 7-8)
# -------------------------------------------------------------
print("\n" + "="*70)
print("STEP 3: EDA - DESCRIPTIVE STATS & Z-SCORE (Lectures 7 & 8)")
print("="*70)

num_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
stats_summary = pd.DataFrame(index=num_cols)
stats_summary['Mean'] = df[num_cols].mean()
stats_summary['Median'] = df[num_cols].median()
stats_summary['Mode'] = df[num_cols].mode().iloc[0]
stats_summary['Variance'] = df[num_cols].var()
stats_summary['Std_Dev'] = df[num_cols].std()

print("Summary of Statistical Measures (Lec 7 & 8):")
print(stats_summary.round(3))

# Overdispersion diagnostic: Mean vs Variance
mean_cnt = df['cnt'].mean()
var_cnt = df['cnt'].var()
dispersion_ratio = var_cnt / mean_cnt
print(f"\n--- OVERDISPERSION TEST (Key for Lecture 20 & 21) ---")
print(f"Target 'cnt': Mean = {mean_cnt:.2f}, Variance = {var_cnt:.2f}")
print(f"Dispersion Ratio (Var / Mean) = {dispersion_ratio:.2f}")
print("-> Since Dispersion Ratio >> 1, the data exhibits SEVERE OVERDISPERSION.")
print("-> Theoretical Implication: Standard Poisson assumes Var = Mean.")
print("   Negative Binomial Regression is theoretically REQUIRED to account for overdispersion!")

# Z-score computation (Lec 8)
df['cnt_zscore'] = stats.zscore(df['cnt'])
z_outliers = df[df['cnt_zscore'].abs() > 3]
print(f"\nZ-Score Analysis (Lec 8): Found {len(z_outliers)} records with |Z| > 3 standard deviations from mean.")

# -------------------------------------------------------------
# 4. DATA VISUALIZATION (Lecture 9: 5 Core Plots)
# -------------------------------------------------------------
print("\n" + "="*70)
print("STEP 4: DATA VISUALIZATION - 5 CORE PLOTS (Lecture 9)")
print("="*70)

# 1. Histogram (Lec 9)
plt.figure(figsize=(8, 5))
sns.histplot(df['cnt'], kde=True, bins=40, color='darkblue')
plt.title("1. Histogram of Hourly Bike Rentals (cnt)\n[Positively Skewed Distribution - Lec 9]")
plt.xlabel("Rental Count per Hour")
plt.ylabel("Frequency")
plt.savefig('figures/02_histogram_lec9.png', dpi=300)
plt.close()

# 2. Scatter Plot (Lec 9)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='temp', y='cnt', data=df.sample(2000, random_state=42), alpha=0.4, color='crimson')
plt.title("2. Scatter Plot: Normalized Temperature vs Rental Count\n[Trend Identification - Lec 9]")
plt.xlabel("Normalized Temperature")
plt.ylabel("Rental Count (cnt)")
plt.savefig('figures/03_scatter_plot_lec9.png', dpi=300)
plt.close()

# 3. Bar Chart (Lec 9)
plt.figure(figsize=(8, 5))
season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
df['season_label'] = df['season'].map(season_names)
season_avg = df.groupby('season_label')['cnt'].mean().reindex(['Spring', 'Summer', 'Fall', 'Winter'])
sns.barplot(x=season_avg.index, y=season_avg.values, palette='Blues_d')
plt.title("3. Bar Chart: Average Hourly Rentals across Seasons\n[Categorical Comparison - Lec 9]")
plt.xlabel("Season")
plt.ylabel("Mean Hourly Rentals")
plt.savefig('figures/04_bar_chart_lec9.png', dpi=300)
plt.close()

# 4. Pie Chart (Lec 9)
plt.figure(figsize=(7, 7))
weather_labels = {1: 'Clear / Few Clouds', 2: 'Mist / Cloudy', 3: 'Light Snow / Rain', 4: 'Heavy Rain / Ice'}
weather_counts = df['weathersit'].map(weather_labels).value_counts()
plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
plt.title("4. Pie Chart: Distribution of Weather Situations\n[Proportions - Lec 9]")
plt.savefig('figures/05_pie_chart_lec9.png', dpi=300)
plt.close()

# 5. Heatmap (Lec 9)
plt.figure(figsize=(9, 7))
corr_matrix = df[['temp', 'atemp', 'hum', 'windspeed', 'hr', 'cnt']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("5. Heatmap: Correlation Matrix of Key Numeric Features\n[Collinearity Inspection - Lec 9]")
plt.savefig('figures/06_heatmap_lec9.png', dpi=300)
plt.close()

print("All 5 Lecture 9 visualization figures generated in ./figures/")

# -------------------------------------------------------------
# 5. FEATURE ENGINEERING & SELECTION (Lectures 10 & 11)
# -------------------------------------------------------------
print("\n" + "="*70)
print("STEP 5: FEATURE ENGINEERING & SELECTION (Lectures 10 & 11)")
print("="*70)

# Rush hour flag: Peak commuter hours 7-9 AM and 17-19 PM on working days
df['rush_hour'] = (((df['hr'].between(7, 9)) | (df['hr'].between(17, 19))) & (df['workingday'] == 1)).astype(int)

# Cyclical encoding for hour (24 hours cycle)
df['sin_hr'] = np.sin(2 * np.pi * df['hr'] / 24)
df['cos_hr'] = np.cos(2 * np.pi * df['hr'] / 24)

# One-hot encoding for season and weathersit
encoded_df = pd.get_dummies(df, columns=['season', 'weathersit'], drop_first=True)

feature_cols = [
    'yr', 'holiday', 'workingday', 'rush_hour',
    'temp', 'hum', 'windspeed', 'sin_hr', 'cos_hr',
    'season_2', 'season_3', 'season_4',
    'weathersit_2', 'weathersit_3', 'weathersit_4'
]

X = encoded_df[feature_cols].astype(float)
y = encoded_df['cnt'].astype(float)

print(f"Selected {len(feature_cols)} features after engineering:")
print(feature_cols)

# -------------------------------------------------------------
# 6. MODEL VALIDATION SPLIT (Lecture 30)
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows (80/20 split)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------
# 7. MODELING: OLS, POISSON, & NEGATIVE BINOMIAL (Lec 19, 20, 21)
# -------------------------------------------------------------
print("\n" + "="*70)
print("STEP 6: GENERALIZED LINEAR MODELS (Lectures 19, 20, 21)")
print("="*70)

# Model 1: Ordinary Least Squares (OLS) Linear Regression (Baseline)
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)
y_pred_ols = ols_model.predict(X_test_scaled)

# Model 2: Poisson Regression (Lec 20) via Scikit-Learn
poisson_model = PoissonRegressor(alpha=1e-3, max_iter=500)
poisson_model.fit(X_train_scaled, y_train)
y_pred_poisson = poisson_model.predict(X_test_scaled)

# Model 3: Negative Binomial Regression (Lec 21) via Statsmodels
train_data_sm = pd.concat([X_train, y_train], axis=1)
test_data_sm = pd.concat([X_test, y_test], axis=1)

formula = 'cnt ~ ' + ' + '.join(feature_cols)
nb_model = smf.glm(
    formula=formula,
    data=train_data_sm,
    family=sm.families.NegativeBinomial(alpha=0.5)
).fit()

y_pred_nb = nb_model.predict(test_data_sm)

print("\n--- STATSMODELS NEGATIVE BINOMIAL ESTIMATION SUMMARY (Lec 21) ---")
print(nb_model.summary().tables[1])

# -------------------------------------------------------------
# 8. PERFORMANCE METRICS & EVALUATION (Lectures 30-36)
# -------------------------------------------------------------
print("\n" + "="*70)
print("STEP 7: PERFORMANCE METRICS & VALIDATION (Lectures 31, 32)")
print("="*70)

def evaluate_predictions(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    neg_preds = (y_pred < 0).sum()
    return {'Model': name, 'RMSE (Lec 31)': rmse, 'MAE (Lec 32)': mae, 'R2': r2, 'Negative Predictions': neg_preds}

results = pd.DataFrame([
    evaluate_predictions(y_test, y_pred_ols, "1. Linear Regression (OLS)"),
    evaluate_predictions(y_test, y_pred_poisson, "2. Poisson Regression (Lec 20)"),
    evaluate_predictions(y_test, y_pred_nb, "3. Negative Binomial (Lec 21)")
])

print("\n--- Model Comparison Benchmark ---")
print(results.round(3).to_string(index=False))

# -------------------------------------------------------------
# 9. HYPERPARAMETER TUNING & REGULARIZATION (Lectures 35 & 36)
# -------------------------------------------------------------
print("\n" + "="*70)
print("STEP 8: HYPERPARAMETER TUNING VIA GRIDSEARCHCV (Lecture 35)")
print("="*70)

param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 5.0]}
grid_search = GridSearchCV(
    PoissonRegressor(max_iter=500),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=1
)
grid_search.fit(X_train_scaled, y_train)
best_poisson = grid_search.best_estimator_
print(f"Optimal Alpha via 5-Fold Grid Search: {grid_search.best_params_['alpha']}")

y_pred_tuned = best_poisson.predict(X_test_scaled)
tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
print(f"Tuned Poisson Test RMSE: {tuned_rmse:.3f}")

# -------------------------------------------------------------
# 10. PREDICTION VISUALIZATION: ACTUAL VS PREDICTED
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Slice of 100 test hours for visual comparison
plt.subplot(1, 2, 1)
test_sample = 120
plt.plot(np.arange(test_sample), y_test.iloc[:test_sample].values, label='Actual Demand', color='black', lw=2)
plt.plot(np.arange(test_sample), y_pred_ols[:test_sample], label='OLS Linear (Predicts < 0)', linestyle='--', color='red', alpha=0.7)
plt.plot(np.arange(test_sample), y_pred_nb.iloc[:test_sample].values, label='Negative Binomial (Non-negative)', color='green', lw=1.8)
plt.title(f"Actual vs Predicted Demand (First {test_sample} Test Hours)")
plt.xlabel("Time Index (Hours)")
plt.ylabel("Hourly Bike Rentals")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_nb, alpha=0.3, color='teal')
plt.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.title("Negative Binomial: Predicted vs Actual Scatter")
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.legend()
plt.tight_layout()
plt.savefig('figures/07_actual_vs_predicted.png', dpi=300)
plt.close()
print("Saved: figures/07_actual_vs_predicted.png")

print("\n" + "="*70)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("="*70)
