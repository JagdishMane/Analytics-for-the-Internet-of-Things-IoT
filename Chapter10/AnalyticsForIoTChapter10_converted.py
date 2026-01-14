"""
Python conversion of the R examples from Chapter10.

Features:
- Load R 'airquality' dataset (via statsmodels)
- Introduce the same missing values as the R script
- Impute missing values using sklearn IterativeImputer
- Center and scale data with StandardScaler
- Train Random Forest and Gradient Boosting models
- Run IsolationForest anomaly detection on `Wind`
- Simple ARIMA forecasting for `Wind` (pmdarima if available)

Run: python Chapter10/AnalyticsForIoTChapter10_converted.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.datasets import get_rdataset

from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

try:
    import pmdarima as pm
    HAS_PMD = True
except Exception:
    HAS_PMD = False


def main():
    # Load R "airquality" dataset
    ds = get_rdataset("airquality", "datasets")
    df = ds.data.copy()

    # Show basic summary and NA counts
    print("Original dataset shape:", df.shape)
    print(df.describe(include='all'))
    print("NA counts:\n", df.isnull().sum())

    # Reproduce the missing-value edits from the R example
    # R: mice_example_data[1:5,4] <- NA  -> rows 0..4, column Temp
    df.loc[0:4, 'Temp'] = np.nan
    # R: mice_example_data[6:10,3] <- NA -> rows 5..9, column Wind
    df.loc[5:9, 'Wind'] = np.nan

    print('\nAfter introducing NAs:')
    print(df.isnull().sum())

    # Impute numeric columns (exclude Month and Day)
    numeric_cols = ['Ozone', 'Solar.R', 'Wind', 'Temp']
    imputer = IterativeImputer(random_state=250, max_iter=50)
    imputed_vals = imputer.fit_transform(df[numeric_cols])
    imputed_df = df.copy()
    imputed_df[numeric_cols] = imputed_vals

    print('\nAfter imputation NA counts:')
    print(imputed_df.isnull().sum())

    # Density plot similar to R's densityplot
    plt.figure(figsize=(10, 6))
    for col in numeric_cols:
        sns.kdeplot(imputed_df[col], label=col)
    plt.title('KDE of imputed numeric features')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Center and scale numeric features
    scaler = StandardScaler()
    cs_vals = scaler.fit_transform(imputed_df[numeric_cols])
    cs_df = imputed_df.copy()
    cs_df[numeric_cols] = cs_vals

    print('\nCentered & scaled summary:')
    print(pd.DataFrame(cs_df[numeric_cols]).describe().loc[['mean', 'std']])

    # Create target class similar to R: tempClass <- ifelse(Temp > 0, "hot", "cold")
    cs_df['tempClass'] = np.where(cs_df['Temp'] > 0, 'hot', 'cold')

    # Define predictors and target
    predictors = ['Ozone', 'Solar.R', 'Wind']
    X = cs_df[predictors]
    y = cs_df['tempClass']

    # Split into training and test sets (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=42
    )

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)
    rf_pred = rf.predict(X_test)
    print('\nRandom Forest Confusion Matrix:')
    print(confusion_matrix(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))

    # Add RF outputs to test dataframe for inspection
    test_out = X_test.copy()
    test_out['RFclass'] = rf_pred
    # probability for positive class 'hot' (if present)
    if 'hot' in rf.classes_:
        hot_idx = list(rf.classes_).index('hot')
        test_out['RFProb'] = rf_proba[:, hot_idx]

    # Gradient Boosting
    gbm = GradientBoostingClassifier(random_state=42)
    gbm.fit(X_train, y_train)
    gbm_proba = gbm.predict_proba(X_test)
    gbm_pred = gbm.predict(X_test)
    print('\nGradient Boosting Confusion Matrix:')
    print(confusion_matrix(y_test, gbm_pred))
    print(classification_report(y_test, gbm_pred))

    if 'hot' in gbm.classes_:
        hot_idx = list(gbm.classes_).index('hot')
        test_out['GBMProb'] = gbm_proba[:, hot_idx]

    # Anomaly detection on Wind using IsolationForest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(cs_df[['Wind']])
    cs_df['anomaly'] = iso.predict(cs_df[['Wind']])  # -1 anomaly, 1 normal
    anomalies = cs_df[cs_df['anomaly'] == -1]
    print(f"\nAnomalies detected in Wind: {len(anomalies)} rows")

    # ARIMA forecasting for Wind
    wind_series = pd.Series(imputed_df['Wind'].values, index=pd.date_range('1973-05-01', periods=len(imputed_df), freq='D'))
    plt.figure()
    wind_series.plot(title='Wind time series (imputed)')
    plt.tight_layout()
    plt.show()

    if HAS_PMD:
        print('\nUsing pmdarima.auto_arima for automatic ARIMA selection')
        arima_model = pm.auto_arima(wind_series, seasonal=False, suppress_warnings=True)
        fc = arima_model.predict(n_periods=30)
        idx = pd.date_range(wind_series.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        forecast = pd.Series(fc, index=idx)
    else:
        print('\nPmdarima not found; fitting a simple ARIMA(1,0,1) via statsmodels')
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(wind_series, order=(1, 0, 1)).fit()
        fc_res = model.get_forecast(steps=30)
        forecast = fc_res.predicted_mean

    plt.figure()
    wind_series.plot(label='observed')
    forecast.plot(label='forecast', linestyle='--')
    plt.legend()
    plt.title('Wind forecast (30 days)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
