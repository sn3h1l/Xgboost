import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['is_train'] = 1
test['is_train'] = 0
test['Obtained Electricity'] = 0
data = pd.concat([train, test], axis=0, ignore_index=True)

def create_features(df):
    df = df.copy()
    df['Wind Speed'] = df.groupby(['Weather', 'Day'])['Wind Speed'].transform(
        lambda x: x.fillna(x.median()) if x.notna().sum() > 0 else x.fillna(df['Wind Speed'].median())
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    categorical_cols = ['Day', 'Weather', 'Hindrance']
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    le_day = LabelEncoder()
    le_weather = LabelEncoder()
    le_hindrance = LabelEncoder()
    df['Day_encoded'] = le_day.fit_transform(df['Day'].astype(str))
    df['Weather_encoded'] = le_weather.fit_transform(df['Weather'].astype(str))
    df['Hindrance_encoded'] = le_hindrance.fit_transform(df['Hindrance'].astype(str))
    df['Wind_Speed_Cubed'] = df['Wind Speed'] ** 3
    df['Wind_Speed_Squared'] = df['Wind Speed'] ** 2
    df['Wind_Speed_Fourth'] = df['Wind Speed'] ** 4
    df['Power_Potential'] = (0.5 * df['Air Density'] * df['Turbine Sweep Area'] * df['Wind_Speed_Cubed'])
    df['Adjusted_Power'] = (df['Power_Potential'] * df['Power Coefficient'] / 100 * df['Average Turbine Efficiency'] / 100)
    df['Active_Turbine_Power'] = df['Adjusted_Power'] * df['Active Number']
    df['Power_Per_Turbine'] = df['Adjusted_Power'] / (df['Active Number'] + 1)
    df['WindSpeed_ActiveNumber'] = df['Wind Speed'] * df['Active Number']
    df['WindSpeed_SweepArea'] = df['Wind Speed'] * df['Turbine Sweep Area']
    df['WindSpeed_AirDensity'] = df['Wind Speed'] * df['Air Density']
    df['WindSpeed_Cubed_ActiveNumber'] = df['Wind_Speed_Cubed'] * df['Active Number']
    df['Total_Efficiency'] = df['Power Coefficient'] * df['Average Turbine Efficiency'] / 10000
    df['Efficiency_ActiveNumber'] = df['Total_Efficiency'] * df['Active Number']
    df['Efficiency_WindSpeed'] = df['Total_Efficiency'] * df['Wind Speed']
    df['AirDensity_WindSpeed'] = df['Air Density'] * df['Wind Speed']
    df['AirDensity_SweepArea'] = df['Air Density'] * df['Turbine Sweep Area']
    df['AirDensity_ActiveNumber'] = df['Air Density'] * df['Active Number']
    df['ActiveTime_ActiveNumber'] = df['Average Active Time'] * df['Active Number']
    df['ActiveTime_WindSpeed'] = df['Average Active Time'] * df['Wind Speed']
    df['ActiveTime_Efficiency'] = df['Average Active Time'] * df['Total_Efficiency']
    df['Radius_Squared'] = df['Average Radius'] ** 2
    df['SweepArea_per_Turbine'] = df['Turbine Sweep Area'] / (df['Active Number'] + 1)
    df['Radius_ActiveNumber'] = df['Average Radius'] * df['Active Number']
    df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Relative Humidity'] / 100
    df['CloudCover_WindSpeed'] = df['Cloud Cover'] * df['Wind Speed']
    df['CloudCover_ActiveNumber'] = df['Cloud Cover'] * df['Active Number']
    df['Pressure_AirDensity'] = df['Pressure'] * df['Air Density']
    df['Pressure_Temperature'] = df['Pressure'] / (df['Temperature'] + 273.15)
    df['DewPoint_Temperature_Diff'] = df['Temperature'] - df['Dew Point']
    df['DewPoint_Humidity'] = df['Dew Point'] * df['Relative Humidity']
    df['Viscosity_WindSpeed'] = df['Viscosity'] * df['Wind Speed']
    df['Viscosity_AirDensity'] = df['Viscosity'] * df['Air Density']
    df['Pitch_WindSpeed'] = df['Pitch'] * df['Wind Speed']
    df['Pitch_ActiveNumber'] = df['Pitch'] * df['Active Number']
    df['Pitch_Efficiency'] = df['Pitch'] * df['Total_Efficiency']
    df['Hindrance_ActiveNumber'] = df['Hindrance_encoded'] * df['Active Number']
    df['Hindrance_WindSpeed'] = df['Hindrance_encoded'] * df['Wind Speed']
    df['Elevation_AirDensity'] = df['Average Elevation'] * df['Air Density']
    df['Elevation_Pressure'] = df['Average Elevation'] * df['Pressure']
    df['WindSpeed_to_Radius'] = df['Wind Speed'] / (df['Average Radius'] + 1)
    df['ActiveNumber_to_SweepArea'] = df['Active Number'] / (df['Turbine Sweep Area'] + 1)
    df['Efficiency_to_WindSpeed'] = df['Total_Efficiency'] / (df['Wind Speed'] + 1)
    df['Power_to_ActiveTime'] = df['Adjusted_Power'] / (df['Average Active Time'] + 1)
    df['WindSpeed_x_Efficiency'] = df['Wind Speed'] * df['Total_Efficiency']
    df['WindSpeed_x_ActiveTime'] = df['Wind Speed'] * df['Average Active Time']
    df['SweepArea_x_Efficiency'] = df['Turbine Sweep Area'] * df['Total_Efficiency']
    df['Total_Energy_Capacity'] = (df['Active Number'] * df['Turbine Sweep Area'] * df['Total_Efficiency'] * df['Average Active Time'])
    df['Wind_Energy_Index'] = (df['Wind_Speed_Cubed'] * df['Air Density'] * df['Active Number'] * df['Total_Efficiency'])
    df['Weather_Impact'] = (df['Cloud Cover'] * df['Hindrance_encoded'] * (100 - df['Relative Humidity']) / 100)
    return df

data = create_features(data)
train = data[data['is_train'] == 1].copy()
test = data[data['is_train'] == 0].copy()

feature_cols = [col for col in train.columns if col not in ['ID', 'Obtained Electricity', 'is_train', 'Day', 'Weather', 'Hindrance']]
X = train[feature_cols]
y = train['Obtained Electricity']
X_test = test[feature_cols]

scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

def mape_lgb(y_pred, y_true):
    y_true = y_true.get_label()
    mask = y_true != 0
    if mask.sum() == 0:
        return 'mape', 0.0, False
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return 'mape', mape, False

def mape_xgb(y_pred, y_true):
    y_true = y_true.get_label()
    mask = y_true != 0
    if mask.sum() == 0:
        return 'mape', 0.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return 'mape', mape

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))
xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))
cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    lgb_params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 64,
        'max_depth': 8,
        'min_child_samples': 10,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'min_split_gain': 0.01,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val)
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_preds += lgb_model.predict(X_test_scaled) / n_splits
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mape',
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_child_weight': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'gamma': 0.01,
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_val = xgb.DMatrix(X_val, y_val)
    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=2000,
        evals=[(xgb_train, 'train'), (xgb_val, 'valid')],
        custom_metric=mape_xgb,
        early_stopping_rounds=100,
        verbose_eval=200
    )
    xgb_oof[val_idx] = xgb_model.predict(xgb_val)
    xgb_preds += xgb_model.predict(xgb.DMatrix(X_test_scaled)) / n_splits
    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        loss_function='RMSE',
        eval_metric='MAPE',
        random_state=42,
        verbose=200,
        early_stopping_rounds=100,
        task_type='CPU',
        thread_count=-1
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=200)
    cat_oof[val_idx] = cat_model.predict(X_val)
    cat_preds += cat_model.predict(X_test_scaled) / n_splits

mask = y != 0
lgb_mape = mean_absolute_percentage_error(y[mask], lgb_oof[mask]) * 100
xgb_mape = mean_absolute_percentage_error(y[mask], xgb_oof[mask]) * 100
cat_mape = mean_absolute_percentage_error(y[mask], cat_oof[mask]) * 100

meta_features_train = np.column_stack([lgb_oof, xgb_oof, cat_oof])
meta_features_test = np.column_stack([lgb_preds, xgb_preds, cat_preds])

best_alpha = 1.0
best_score = float('inf')
for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(meta_features_train, y)
    meta_pred = ridge.predict(meta_features_train)
    score = mean_absolute_percentage_error(y[mask], meta_pred[mask]) * 100
    if score < best_score:
        best_score = score
        best_alpha = alpha

ridge = Ridge(alpha=best_alpha)
ridge.fit(meta_features_train, y)
stacked_oof = ridge.predict(meta_features_train)
stacked_preds = ridge.predict(meta_features_test)
stacked_mape = mean_absolute_percentage_error(y[mask], stacked_oof[mask]) * 100

if lgb_mape < xgb_mape and lgb_mape < cat_mape:
    ensemble_preds = lgb_preds * 0.5 + xgb_preds * 0.3 + cat_preds * 0.2
elif xgb_mape < lgb_mape and xgb_mape < cat_mape:
    ensemble_preds = xgb_preds * 0.5 + lgb_preds * 0.3 + cat_preds * 0.2
else:
    ensemble_preds = lgb_preds * 0.4 + xgb_preds * 0.35 + cat_preds * 0.25

ensemble_oof = lgb_oof * 0.4 + xgb_oof * 0.35 + cat_oof * 0.25
ensemble_mape = mean_absolute_percentage_error(y[mask], ensemble_oof[mask]) * 100

if stacked_mape < ensemble_mape:
    final_preds = stacked_preds
else:
    final_preds = ensemble_preds

submission = pd.DataFrame({
    'ID': test['ID'],
    'Obtained Electricity': final_preds
})
submission['Obtained Electricity'] = submission['Obtained Electricity'].clip(lower=0)
submission.to_csv('submission.csv', index=False)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance()
}).sort_values('importance', ascending=False)
