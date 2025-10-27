import os
os.makedirs('plots', exist_ok=True)
_plot_idx = 1

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, make_scorer, confusion_matrix, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from sklearn import set_config

raw_data = pd.read_csv('data.csv')
pd.set_option('display.max_columns', None)
raw_data.head()
data_df = raw_data.copy()
col_names = {
    'KIDSDRIV': 'num_young_drivers',
    'BIRTH': 'date_of_birth',
    'AGE': 'age',
    'HOMEKIDS': 'num_of_children',
    'YOJ': 'years_job_held_for',
    'INCOME': 'income',
    'PARENT1': 'single_parent',
    'HOME_VAL': 'value_of_home',
    'MSTATUS': 'married',
    'GENDER': 'gender',
    'EDUCATION': 'highest_education',
    'OCCUPATION': 'occupation',
    'TRAVTIME': 'commute_dist',
    'CAR_USE': 'type_of_use',
    'BLUEBOOK': 'vehicle_value',
    'TIF': 'policy_tenure',
    'CAR_TYPE': 'vehicle_type',
    'RED_CAR': 'red_vehicle',
    'OLDCLAIM': '5_year_total_claims_value',
    'CLM_FREQ': '5_year_num_of_claims',
    'REVOKED': 'licence_revoked',
    'MVR_PTS': 'license_points',
    'CLM_AMT': 'new_claim_value',
    'CAR_AGE': 'vehicle_age',
    'CLAIM_FLAG': 'is_claim',
    'URBANICITY': 'address_type'
}
data_df.rename(columns=col_names, inplace=True)
data_df.info()
data_df.duplicated().sum()
data_df.drop_duplicates(inplace=True)
currency_cols = ['income', 'value_of_home', 'vehicle_value', '5_year_total_claims_value', 'new_claim_value']
def format_currency_cols(data, cols):
    for col in cols:
        data[col] = data[col].replace('[\\$,]', '', regex=True).astype('Int64')
    return data
data_df = format_currency_cols(data_df, currency_cols)
z_prefix_cols = ['married', 'gender', 'highest_education', 'occupation', 'vehicle_type', 'address_type']
def remove_prefix(data, cols):
    for col in cols:
        data[col] = data[col].replace('[z_]', '', regex=True)
    return data
data_df = remove_prefix(data_df, z_prefix_cols)
data_df.drop(['ID', 'date_of_birth'], axis=1, inplace=True)
data_df.head()
mask = (data_df['new_claim_value'] > 0) & (data_df['is_claim'] == 0)
data_df[mask]
sns.histplot(data_df['new_claim_value'], bins=10)
bins = [0.0, 5000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000, np.inf]
labels = np.arange(1, 12)
data_df['claim_value_cat'] = pd.cut(data_df['new_claim_value'], bins = bins, labels= labels, include_lowest=True)
sns.barplot(data_df['claim_value_cat'])
X = data_df.copy()
y = data_df['is_claim']
X.drop(columns=['new_claim_value','is_claim'], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['claim_value_cat'])
train_ratio = round((y_train.sum()/len(y_train))*100,2)
test_ratio = round((y_test.sum()/len(y_test)*100),2)
print(f'Train positive class ratio: {train_ratio}%')
print(f'Test positive class ratio: {test_ratio}%')
for set_ in (X_train, X_test):
    set_.drop(columns=['claim_value_cat'], inplace=True)
eda_test_data = X_train.copy()
eda_test_data['is_claim'] = y_train
binary_cols = ['single_parent', 'married', 'gender', 'red_vehicle', 'licence_revoked']
mapping = {'Yes': 1,
           'No': 0,
           'yes': 1,
           'no': 0,
           'M': 1,
           'F': 0,
           }
def binarise_values(data, cols, map):
    for col in cols:
        data[col] = data[col].map(map)
    return data
eda_test_data = binarise_values(eda_test_data, binary_cols, mapping)
eda_test_data.head()
eda_test_data.corr(numeric_only=True).sort_values(by='is_claim',ascending=False)
fig, ax = plt.subplots(figsize=(4, 8))
corr_matrix = eda_test_data.corr(numeric_only=True).sort_values(by='is_claim',ascending=False)
corr_matrix_no_claim = corr_matrix.drop('is_claim')
sns.heatmap(corr_matrix_no_claim[['is_claim']],cmap='coolwarm', annot=True, vmax=0.25, vmin=-0.25)
X_train_raw = X_train.copy()
cols_to_drop = [
    'red_vehicle',
]
X_train_raw.drop(columns=cols_to_drop, axis=1, inplace=True)
X_train_raw.isnull().sum().sum()
X_train_raw.isnull().sum()
knn_imputer = KNNImputer(n_neighbors=2)
numerical_cols_df = X_train_raw.select_dtypes(include=['number'])
numerical_cols = numerical_cols_df.columns.tolist()
cat_cols_df = X_train_raw.select_dtypes(include=['object'])
cat_cols = cat_cols_df.columns.tolist()
def num_knn_impute(data, cols, imputer):
    data = data[cols]
    data_imputed = pd.DataFrame(imputer.fit_transform(data))
    data_imputed.columns = data.columns
    return data_imputed
num_test_data_imputed = num_knn_impute(X_train_raw, numerical_cols, knn_imputer)
num_cols_df = X_train_raw[numerical_cols].reset_index(drop=True)
missing_data_df = num_cols_df[num_cols_df.isna().any(axis=1)]
missing_data_df.head()
samples = missing_data_df.index.to_list()
num_test_data_imputed.loc[samples].head()
simple_imputer = SimpleImputer(strategy='most_frequent')
def cat_simple_imputer(data, cols, imputer):
    data = data[cols]
    data_imputed = pd.DataFrame(imputer.fit_transform(data))
    data_imputed.columns = data.columns
    return data_imputed
cat_test_data_imputed = cat_simple_imputer(X_train_raw, cat_cols, simple_imputer)
cat_cols_df = X_train_raw[cat_cols].reset_index(drop=True)
missing_cat_data_df = cat_cols_df[num_cols_df.isna().any(axis=1)]
missing_cat_data_df.head()
samples = missing_cat_data_df.index.to_list()
cat_test_data_imputed.loc[samples].head()
train_imputed_df = pd.concat([num_test_data_imputed, cat_test_data_imputed], axis=1)
train_imputed_df.head()
train_imputed_df.isnull().sum()
cat_test_data_imputed.nunique()
cat_cols_ord = ['highest_education']
cat_cols_bin = ['single_parent', 'married', 'gender', 'type_of_use', 'licence_revoked', 'address_type']
cat_cols_one_hot = ['occupation', 'vehicle_type']
education_rank = [['<High School', 'High School', 'Bachelors', 'Masters', 'PhD']]
ord_encoder = OrdinalEncoder(categories=education_rank)
bin_encoder = OrdinalEncoder()
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ord_encoded = ord_encoder.fit_transform(cat_test_data_imputed[cat_cols_ord])
bin_encoded = bin_encoder.fit_transform(cat_test_data_imputed[cat_cols_bin])
one_hot_encoded = one_hot_encoder.fit_transform(cat_test_data_imputed[cat_cols_one_hot])
ord_encoded_df = pd.DataFrame(ord_encoded)
ord_encoded_df.columns = cat_cols_ord
bin_encoded_df = pd.DataFrame(bin_encoded)
bin_encoded_df.columns = cat_cols_bin
one_hot_encoded_df = pd.DataFrame(one_hot_encoded)
one_hot_encoded_df.columns = one_hot_encoder.get_feature_names_out()
all_cat_encoded_df = pd.concat([ord_encoded_df, bin_encoded_df, one_hot_encoded_df], axis=1)
all_cat_encoded_df.head()
X_train_cleaned = pd.concat([train_imputed_df[numerical_cols], all_cat_encoded_df], axis=1)
X_train_cleaned.head()
def calculate_vif(dataframe):
    df_with_constant = add_constant(dataframe)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_with_constant.columns
    vif_data["VIF"] = [variance_inflation_factor(df_with_constant.values, i)
                       for i in range(df_with_constant.shape[1])]
    return vif_data
vif_df = calculate_vif(X_train_cleaned)
vif_df
X_train_cleaned.drop(['occupation_Blue Collar' ,'vehicle_type_Minivan'], axis=1, inplace=True)
vif_df = calculate_vif(X_train_cleaned)
vif_df
clfs = [
    ('Logistic Regression', LogisticRegression(solver='liblinear', max_iter=2000)),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Linear SVM', LinearSVC(random_state=42, max_iter=1000, dual='auto')),
    ('XGBoost', XGBClassifier(random_state=42)),
    ('AdaBoost', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
    ('Gradient Boost', GradientBoostingClassifier(random_state=42)),
    ('Bagging', BaggingClassifier(random_state=42)),
    ('CatBoost', CatBoostClassifier(random_state=42, verbose=0)),
]
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}
for clf_name, clf in clfs:
    cv_scores = cross_val_score(clf, X_train_cleaned, y_train, cv=kf)
    results[clf_name] = cv_scores
cv_scores_df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(cv_scores_df)
ax.set_xlabel('Classifier', fontsize=12)
ax.set_ylabel('CV Accuracy Score', fontsize=12)
ax.set_title('Cross-Validation Scores for Different Classifiers', fontsize=14)
melted_X_train = X_train_cleaned[numerical_cols].melt(var_name='Column', value_name='Value')
g = sns.FacetGrid(melted_X_train, col='Column', col_wrap=4, sharex=False, sharey=False, height=4)
g.map(sns.histplot, 'Value', bins=25)
g.set_axis_labels('Value', 'Frequency')
g.set_titles(col_template='{col_name}')
plt.tight_layout()
plt.savefig(os.path.join('plots', f'plot_{_plot_idx}.png'))
_plot_idx += 1
plt.close()

skewed_features = ['income', 'value_of_home', 'commute_dist', 'vehicle_value', 'policy_tenure', 'license_points']
def log_of_feature(data_df, skewed_features):
    data = data_df.copy()
    for feature in skewed_features:
        data[feature] = np.sqrt(data[feature])
    return data
X_train_cleaned_log = log_of_feature(X_train_cleaned, skewed_features)
melted_X_train = X_train_cleaned_log[numerical_cols].melt(var_name='Column', value_name='Value')
g = sns.FacetGrid(melted_X_train, col='Column', col_wrap=4, sharex=False, sharey=False, height=4)
g.map(sns.histplot, 'Value', bins=25)
g.set_axis_labels('Value', 'Frequency')
g.set_titles(col_template='{col_name}')
plt.tight_layout()
plt.savefig(os.path.join('plots', f'plot_{_plot_idx}.png'))
_plot_idx += 1
plt.close()

xgb_boost_clf = XGBClassifier(random_state=42)
cv_scores = cross_val_score(xgb_boost_clf, X_train_cleaned, y_train, cv=kf)
cv_scores_log = cross_val_score(xgb_boost_clf, X_train_cleaned_log, y_train, cv=kf)
print(f'CV score without log transform: {cv_scores.mean()}')
print(f'CV score with log transform: {cv_scores_log.mean()}')
def scale_features(data_df, numeric_features):
    data = data_df.copy()
    scaler = StandardScaler()
    scaler.fit(data[numeric_features])
    data[numeric_features] = scaler.transform(data[numeric_features])
    return data
X_train_cleaned_scaled = scale_features(X_train_cleaned_log, numerical_cols)
xgb_boost_clf = XGBClassifier(random_state=42)
cv_scores = cross_val_score(xgb_boost_clf, X_train_cleaned, y_train, cv=kf)
cv_scores_scaled = cross_val_score(xgb_boost_clf, X_train_cleaned_scaled, y_train, cv=kf)
print(f'CV score without log transform: {cv_scores.mean()}')
print(f'CV score with log transform: {cv_scores_scaled.mean()}')
X_train_cleaned = X_train_cleaned_scaled.copy()
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)
    def get_feature_names_out(self, input_features=None):
        return None
class SqrtTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_transform):
        self.columns_to_transform = columns_to_transform
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X[self.columns_to_transform] = np.sqrt(X[self.columns_to_transform])
        return X
    def get_feature_names_out(self, input_features=None):
        return input_features
set_config(transform_output='pandas')
cols_to_drop_pipeline = Pipeline([
    ('col_dropper', ColumnDropper(cols_to_drop))
])
skewed_features = ['income', 'value_of_home', 'commute_dist', 'vehicle_value', 'policy_tenure', 'license_points']
num_pipeline = Pipeline([
    ('knn_imputer', KNNImputer(n_neighbors=2)),
    ('sqrt', SqrtTransformer(skewed_features)),
    ('scaler', StandardScaler()),
])
education_rank = [['<High School', 'High School', 'Bachelors', 'Masters', 'PhD']]
cat_ord_pipeline = Pipeline([
    ('simple_imputer', SimpleImputer(strategy='most_frequent')),
    ('ord_encoder', OrdinalEncoder(categories=education_rank)),
])
cat_bin_pipeline = Pipeline([
    ('simple_imputer', SimpleImputer(strategy='most_frequent')),
    ('binary_encoder', OrdinalEncoder()),
])
cat_one_hot_pipeline = Pipeline([
    ('cat_simple_imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')),
])
preprocess_pipeline = ColumnTransformer([
    ('drop_features', cols_to_drop_pipeline, cols_to_drop),
    ('num', num_pipeline, numerical_cols),
    ('cat_ord', cat_ord_pipeline, cat_cols_ord),
    ('cat_bin', cat_bin_pipeline, cat_cols_bin),
    ('cat_one_hot', cat_one_hot_pipeline, cat_cols_one_hot),
])
X_train_prepared = preprocess_pipeline.fit_transform(X_train)
X_train_prepared_df = pd.DataFrame(
    X_train_prepared,
    columns=preprocess_pipeline.get_feature_names_out(),
)
one_hot_col_names = list(preprocess_pipeline.transformers_[4][1][1].get_feature_names_out(cat_cols_one_hot))
new_col_names = numerical_cols + cat_cols_ord + cat_cols_bin + one_hot_col_names
X_train_prepared_df.columns = new_col_names
X_train_prepared_df.reset_index(drop=True, inplace=True)
X_train_prepared_df.head()
X_train_cleaned.head()
X_train_cleaned.equals(X_train_prepared_df)
xgb_param_grid = {
    'n_estimators': randint(low=50, high=300),
    'learning_rate':uniform(0.01, 0.29),
    'max_depth': randint(low=1, high = 20),
    'subsample': uniform(0, 1),
    'colsample_bytree': uniform(0, 1),
    'min_child_weight': randint(low=1, high= 20),
    'reg_alpha': randint(low=0, high=100),
    'reg_lambda':randint(low=0, high=10),
    'gamma': uniform(0, 1),
}
xgb_model = XGBClassifier(random_state=42, eval_metric='error')
scorer = make_scorer(f1_score, average='weighted')
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    n_iter=2000,
    scoring=scorer,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train_prepared, y_train)
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)
xgb_param_grid_grid_search = {
    'n_estimators': [280, 290, 300],
    'max_depth': [6, 7, 8],
    'learning_rate': [0.03, 0.04, 0.05],
    'subsample': [0.45, 0.5, 0.55],
    'colsample_bytree': [0.5, 0.55, 0.6],
    'gamma': [0.05, 0.1, 0.15],
    'min_child_weight': [18],
    'reg_alpha': [1],
    'reg_lambda': [1],
}
xgb_model = XGBClassifier(random_state=42, eval_metric='error')
scorer = make_scorer(f1_score, average='weighted')
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid_grid_search,
    scoring=scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_prepared, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)
print(f'XGBoost model score (default hyperparameters): {cv_scores.mean()}')
print(f'XGBoost model score (tuned hyperparamters: {grid_search.best_score_}')
X_test_prepared = preprocess_pipeline.fit_transform(X_test)
y_pred = grid_search.best_estimator_.predict(X_test_prepared)
f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix on Training Data')
claim_data = data_df[data_df['new_claim_value'] > 0]
X_reg = claim_data.copy()
y_reg = claim_data['new_claim_value']
X_reg.drop(columns=['new_claim_value','is_claim', 'claim_value_cat'], inplace=True)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_reg_train_prepared = preprocess_pipeline.fit_transform(X_reg_train)
regs = [
    ('Linear Regression', LinearRegression()),
    ('Logistic Regression', LogisticRegression(random_state=42, max_iter=25, solver='sag', tol=3)),
    ('SGD Regressor', SGDRegressor(random_state=42)),
    ('Decision Tree Regression ', DecisionTreeRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('KNN Model ', KNeighborsRegressor()),
    ('Support Vector Machines (SVM)', SVR(gamma=2, C=1)),
    ('XGBRegressor', XGBRegressor(random_state=42))
]
reg_kf = KFold(n_splits=10, shuffle=True, random_state=42)
reg_results = {}
for reg_name, reg in regs:
    cv_rmses = -cross_val_score(reg, X_reg_train_prepared, y_reg_train, cv=reg_kf, scoring='neg_root_mean_squared_error')
    reg_results[reg_name] = cv_rmses
reg_cv_scores_df = pd.DataFrame(reg_results)
fig, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(reg_cv_scores_df)
ax.set_xlabel('Regressor', fontsize=12)
ax.set_ylabel('CV RMSE', fontsize=12)
ax.set_title('Cross-Validation Scores for Different Regressors', fontsize=14)
reg_param_grid = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate': ['constant', 'invscaling'],
    'eta0': uniform(0.001, 0.1),
    'max_iter': randint(100, 1000),
    'tol': uniform(1e-6, 1e-3)
}
sgd_regressor = SGDRegressor(random_state=42)
reg_random_search = RandomizedSearchCV(
    estimator=sgd_regressor,
    param_distributions=reg_param_grid,
    n_iter=500,
    scoring='neg_mean_squared_error',
    cv=reg_kf,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_reg_train_prepared, y_reg_train)
score = np.sqrt(-random_search.best_score_)
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", score)
reg_param_grid_gs = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.004, 0.008, 0.012],
    'learning_rate': ['invscaling'],
    'eta0': [0.001, 0.003, 0.005],
    'max_iter': [180, 200, 220],
    'tol': [1e-5, 1e-4, 1e-6]
}
sgd_regressor = SGDRegressor(random_state=42)
reg_grid_search = GridSearchCV(
    estimator=sgd_regressor,
    param_grid=reg_param_grid_gs,
    scoring='neg_mean_squared_error',
    cv=reg_kf,
    verbose=1,
    n_jobs=-1
)
reg_grid_search.fit(X_reg_train_prepared, y_reg_train)
reg_score = np.sqrt(-reg_grid_search.best_score_)
print("Best parameters found: ", reg_grid_search.best_params_)
print("Best cross-validation score: ", reg_score)
X_reg_test_prepared = preprocess_pipeline.fit_transform(X_reg_test)
y_reg_pred = reg_grid_search.best_estimator_.predict(X_reg_test_prepared)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
