Project: Insurance Claim Modeling with XGBoost and Comparators

Overview
--------
This repository contains a working experiment that explores classification and regression models for an insurance claims dataset. The primary goals are:

- Predict whether a policyholder will file a claim (binary classification).
- For policyholders who filed a claim, predict the claim amount (regression).

The main pipeline (in `main.py`) covers data loading, light preprocessing and feature engineering, exploratory data analysis, model comparison, and hyperparameter tuning with a focus on XGBoost for production-ready performance. Several other classical classifiers and regressors are used for benchmarking (Logistic Regression, KNN, Decision Trees, Random Forests, CatBoost, etc.).

Repository structure
--------------------
- `main.py` — End-to-end script that runs the full experiment: data cleaning, EDA, preprocessing pipelines, model evaluation, hyperparameter search, and final evaluation.
- `data.csv` — Source dataset used by the script (expected at repository root).
- `plots/` — Generated visualizations saved by the script (histograms, heatmaps, CV score boxplots, confusion matrix, etc.).
- `catboost_info/` — Artifacts from CatBoost training runs (logs, metrics) captured during experiments.
- `logs.txt` — Optional runtime logs (if created during execution).
- `requirements.txt` — Python dependencies required to run the project.

Key design and processing steps
------------------------------
- Column renaming and basic cleaning: columns are renamed to clearer, snake_case names. Currency-like string columns are stripped of symbols and converted to numeric types. Some categorical values get a removed prefix step.
- Target definitions:
	- Classification target: `is_claim` (binary flag whether a claim occurred).
	- Regression target: `new_claim_value` for rows where a claim occurred.
- Data splitting: Stratified train/test split (stratified on a binned claim-value category) to keep class balance.
- Missing value handling:
	- Numerical: KNN imputation.
	- Categorical: Most-frequent (mode) imputation.
- Encoding:
	- Ordinal encoding for ordered categories (education).
	- Ordinal-style encoding for binary categorical flags.
	- One-hot encoding (with handling for unknowns) for multi-class categoricals.
- Feature transformations:
	- Square-root transform for skewed numeric features (and an experiment with log/sqrt transforms to compare CV performance).
	- Standard scaling applied before model fitting.
	- VIF calculation used to detect multicollinearity and drop highly collinear features.
- Pipelines and reproducibility:
	- Scikit-learn `Pipeline` and `ColumnTransformer` are used to keep preprocessing reproducible and portable.

Modeling and evaluation
-----------------------
- Classification benchmarking: multiple classifiers are cross-validated (10-fold) and compared using accuracy and CV score distributions (boxplots). Models include Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forests, Linear SVM, XGBoost, AdaBoost, Gradient Boosting, Bagging, and CatBoost.
- XGBoost receives focused hyperparameter tuning:
	- A large randomized search across many hyperparameters is run, followed by a smaller grid search to refine the best region.
	- The script reports the best parameters and best CV scores, then evaluates the tuned model on the hold-out test set using F1 and confusion matrix.
- Regression benchmarking: for claim amount prediction, several regressors (Linear Regression, SGDRegressor, Decision Tree Regressor, RandomForestRegressor, KNN Regressor, SVM, XGBRegressor) are cross-validated and compared using RMSE.
- Regression hyperparameter tuning: SGDRegressor is tuned using RandomizedSearchCV and GridSearchCV; best model performance is reported on a test split using RMSE and MAE.

Outputs and artifacts
---------------------
- Saved plots in `plots/` (histograms, heatmaps, CV score comparisons, etc.).
- Printed console outputs with cross-validation scores, best hyperparameters and final evaluation metrics.
- CatBoost training metadata and logs under `catboost_info/`.

How to run
----------
1. Create a Python environment and install dependencies from `requirements.txt`.
2. Place your dataset as `data.csv` in the repository root (script expects this filename).
3. Run the main experiment:

	 python main.py

Notes and assumptions
---------------------
- The script assumes `data.csv` contains specific columns referenced in `main.py`. If your dataset differs, update the column mapping near the top of `main.py`.
- The script writes output plots to the `plots/` folder and may create CatBoost artifacts under `catboost_info/`.
- This README is an overview; the `main.py` file includes the concrete, implementation-level details (imputation strategies, exact hyperparameter ranges, and plotting code).

Next steps / improvements
------------------------
- Split the monolithic `main.py` into reusable modules (data, preprocessing, models, evaluation) for maintainability and testing.
- Add a lightweight entry-point script and CLI flags to run only classification or regression experiments, control hyperparameter search budgets, and toggle plotting.
- Persist model artifacts (pickled pipelines + models) and add a small inference example.

Contact / license
-----------------
This is an experiment-style repository — adapt and reuse as needed. No explicit license is included in the repo.

