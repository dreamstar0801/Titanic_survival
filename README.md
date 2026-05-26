# Titanic Survival Prediction

An end-to-end machine learning pipeline on the Kaggle Titanic dataset, comparing regularized logistic regression, random forests, and gradient boosting. The goal is less to win the leaderboard and more to practice a careful, honest workflow: inspect missingness before imputing, justify each engineered feature, and benchmark models on the same held-out validation set.

## What's in the notebook

1. **Data inspection and missingness analysis.** Checked the missingness pattern for `Age` and tested whether missingness was associated with the survival outcome (independent-samples *t*-test on survival rates between rows with and without `Age`).
2. **Feature engineering.**
   - Extracted passenger titles from the `Name` field via regex and collapsed rare titles into a single `Rare` bucket.
   - Built `Family_Size` (`SibSp + Parch + 1`) and an `IsAlone` indicator.
   - Derived `Deck` from the first character of `Cabin` (with `'U'` for unknown), then folded the single `T` deck into `Other`.
   - Encoded `Sex` numerically and computed `FarePerPerson` to control for group bookings.
3. **Imputation.** KNN imputation (k = 20) on the aligned train/test design matrix, fit on training data only and applied to the test set, to avoid leakage from the holdout into the imputer.
4. **Modeling.** Trained and compared:
   - **Lasso (L1) logistic regression** — for sparse, interpretable baseline coefficients.
   - **Random Forest** with grid-search hyperparameter tuning.
   - **XGBoost** with early stopping on a validation split.
5. **Evaluation.** Cross-validated accuracy, model-vs-model comparison on the same validation set, and feature importance for both tree-based models to interpret which engineered features actually mattered.

## Tech stack

Python, pandas, NumPy, scikit-learn (KNNImputer, LogisticRegression, RandomForestClassifier, GridSearchCV), XGBoost, seaborn, matplotlib, missingno, SciPy.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook titanic-prediction-1.ipynb
```

The notebook expects the standard Kaggle Titanic files at:

- `train.csv`
- `test.csv`
- `gender_submission.csv` (used only as a submission template)

If you're running outside Kaggle, update the file paths in the first data-loading cell.

## What I'd improve next

- Stratified K-fold CV across the full pipeline (imputation included) rather than a single train/validation split, so the imputer's variance is reflected in the CV error.
- A small ablation study isolating the contribution of each engineered feature (title, deck, fare-per-person, is-alone) — useful for understanding which features actually move the needle versus which are along for the ride.
- A calibration check (reliability diagram) for the gradient-boosted model, since raw XGBoost probabilities are often poorly calibrated even when accuracy is high.

## About

Built by Sohee Cho as a portfolio project to practice clean ML workflow design — feature engineering with explicit reasoning, leakage-aware preprocessing, and head-to-head model comparison rather than just chasing leaderboard score.
