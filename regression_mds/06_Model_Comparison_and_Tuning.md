# Model Comparison, Metrics, and Hyperparameter Tuning

## 1. Evaluation metrics

### R² (coefficient of determination)

R² measures how well predictions explain variation in the target
relative to predicting the target mean.

-   1.0: perfect predictions.
-   0.0: no improvement over predicting the test target mean under the
    R² definition.
-   Negative: worse than that baseline.

R² is not literally the percentage of predictions that are correct.

### MAE (Mean Absolute Error)

\[ MAE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}\^{n}\|y_i-`\hat{y}`{=tex}\_i\| \]

MAE is the average absolute prediction error. In Auto MPG, it is
measured in MPG. Lower is better.

### RMSE (Root Mean Squared Error)

\[ RMSE = `\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}`{=tex} \]

RMSE also uses MPG units. Squaring the errors before averaging makes
large errors count more heavily. Lower is better.

## 2. Calculate all three metrics

``` python
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

Replace `model` with the fitted estimator you want to evaluate.

## 3. Hyperparameter tuning workflow

1.  Decide which hyperparameters to test.
2.  Fit each candidate model on training data.
3.  Score each candidate on validation data.
4.  Select the best setting based on the chosen metric.
5.  Retrain the selected configuration on training + validation data.
6.  Evaluate once on the untouched test set.

## 4. Example: tune Decision Tree depth

``` python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
results = []

for depth in depths:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    score = r2_score(y_val, model.predict(X_val))
    results.append((depth, score))
    print(depth, score)

best_depth, best_val_score = max(results, key=lambda item: item[1])
print("Best depth:", best_depth)
print("Best validation R²:", best_val_score)
```

## 5. Example: tune Random Forest tree count

``` python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

results = []

for n in [10, 50, 100, 200]:
    model = RandomForestRegressor(
        n_estimators=n,
        random_state=42
    )
    model.fit(X_train, y_train)
    score = r2_score(y_val, model.predict(X_val))
    results.append((n, score))
    print(n, score)

best_n, best_val_score = max(results, key=lambda item: item[1])
print("Best n_estimators:", best_n)
print("Best validation R²:", best_val_score)
```

These are simple examples, not exhaustive searches. Cross-validation can
make model selection more robust when data is limited.

## 6. Final fit and test

After choosing settings, combine training and validation rows, fit the
chosen model, then evaluate on test data:

``` python
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

final_model = RandomForestRegressor(
    n_estimators=best_n,
    random_state=42
)

final_model.fit(X_train_val, y_train_val)
test_pred = final_model.predict(X_test)

print("Test R²:", r2_score(y_test, test_pred))
print("Test MAE:", mean_absolute_error(y_test, test_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))
```

Use a model-specific final estimator and its selected hyperparameters.
This example assumes the selected model is Random Forest and `best_n`
came from the tuning step.

## 7. Avoid these mistakes

-   Tuning hyperparameters on the test set.
-   Comparing models evaluated on different test rows.
-   Using target information as an input feature.
-   Assuming the highest training score means the best model.
-   Assuming one random split proves that a model will always perform
    best.
-   Treating feature importance as proof of causation.
