# Random Forest Regressor --- Complete Notes

## 1. What is Random Forest?

Random Forest is an ensemble model made of many Decision Trees. For
regression, the forest generally averages the predictions from its
trees.

If three trees predict 25, 28, and 24 MPG:

\[ `\text{prediction}`{=tex} = (25 + 28 + 24) / 3 = 25.67
`\text{ MPG}`{=tex} \]

With `n_estimators=100`, the forest combines predictions from 100 trees.

## 2. What stays the same as a Decision Tree?

Inside each tree: - It evaluates candidate feature thresholds. - It
chooses a split that reduces squared error. - It repeats splitting at
later nodes. - Each leaf predicts the average target of training rows
that reach it.

The basic decision-tree splitting mechanism is the same. Random Forest
changes how the trees are trained and which features each split can
consider.

## 3. How does Random Forest make trees different?

### A. Bootstrap sampling (rows)

Each tree generally receives its own bootstrap sample of training rows,
sampled **with replacement**.

That means: - Some rows can appear more than once in a tree's sample. -
Some rows may not appear in that sample. - Different trees see different
samples.

### B. Random feature selection (features)

We provide the model with all selected input columns. We do **not**
manually assign specific features to each tree.

At each split, the algorithm randomly selects a subset of features to
consider, controlled by `max_features`. It then tests candidate
thresholds for the selected features and chooses the best split among
those candidates.

For example, if the dataset has `horsepower`, `weight`, `displacement`,
and `model_year`, a node might consider horsepower and weight. It
evaluates thresholds for both, but **one split at that node uses one
feature and one threshold**. Another node may consider a different
subset.

A path through a tree can contain multiple conditions at different
nodes, such as: - `weight <= 2500` - later, `horsepower <= 80`

Those are separate splits, not a single split combining two features.

## 4. Train a Random Forest with horsepower only

``` python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X = df[["horsepower"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Random Forest R²:", r2_score(y_test, y_pred))
```

## 5. Train with all features

``` python
X = df.drop(columns=["mpg"])
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Random Forest R²:", r2_score(y_test, y_pred))
```

Changing from horsepower-only to all features provides more information
to the model. In our session, the horsepower-only Random Forest test R²
was about 0.5645, and the all-feature version reached about 0.8869.
These are results from that particular split and setup, not guaranteed
scores.

## 6. Important hyperparameters

  -----------------------------------------------------------------------
  Hyperparameter                      Meaning
  ----------------------------------- -----------------------------------
  `n_estimators`                      Number of trees

  `max_depth`                         Maximum depth of each tree

  `max_features`                      Number/fraction of features
                                      considered at each split

  `min_samples_split`                 Minimum samples needed to split a
                                      node

  `min_samples_leaf`                  Minimum samples required in a leaf

  `random_state`                      Controls randomness for
                                      reproducibility
  -----------------------------------------------------------------------

### `max_features` examples

The exact accepted options depend on the estimator and scikit-learn
version, but common regression settings include: - `1.0`: all features
can be considered at each split. - An integer: that many features are
considered. - A fraction: that fraction of features is considered.

Random Forest can still create different trees through bootstrap
sampling even when all features are considered at every split.

## 7. Hyperparameter tuning with validation data

Hyperparameter tuning means comparing settings and selecting the ones
that work best on validation data.

Example: compare the number of trees while keeping other settings fixed.

``` python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

for n in [10, 50, 100, 200]:
    model = RandomForestRegressor(
        n_estimators=n,
        random_state=42
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    print(
        f"n_estimators={n}, "
        f"validation R²={r2_score(y_val, val_pred):.4f}"
    )
```

This assumes a train/validation/test split already exists. Choose
settings based on validation performance, then evaluate the selected
model once on the test set.

You can tune more than one hyperparameter, but change a few settings at
a time so you can interpret the results.

## 8. Feature importance

Random Forest's `feature_importances_` provides impurity-based feature
importance. It summarizes how much each feature contributes to reducing
split impurity across the forest, normalized so the values sum to 1.

``` python
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values()

importance.plot(kind="barh")
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.show()
```

Interpretation: - A larger bar means the feature contributed more to the
model's impurity reductions. - It does **not** mean that feature alone
explains that fraction of MPG. - It does **not** prove causation. -
Correlated features can affect each other's importance. - Impurity-based
importance can favor features with many possible split points.

In our plot, displacement had the largest importance (about 0.42),
followed by horsepower, weight, and cylinders. These are approximate
readings from our particular trained model.

## 9. Evaluation metrics

``` python
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

y_pred = rf_model.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

-   R²: higher is generally better; 1 is perfect. It can be negative on
    test data.
-   MAE: average absolute prediction error, in MPG here. Lower is
    better.
-   RMSE: square root of mean squared error; it penalizes larger errors
    more heavily than MAE. Lower is better.

## 10. Why Random Forest can perform better

A single tree may be unstable: small changes in data can change its
splits. Averaging many diverse trees often reduces variance and improves
generalization.

It is not guaranteed to outperform every other model. Results depend on
features, data, hyperparameters, and the train/test split.

## 11. Strengths and limitations

**Strengths** - Handles nonlinear patterns and feature interactions. -
Usually more stable than one decision tree. - Often works well without
feature scaling. - Provides a feature-importance measure.

**Limitations** - Larger and slower than a single tree. - Less
interpretable than one small tree. - Feature importance is not causal
evidence. - May not extrapolate beyond the target patterns represented
in training data.
