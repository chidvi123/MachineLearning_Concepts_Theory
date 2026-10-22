# Regression Models --- Complete A-to-Z Notebook Guide

> This is the main reference for our Auto MPG regression work. It brings
> together the workflow, Linear Regression, Polynomial Regression,
> Decision Tree Regression, Random Forest Regression, evaluation
> metrics, validation, hyperparameter tuning, plotting, common mistakes,
> and a study checklist.

## Table of contents

1.  Regression basics
2.  Dataset and data cleaning
3.  Features and target
4.  Train, validation, and test splits
5.  Evaluation metrics
6.  Linear Regression
7.  Polynomial Regression
8.  Decision Tree Regressor
9.  Random Forest Regressor
10. Hyperparameter tuning
11. Feature importance
12. Model comparison
13. Common mistakes
14. Quick reference table
15. Final checklist

------------------------------------------------------------------------

## 1. Regression basics

**Regression** predicts a continuous numeric value. In our example, the
target is `mpg` (miles per gallon).

Examples of regression targets: - House price - Temperature - Sales
revenue - Fuel efficiency

Classification predicts categories; regression predicts numeric
quantities.

### Vocabulary

-   **Feature (`X`)**: input information used for prediction.
-   **Target (`y`)**: the value being predicted.
-   **Fit / train**: learn a model from training data.
-   **Predict**: produce outputs for input rows.
-   **Parameter**: a value learned during training.
-   **Hyperparameter**: a setting chosen by the developer, such as tree
    depth.
-   **Generalization**: performance on unseen data.
-   **Overfitting**: the model learns training-specific details that do
    not generalize.
-   **Underfitting**: the model is too simple to capture useful
    patterns.

## 2. Dataset and cleaning

Our dataset columns are:

  Column           Meaning
  ---------------- --------------------------
  `mpg`            Miles per gallon; target
  `cylinders`      Number of cylinders
  `displacement`   Engine displacement
  `horsepower`     Engine horsepower
  `weight`         Vehicle weight
  `acceleration`   Acceleration measure
  `model_year`     Model year
  `origin`         Origin code

Load the no-header CSV and handle missing horsepower values:

``` python
import pandas as pd

columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin"
]

df = pd.read_csv(
    "auto-mpg.csv",
    header=None,
    names=columns,
    na_values="?"
)

df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna().copy()

print(df.info())
print(df.head())
```

Why do this? - The file has no column headers, so we provide them. - `?`
indicates missing values. - `horsepower` must be numeric. - Rows with
missing values are removed in this simple workflow.

After cleaning, our session's dataset had 392 rows.

## 3. Select the input features and target

### One feature

``` python
X = df[["horsepower"]]
y = df["mpg"]
```

This tests how well horsepower alone predicts MPG.

### All features

``` python
X = df.drop(columns=["mpg"])
y = df["mpg"]
```

This gives the model all available predictor columns. `mpg` must not be
included in `X`, because it is the answer we are trying to predict.

The choice of features can strongly affect model performance. In our
session, Random Forest scored about 0.5645 R² using horsepower only,
then about 0.8869 with all features on the corresponding experiment's
test split.

## 4. Split the data

### Simple train/test split

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

-   Training set: model learns from these rows.
-   Test set: used for final evaluation.
-   `test_size=0.2`: 20% is reserved for testing.
-   `random_state=42`: makes the random split reproducible.

The features and target are split together so each row stays paired with
its correct MPG value.

### Train/validation/test split

Use this when tuning hyperparameters:

``` python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)
```

For 392 cleaned rows, this gives approximately: - Training: 234 rows -
Validation: 79 rows - Test: 79 rows

Why? The first split holds out 79 test rows. Of the remaining 313 rows,
25% rounds up to 79 validation rows, leaving 234 training rows.

### Why not tune using test data?

If we repeatedly choose settings based on the test score, our decisions
become indirectly tailored to the test set. Keep validation data for
choosing settings and reserve test data for a final check.

## 5. Regression evaluation metrics

Use multiple metrics to understand model performance.

### R²

R² measures how well predictions explain variation in the target
compared with a baseline that predicts the test target mean.

-   1: perfect predictions.
-   0: equivalent to the mean-prediction baseline under the R²
    definition.
-   Negative: worse than that baseline.

R² is not the percentage of individual predictions that are correct.

### MAE

Mean Absolute Error is the average absolute difference between actual
and predicted values.

\[ MAE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}\^{n}\|y_i-`\hat{y}`{=tex}\_i\| \]

Lower is better. In this dataset, the unit is MPG.

### RMSE

Root Mean Squared Error is the square root of average squared error.

\[ RMSE = `\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}`{=tex} \]

Lower is better. RMSE penalizes large errors more strongly than MAE.

### Code for all metrics

``` python
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

Replace `model` with whichever fitted model you want to evaluate.

------------------------------------------------------------------------

## 6. Linear Regression

### Idea

Linear Regression learns a straight-line relationship between input and
target.

With one feature:

\[ `\hat{y}`{=tex} = b_0 + b_1x \]

-   (b_0): intercept.
-   (b_1): coefficient.
-   (x): feature, such as horsepower.
-   (`\hat{y}`{=tex}): predicted MPG.

The model learns coefficients that minimize the sum of squared residuals
on training data.

A residual is actual value minus predicted value.

### Code: horsepower only

``` python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[["horsepower"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

print("Intercept:", linear_model.intercept_)
print("Coefficient:", linear_model.coef_)
print("Linear Regression R²:", r2_score(y_test, y_pred))
```

### How to interpret it

`fit()` learns the intercept and coefficient. `predict()` uses them to
produce MPG estimates. The coefficient describes the fitted change in
prediction per one-unit change in the feature; it does not prove
causation.

### Plot the line

``` python
import matplotlib.pyplot as plt

X_sorted = X.sort_values("horsepower")
plt.scatter(X["horsepower"], y, alpha=0.6)
plt.plot(X_sorted["horsepower"], linear_model.predict(X_sorted), color="red")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Linear Regression")
plt.show()
```

The points are observed cars; the red line is the fitted model.

### Strengths and limitations

Strengths: fast, simple, interpretable baseline.

Limitations: a straight line may miss curved relationships; outliers and
correlated features can affect the fit or interpretation.

------------------------------------------------------------------------

## 7. Polynomial Regression

### Idea

Polynomial Regression creates powers of the input so a linear model can
fit a curved relationship.

For degree 2:

\[ `\hat{y}`{=tex} = b_0 + b_1x + b_2x\^2 \]

It is nonlinear in the original input (x), but the model remains linear
in its coefficients.

### Code: degree 2

``` python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[["horsepower"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)

print("Polynomial Regression R²:", r2_score(y_test, y_pred))
```

### What happens in the pipeline?

1.  `PolynomialFeatures` adds `horsepower²`.
2.  `LinearRegression` learns coefficients for horsepower and horsepower
    squared.
3.  The pipeline applies the same transformation during prediction.

### Plot the curve

``` python
X_sorted = X.sort_values("horsepower")

plt.scatter(X["horsepower"], y, alpha=0.6)
plt.plot(X_sorted["horsepower"], poly_model.predict(X_sorted), color="red")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression (Degree 2)")
plt.show()
```

Sort inputs before drawing the curve so the line follows increasing
horsepower.

### Choosing degree

Degree is a hyperparameter. Compare degrees using validation data.
Higher degrees are more flexible but may overfit. A degree-2 model is
not guaranteed to beat degree 1 on unseen data.

------------------------------------------------------------------------

## 8. Decision Tree Regressor

### Idea

A Decision Tree divides the feature space using threshold-based
conditions. At a node it might ask:

`horsepower <= 84.5?`

The threshold is learned from the training data.

### How a split is selected

For regression, a tree generally evaluates candidate features and
thresholds, calculates the weighted mean squared error (MSE) of the
resulting child nodes, and chooses the split that reduces error the
most.

The process repeats at child nodes.

### What does a leaf predict?

A regression leaf predicts the average target of training samples that
reach it. For example, if the leaf contains MPG values 20, 22, and 24,
it predicts 22.

### Code

``` python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

X = df[["horsepower"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("Decision Tree R²:", r2_score(y_test, y_pred))
```

### Tree visualization

``` python
from sklearn.tree import plot_tree

plt.figure(figsize=(16, 8))
plot_tree(
    dt_model,
    feature_names=X.columns,
    filled=True,
    rounded=True
)
plt.show()
```

-   `samples`: training rows that reach a node.
-   `value`: mean target at that node.
-   `squared_error`: target variation at the node.
-   `feature <= threshold`: decision condition.

### Prediction curve

``` python
X_sorted = X.sort_values("horsepower")
plt.scatter(X["horsepower"], y, alpha=0.6)
plt.plot(X_sorted["horsepower"], dt_model.predict(X_sorted), color="red")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Decision Tree Regression")
plt.show()
```

The line is step-like because all points in the same leaf get the same
prediction.

### Hyperparameters and overfitting

-   `max_depth`: maximum tree depth.
-   `min_samples_split`: minimum rows needed to split a node.
-   `min_samples_leaf`: minimum rows required in a leaf.
-   `max_features`: features considered when searching for each split.

A deep tree can overfit. Tune its settings using validation data.

### Tune depth

``` python
depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
best_depth = None
best_score = float("-inf")

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    score = r2_score(y_val, model.predict(X_val))
    print(depth, score)

    if score > best_score:
        best_score = score
        best_depth = depth

print("Best depth:", best_depth)
print("Best validation R²:", best_score)
```

This requires a train/validation/test split. Choose the depth on
validation data, then retrain using training + validation data and
evaluate once on the test set.

------------------------------------------------------------------------

## 9. Random Forest Regressor

### Idea

Random Forest is an ensemble of many Decision Trees. For regression, it
averages their predictions.

If three trees predict 25, 28, and 24 MPG:

\[ `\frac{25+28+24}{3}`{=tex}=25.67 \]

A forest with 100 trees averages 100 predictions.

### What is the same as a Decision Tree?

Each tree still: 1. Evaluates candidate thresholds. 2. Chooses a split
that reduces squared error. 3. Repeats splitting at child nodes. 4.
Predicts the average target at a leaf.

### What makes the trees different?

**Bootstrap sampling:** Each tree generally trains on its own random
sample of training rows, sampled with replacement. Some rows can repeat;
some may be absent from that sample.

**Random feature selection:** You give the model all the selected input
features. You do not assign features to individual trees manually. At
each split, the algorithm randomly selects a subset of features to
consider, then tests thresholds for those features and chooses the best
split among the candidates.

If a node considers horsepower and weight, it compares candidate
thresholds for both, but it chooses one feature and one threshold for
that split. Another node can consider a different subset. A path can use
several features at different nodes.

### Code: all features

``` python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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

In our session: - Random Forest with horsepower only: test R² about
**0.5645**. - Random Forest with all features: test R² about **0.8869**.

These results came from the experiments we ran; they are not guaranteed
for a different split or setup. The all-feature model can learn from
weight, displacement, cylinders, model year, and other information as
well as horsepower.

### Main hyperparameters

  Hyperparameter        Meaning
  --------------------- --------------------------------------------------
  `n_estimators`        Number of trees
  `max_depth`           Maximum depth of each tree
  `max_features`        Number/fraction of features considered per split
  `min_samples_split`   Minimum samples to split a node
  `min_samples_leaf`    Minimum samples in a leaf
  `random_state`        Reproducibility

`max_features=1.0` lets each split consider all features. A smaller
setting adds more feature randomness. Even with all features considered,
bootstrap samples still help diversify trees.

### Tune number of trees

``` python
for n in [10, 50, 100, 200]:
    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    print(n, r2_score(y_val, val_pred))
```

Use validation scores to compare the candidates, then evaluate the final
selected configuration on the test set.

### Feature importance

``` python
import pandas as pd

importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values()

importance.plot(kind="barh")
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.show()
```

This is impurity-based feature importance. It indicates how much
features contributed to reductions in split impurity across the forest,
not how much MPG each feature independently explains. It does not prove
causation, and correlated features can affect the scores.

In our graph, displacement was highest at approximately 0.42, followed
by horsepower, weight, and cylinders. These are approximate values from
that model.

### Strengths and limitations

Strengths: often more stable than a single tree, captures nonlinear
patterns and interactions, usually does not require feature scaling.

Limitations: more memory and compute, less interpretable than a small
tree, feature importance has limitations, and predictions may not
extrapolate well beyond training patterns.

------------------------------------------------------------------------

## 10. Hyperparameter tuning --- the general pattern

Hyperparameters are settings selected by the developer, not values
learned as ordinary model parameters. Examples: - Polynomial Regression:
polynomial degree. - Decision Tree: `max_depth`, `min_samples_leaf`. -
Random Forest: `n_estimators`, `max_features`, `max_depth`.

General tuning procedure: 1. Choose candidate settings. 2. Fit each
candidate on training data. 3. Compare validation metrics. 4. Select the
best setting. 5. Refit using train + validation data. 6. Evaluate once
on the test set.

For a more systematic search, scikit-learn provides `GridSearchCV` and
`RandomizedSearchCV`. Cross-validation is useful when data is limited.

------------------------------------------------------------------------

## 11. Model comparison

During our session, the following test R² scores were reported:

  Model                               Test R² Feature setup
  --------------------------------- --------- -----------------
  Linear Regression                     0.566 Horsepower only
  Polynomial Regression, degree 2       0.639 Horsepower only
  Decision Tree, depth 3                0.610 Horsepower only
  Random Forest                        0.5645 Horsepower only
  Random Forest                        0.8869 All features

**Important:** The first four scores used horsepower only; the last used
all features. The last score is not a perfectly controlled
model-to-model comparison because the input features changed. For a fair
comparison, evaluate every model with the same feature set and same test
rows.

Higher R² is generally better on the same test set, but also look at MAE
and RMSE. Model selection should consider validation performance and
practical needs, not just one score.

------------------------------------------------------------------------

## 12. Useful plots

### Actual vs predicted scatter plot

``` python
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()
```

Points close to the ideal diagonal relationship (predicted equals
actual) indicate better predictions. To draw an explicit reference line,
add a diagonal line spanning the range of values.

### Feature importance

Use the Random Forest code above. It is useful for understanding which
features the model relied on, with the limitations already described.

### Decision Tree diagram

Use `plot_tree()` for a single tree. A Random Forest contains many
trees, so plotting all of them at once is usually not useful.

------------------------------------------------------------------------

## 13. Common mistakes to avoid

1.  **Target leakage:** never include `mpg` in `X` when predicting MPG.
2.  **Missing values:** clean or impute missing values before fitting
    estimators that cannot handle them.
3.  **Wrong input shape:** use `df[["horsepower"]]` for a one-feature
    DataFrame.
4.  **Mismatched rows:** split `X` and `y` together using
    `train_test_split`.
5.  **Different test splits:** compare models on the same held-out rows.
6.  **Test-set tuning:** do not use test results to choose
    hyperparameters.
7.  **Overfitting:** a model that fits training data very well may still
    perform poorly on new data.
8.  **Overinterpreting R²:** R² is not the percentage of individual
    predictions that are correct.
9.  **Overinterpreting feature importance:** it is not causation and
    does not mean a feature alone explains that percentage of the
    target.
10. **Unsorted plots:** sort a single feature before plotting a
    predicted line or the line may zigzag.
11. **Assuming more complexity always helps:** higher polynomial degree,
    deeper trees, or more trees do not guarantee better test
    performance.
12. **Comparing unequal setups:** if one model uses one feature and
    another uses seven, the comparison includes a feature-set change as
    well as a model change.

------------------------------------------------------------------------

## 14. Quick reference

  ----------------------------------------------------------------------------
  Model             Main idea         Main settings         Common risk
  ----------------- ----------------- --------------------- ------------------
  Linear Regression Fits a linear     Feature selection;    Underfitting
                    relationship      regularization        nonlinear patterns
                                      variants if used      

  Polynomial        Adds powers of    `degree`              High-degree
  Regression        features to fit                         overfitting
                    curves                                  

  Decision Tree     Recursive         `max_depth`,          Overfitting /
  Regressor         threshold splits; `min_samples_leaf`,   instability
                    leaf predicts     `min_samples_split`   
                    average target                          

  Random Forest     Many randomized   `n_estimators`,       More compute,
  Regressor         trees; average    `max_features`,       lower
                    predictions       `max_depth`           interpretability
  ----------------------------------------------------------------------------

## 15. Final checklist

-   [ ] I can explain features (`X`) and target (`y`).
-   [ ] I can clean missing data and split into train/test.
-   [ ] I understand why validation data is useful.
-   [ ] I can explain R², MAE, and RMSE.
-   [ ] I can fit and plot Linear Regression.
-   [ ] I can explain why Polynomial Regression creates polynomial
    features.
-   [ ] I can explain how a Decision Tree chooses a threshold and
    predicts from a leaf.
-   [ ] I can explain bootstrap sampling in Random Forest.
-   [ ] I understand that Random Forest considers a subset of features
    at each split, then selects one feature and threshold for that
    split.
-   [ ] I can tune hyperparameters using validation data.
-   [ ] I can plot feature importance and explain its limitations.
-   [ ] I can compare models fairly using the same test rows and feature
    set.

**Next learning step:** Gradient Boosting Regressor. Random Forest trees
are generally built independently and their predictions are averaged;
boosting builds trees sequentially so later trees can improve on earlier
predictions.
