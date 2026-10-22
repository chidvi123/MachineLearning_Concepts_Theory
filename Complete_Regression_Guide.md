# Regression --- Complete Beginner-to-Practical Guide

This guide covers regression from the basics through Linear Regression,
Polynomial Regression, Decision Tree Regression, Random Forest
Regression, evaluation, plotting, and hyperparameter tuning. It uses the
Auto MPG dataset as a running example.

> **How to use this file:** Read the Markdown theory sections. Copy each
> fenced Python code block into a separate Jupyter Notebook cell, in
> order. Code blocks are marked as setup, data preparation, model,
> evaluation, or visualization cells.

------------------------------------------------------------------------

# 1. Regression basics

## What is machine learning?

Machine learning allows a computer to learn patterns from example data
instead of us manually writing every prediction rule.

## What is regression?

**Regression predicts a numerical value.** Examples include house price,
temperature, sales, or a car's fuel efficiency.

Classification predicts a category, such as spam/not spam. Regression
predicts a number, such as 25.4 MPG.

## Important vocabulary

  -----------------------------------------------------------------------
  Term                    Meaning                 Auto MPG example
  ----------------------- ----------------------- -----------------------
  Dataset                 Collection of examples  Car records

  Sample / row            One example             One car

  Feature (`X`)           Input used to predict   Horsepower, weight

  Target (`y`)            Value being predicted   MPG

  Model                   Learned prediction      Random Forest
                          method                  

  Training                Learning patterns from  Learning MPG
                          data                    relationships

  Prediction              Model's estimated       Predicted MPG
                          answer                  

  Residual                Actual minus predicted  Actual MPG − predicted
                                                  MPG

  Parameter               Value learned by the    Linear Regression
                          model                   coefficient

  Hyperparameter          Setting chosen by us    Tree `max_depth`
  -----------------------------------------------------------------------

## Features and target

Suppose horsepower is used to predict MPG:

``` python
X = df[["horsepower"]]  # input features; double brackets keep it as a DataFrame
y = df["mpg"]           # target; a Series
```

Use more columns if you want the model to consider more information:

``` python
X = df[["horsepower", "weight", "displacement", "cylinders",
        "acceleration", "model_year", "origin"]]
y = df["mpg"]
```

A model can only use information supplied in `X`. Do not put the target
(`mpg`) into `X` when predicting MPG.

## Underfitting and overfitting

-   **Underfitting:** the model is too simple to learn the useful
    patterns.
-   **Overfitting:** the model fits training details/noise too closely
    and performs worse on unseen data.
-   **Generalization:** how well a model works on data it did not train
    on.

## Parameters versus hyperparameters

A model learns parameters during fitting. We choose hyperparameters to
control how learning happens. Examples: Polynomial Regression's
`degree`, Decision Tree's `max_depth`, Random Forest's `n_estimators`.

------------------------------------------------------------------------

# 2. Load and clean the Auto MPG dataset

The commonly used Auto MPG CSV has no header row and uses `?` for
missing horsepower values. This code assumes the file is named
`auto-mpg.csv` and is in the notebook's working directory.

### Jupyter cell 1 --- imports

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
```

### Jupyter cell 2 --- load and clean

``` python
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

# Convert horsepower to numeric and remove rows with missing values.
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna().copy()

print(df.head())
print(df.info())
print("Rows after cleaning:", len(df))
```

Why clean first? A text value like `?` cannot be used as a numeric
horsepower value. In the version we worked with, cleaning left 392 rows.

### Jupyter cell 3 --- choose one feature for the first comparisons

``` python
X = df[["horsepower"]]
y = df["mpg"]
```

Use the double brackets for `X` because scikit-learn expects a
two-dimensional feature table. `y` is the one-dimensional target.

### Jupyter cell 4 --- make a consistent train/test split

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
```

The rows in `X_train` stay paired with the corresponding rows in
`y_train`. `random_state=42` makes the random split reproducible. Do not
use test scores to select hyperparameters.

------------------------------------------------------------------------

# 3. Regression evaluation metrics

We predict values and compare them with the actual target values.

## Residual

\[ e_i = y_i - `\hat{y}`{=tex}\_i \]

If actual MPG is 24 and predicted MPG is 21, the residual is (24-21=3)
MPG. The prediction was 3 MPG too low.

## R² --- coefficient of determination

\[ R\^2 =
1-`\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}`{=tex} \]

-   `1.0`: perfect predictions.
-   `0.0`: same squared-error performance as predicting the actual
    target mean on that evaluation set.
-   Below `0`: worse than the mean-prediction baseline.

R² is not the percentage of predictions that are correct. Higher is
generally better.

## MAE --- mean absolute error

\[ MAE = `\frac{1}{n}`{=tex}`\sum`{=tex}\_i \|y_i-`\hat{y}`{=tex}\_i\|
\]

MAE is the average absolute error and uses the target's unit. For MPG,
MAE is measured in MPG. Lower is better.

## RMSE --- root mean squared error

\[ RMSE = `\sqrt{\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2}`{=tex} \]

RMSE uses the target's unit and penalizes large errors more than MAE.
Lower is better.

### Jupyter cell --- reusable evaluation

Run this cell after fitting any model and creating `y_pred`.

``` python
def evaluate_regression(y_true, y_pred):
    print(f"R2:   {r2_score(y_true, y_pred):.4f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.4f}")
```

------------------------------------------------------------------------

# 4. Linear Regression

## What it does

Linear Regression fits a straight-line relationship between input
features and a numerical target.

With one feature:

\[ `\hat{y}`{=tex}=b_0+b_1x \]

-   (`\hat{y}`{=tex}): predicted value.
-   (x): input feature.
-   (b_0): intercept.
-   (b_1): slope/coefficient.

For MPG and horsepower:

\[ `\widehat{MPG}`{=tex}=b_0+b_1(`\text{horsepower}`{=tex}) \]

### Example calculation

Suppose a model's illustrative equation is:

\[ `\widehat{MPG}`{=tex}=40-0.15(`\text{horsepower}`{=tex}) \]

For horsepower = 100, prediction is (40-0.15(100)=25) MPG. This is a
made-up example to explain the equation, not the fitted equation from
our dataset.

## How it learns

Ordinary least-squares Linear Regression chooses coefficients to
minimize the sum of squared residuals:

\[ SSE=`\sum`{=tex}\_i(y_i-`\hat{y}`{=tex}\_i)\^2 \]

Squaring prevents positive and negative residuals from cancelling and
penalizes larger errors more strongly.

With several features:

\[ `\hat{y}`{=tex}=b_0+b_1x_1+b_2x_2+`\cdots`{=tex}+b_px_p \]

Each coefficient describes the fitted relationship with that feature
while holding other included features fixed. This does not prove cause
and effect.

### Jupyter cell --- fit and predict

``` python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

print("Intercept:", linear_model.intercept_)
print("Coefficient(s):", linear_model.coef_)
evaluate_regression(y_test, linear_pred)
```

## Strengths and limitations

**Strengths:** simple, fast, interpretable coefficients, useful
baseline.

**Limitations:** a straight line may miss curves; outliers can strongly
affect it; correlated features can make coefficients unstable or
difficult to interpret.

Ordinary Linear Regression usually does not require feature scaling.

------------------------------------------------------------------------

# 5. Polynomial Regression

## Why use it?

A straight line cannot represent every relationship. Polynomial
Regression adds powers of features so Linear Regression can fit a curve.

Degree 1:

\[ `\hat{y}`{=tex}=b_0+b_1x \]

Degree 2:

\[ `\hat{y}`{=tex}=b_0+b_1x+b_2x\^2 \]

Degree 3:

\[ `\hat{y}`{=tex}=b_0+b_1x+b_2x^2+b_3x^3 \]

The model is still linear in the coefficients, but it can be curved in
the original input `x`.

### Example calculation

Suppose an illustrative degree-2 model is:

\[ `\widehat{MPG}`{=tex}=10+0.5x-0.004x\^2 \]

For (x=50): (10+0.5(50)-0.004(50\^2)=25). This is an example equation,
not the fitted model from the dataset.

## What the transformer does

For degree 2, `PolynomialFeatures` transforms `x` into columns
representing `x` and `x²`. Linear Regression learns a coefficient for
each transformed column.

### Jupyter cell --- fit degree 2

``` python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

evaluate_regression(y_test, poly_pred)
```

A pipeline applies the feature transformation and regression model in
the correct order.

## Choosing the degree

-   Degree 1: straight line.
-   Degree 2: quadratic curve.
-   Degree 3: more flexible curve.
-   High degrees: more flexibility but greater risk of overfitting and
    unstable behavior.

`degree` is a hyperparameter. Choose it using validation data or
cross-validation, not repeated inspection of test scores.

------------------------------------------------------------------------

# 6. Decision Tree Regressor

## What it does

A Decision Tree predicts a number by asking feature-threshold questions.
A split might be `horsepower <= 84.5`. The algorithm learns the
threshold from training data.

## How a split is chosen

At each node, the algorithm evaluates candidate features and thresholds.
For squared-error regression, it prefers the split that most reduces the
weighted mean squared error (MSE) of the child groups.

\[ MSE=`\frac{1}{n}`{=tex}`\sum`{=tex}\_i(y_i-`\bar`{=tex}{y})\^2 \]

Example: target values 20, 22, 24 have mean 22. Their squared deviations
are 4, 0, 4, so MSE is ((4+0+4)/3=2.67).

The algorithm considers candidate splits, calculates the resulting
weighted errors, and chooses the best available split. It repeats this
process down the tree.

**One node uses one feature and one threshold.** A later node can use a
different feature. For example, one node might split on weight and a
later node might split on horsepower.

## What does a leaf predict?

For squared-error regression, a leaf predicts the mean target value of
training rows that reach it. If those MPG values are 20, 22, and 24, the
leaf predicts 22 MPG for new rows reaching that leaf.

## Important tree plot labels

-   `feature <= threshold`: decision at that node.
-   `samples`: training rows reaching that node.
-   `value`: mean target value at that node.
-   `squared_error`: target variation at that node.

### Jupyter cell --- train a tree

``` python
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)
evaluate_regression(y_test, tree_pred)
```

### Jupyter cell --- plot the tree

``` python
from sklearn.tree import plot_tree

plt.figure(figsize=(16, 8))
plot_tree(
    tree_model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    precision=2
)
plt.show()
```

### Jupyter cell --- show its prediction pattern

``` python
X_sorted = X.sort_values("horsepower")
plt.scatter(X["horsepower"], y, alpha=0.5)
plt.plot(
    X_sorted["horsepower"],
    tree_model.predict(X_sorted),
    color="red"
)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Decision Tree Regression")
plt.show()
```

The line is step-shaped because each leaf gives a constant prediction.

## Main hyperparameters

-   `max_depth`: maximum tree depth.
-   `min_samples_split`: minimum rows needed to split a node.
-   `min_samples_leaf`: minimum rows allowed in a leaf.
-   `max_features`: number/fraction of features considered when
    searching for a split.

A shallow tree can underfit; a very deep tree can overfit. Tune depth
and leaf settings using validation data.

------------------------------------------------------------------------

# 7. Random Forest Regressor

## What it does

Random Forest is an ensemble of many Decision Trees. For regression,
each tree predicts a number and the forest averages them.

\[
`\hat{y}`{=tex}*{forest}=`\frac{1}{T}`{=tex}`\sum`{=tex}*{t=1}\^{T}`\hat{y}`{=tex}\_t
\]

If three trees predict 25, 28, and 24 MPG, the forest prediction is
((25+28+24)/3=25.67) MPG.

## How the trees become different

### 1. Bootstrap sampling --- different rows

Each tree is generally trained using a bootstrap sample of training
rows, sampled **with replacement**. A row may appear multiple times in
one tree's sample, while other rows may not appear. Different trees
therefore see different samples.

### 2. Random feature selection --- candidate features at each split

You normally give the forest all selected input columns. The algorithm
does not require you to manually assign a fixed group of columns to each
tree.

Instead, at each node, it randomly chooses a subset of features to
consider. It evaluates candidate thresholds for those features and
chooses one feature/threshold pair for that split. A later node may
consider a different subset.

**A single split still uses one feature and one threshold.** Random
feature selection determines which features are eligible to compete for
that split; it does not combine all selected features into one split
condition.

## Why average trees?

A single tree can be sensitive to changes in training data. Diverse
trees can make different errors, and averaging their predictions often
reduces variance. Random Forest is not guaranteed to beat every other
model on every dataset.

### Jupyter cell --- train and predict

``` python
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)

evaluate_regression(y_test, forest_pred)
```

-   `n_estimators=100`: build 100 trees.
-   `random_state=42`: make the randomized training reproducible.
-   `max_depth`: optional maximum depth of each tree.
-   `max_features`: controls candidate feature subset size at each
    split.
-   `min_samples_leaf`: controls minimum leaf size.

Random Forest usually does not need feature scaling.

## Feature importance

Scikit-learn provides impurity-based importance through
`feature_importances_`. It summarizes how much features contributed to
impurity reductions across the fitted forest, then normalizes the values
to sum to 1.

It does **not** mean a feature causes the target, nor does an importance
of 0.4 mean it explains exactly 40% of the target. Correlated features
and features with many possible split points can affect this measure.

### Jupyter cell --- feature importance plot

``` python
importances = forest_model.feature_importances_
indices = np.argsort(importances)

plt.barh(X.columns[indices], importances[indices])
plt.xlabel("Importance")
plt.title("Random Forest - Feature Importance")
plt.show()
```

This plot uses the features supplied to `forest_model`. If `X` contains
only horsepower, there will only be one feature bar. To compare
importance across multiple features, fit the forest with multiple input
columns.

------------------------------------------------------------------------

# 8. Train, validation, and test sets

## Why separate the data?

-   **Training set:** fits the model.
-   **Validation set:** compares models and chooses hyperparameters.
-   **Test set:** final evaluation after decisions are made.

A model can perform very well on training data but poorly on unseen data
if it overfits.

## Train/test split

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

For 392 rows, `test_size=0.2` gives 79 test rows and 313 training rows
(scikit-learn rounds the requested test count up to a whole row).

## Train/validation/test split

``` python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

print(len(X_train), len(X_val), len(X_test))
```

For 392 rows, this results in 234 training rows, 79 validation rows, and
79 test rows. This is approximately a 60/20/20 split.

`X` and `y` must be split together so each feature row stays paired with
its actual target.

------------------------------------------------------------------------

# 9. Hyperparameter tuning with validation data

## What is tuning?

Hyperparameter tuning means trying candidate settings and selecting one
based on validation performance or cross-validation.

For a Decision Tree, `max_depth` controls how deep it can grow. We can
try several depths, compare validation R², and select the best. We
should not use the test set to select the depth.

### Jupyter cell --- tune Decision Tree depth

Run this after creating the train/validation/test split above.

``` python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
results = []

for depth in depths:
    candidate = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )
    candidate.fit(X_train, y_train)

    val_pred = candidate.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)
    results.append((depth, val_r2))
    print(f"Depth: {depth}, Validation R2: {val_r2:.4f}")

best_depth, best_val_r2 = max(results, key=lambda item: item[1])
print("Best depth:", best_depth)
print("Best validation R2:", round(best_val_r2, 4))
```

### Jupyter cell --- fit chosen model and test once

``` python
best_model = DecisionTreeRegressor(
    max_depth=best_depth,
    random_state=42
)

# Refit on training + validation data after selecting the depth.
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

best_model.fit(X_train_val, y_train_val)
test_pred = best_model.predict(X_test)

print("Final test results:")
evaluate_regression(y_test, test_pred)
```

The validation score chooses the setting. The test score estimates how
well the final selected model performs on unseen data.

## Other settings to tune

  -----------------------------------------------------------------------
  Model                               Hyperparameters to try
  ----------------------------------- -----------------------------------
  Linear Regression                   Usually begin with default OLS;
                                      consider regularized variants when
                                      appropriate

  Polynomial Regression               `degree`

  Decision Tree                       `max_depth`, `min_samples_leaf`,
                                      `min_samples_split`

  Random Forest                       `n_estimators`, `max_features`,
                                      `max_depth`, `min_samples_leaf`
  -----------------------------------------------------------------------

More complexity is not automatically better. Select settings using
held-out validation data.

## Cross-validation

Cross-validation repeatedly trains and validates on different folds of
the available training data. Averaging the validation scores can give a
more stable estimate for model selection. Keep a final test set
separate.

------------------------------------------------------------------------

# 10. Compare models fairly

For a fair comparison: 1. Use the same target. 2. Use the same
train/test rows. 3. Use the same feature set when comparing algorithms
themselves. 4. Tune settings on validation data. 5. Compare R², MAE,
and/or RMSE. 6. Use the test set for the final comparison only after
tuning.

Our earlier reported Auto MPG results were: - Linear Regression,
horsepower only: R² about 0.566. - Polynomial Regression degree 2,
horsepower only: R² about 0.639. - Decision Tree with `max_depth=3`,
horsepower only: R² about 0.610. - Random Forest, horsepower only: R²
about 0.5645. - Random Forest with all available features: R² about
0.8869.

These are results from particular experiments, not guaranteed outcomes.
The last Random Forest used more features, so it is not a direct
algorithm-only comparison with the horsepower-only models.

------------------------------------------------------------------------

# 11. Quick revision

-   Regression predicts a number.
-   `X` contains input features; `y` is the target.
-   Linear Regression learns coefficients for a line/hyperplane.
-   Polynomial Regression adds powers such as (x\^2) to model curves.
-   A Decision Tree learns feature thresholds and predicts the target
    mean at a regression leaf.
-   A single tree node uses one feature and one threshold.
-   Random Forest trains many randomized trees and averages their
    predictions.
-   Bootstrap sampling creates different row samples; `max_features`
    controls candidate features considered at each split.
-   R² is generally better when higher; MAE and RMSE are generally
    better when lower.
-   Hyperparameters are selected using validation data or
    cross-validation.
-   The test set should remain untouched during model selection.
-   Feature importance is model-specific and does not establish
    causation.

## Self-check questions

1.  What is the difference between a feature and a target?
2.  Why can a model have a high training score but a lower test score?
3.  What does the slope mean in a simple Linear Regression equation?
4.  What changes when polynomial degree increases?
5.  How does a Decision Tree choose a split threshold?
6.  What does a regression leaf predict?
7.  What is bootstrap sampling?
8.  How does Random Forest select features at each split?
9.  What does `max_depth` control?
10. Why is validation data used for tuning?
11. What does a negative R² mean?
12. Why is feature importance not proof of causation?
