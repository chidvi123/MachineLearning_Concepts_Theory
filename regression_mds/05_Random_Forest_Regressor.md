# Random Forest Regression

## 1. What Is Random Forest Regression?

Random Forest Regression is a **supervised machine learning algorithm** used to predict continuous numerical values.

It is an **ensemble learning algorithm** because it combines predictions from multiple decision trees to produce one final prediction.

Instead of relying on a single Decision Tree, Random Forest trains multiple trees and averages their predictions. This generally makes the model more stable and helps reduce overfitting compared with a single deep decision tree.

### Examples

- Predicting a car's fuel efficiency (MPG)
- Predicting house prices
- Predicting salaries
- Predicting product demand
- Predicting sales revenue

### Decision Tree vs Random Forest

| Decision Tree Regression | Random Forest Regression |
|---|---|
| Uses one decision tree | Uses multiple decision trees |
| Can easily overfit | Usually reduces overfitting |
| Predictions can be unstable | Predictions are generally more stable |
| Simpler and faster to train | More computationally expensive |
| Prediction comes from one leaf | Prediction is the average of tree predictions |

**Key idea:** Random Forest combines multiple decision trees to improve prediction stability and generalization.

---

## 2. How Does Random Forest Regression Work?

Random Forest uses two important sources of randomness:

1. **Bootstrap sampling:** Each tree is trained on a randomly sampled dataset created from the original training data, usually by sampling rows with replacement.
2. **Random feature selection:** At each split, the tree considers only a randomly selected subset of input features rather than every feature.

Each tree grows using these randomized choices. For regression, the forest averages the predictions from all its trees.

### Step-by-step workflow

1. Start with the training dataset.
2. Generate a bootstrap sample of training rows for each tree.
3. Train a Decision Tree on each bootstrap sample.
4. At each tree split, consider a random subset of features.
5. Select the best split among the features considered.
6. Repeat until the trees reach their stopping conditions.
7. Send a new input to every tree.
8. Collect all tree predictions.
9. Calculate their average to obtain the final prediction.

### Visual workflow

```text
                 Training Dataset
                        |
           +------------+------------+
           |            |            |
      Bootstrap 1  Bootstrap 2  Bootstrap 3
           |            |            |
        Tree 1       Tree 2       Tree 3
           |            |            |
      Prediction 1  Prediction 2  Prediction 3
           |            |            |
           +------------+------------+
                        |
                Average Predictions
                        |
                 Final Prediction
```

---

## 3. Mathematical Intuition

Suppose a Random Forest contains \(N\) decision trees.

Each tree produces a prediction:

\[
\hat{y}_1,\hat{y}_2,\hat{y}_3,\ldots,\hat{y}_N
\]

The final prediction is the average:

\[
\boxed{\hat{y}=\frac{1}{N}\sum_{i=1}^{N}\hat{y}_i}
\]

### Example

Suppose five trees predict the following MPG values for the same car:

| Tree | Predicted MPG |
|---|---:|
| Tree 1 | 24 |
| Tree 2 | 26 |
| Tree 3 | 25 |
| Tree 4 | 27 |
| Tree 5 | 23 |

The final prediction is:

\[
\hat{y}=\frac{24+26+25+27+23}{5}=25
\]

Therefore, the Random Forest predicts **25 MPG**.

Each tree may make a slightly different prediction because it was trained using different sampled rows and randomized feature choices.

---

## 4. Understanding Bootstrap Sampling

Bootstrap sampling means selecting training rows **with replacement**.

Suppose our original dataset contains five rows:

```text
Original dataset: A, B, C, D, E
```

A bootstrap sample might be:

```text
Sample 1: A, B, B, D, E
Sample 2: A, C, C, D, E
Sample 3: B, B, C, D, D
```

Notice that:
- A row can appear multiple times.
- Some rows may not appear in a particular sample.
- Each bootstrap sample typically contains the same number of draws as the original training dataset.

These different samples expose the trees to slightly different versions of the training data.

### What are out-of-bag samples?

Rows that are not selected for a particular tree's bootstrap sample are called **out-of-bag (OOB) samples** for that tree.

They can be used to estimate model performance without requiring a separate validation split, provided enough trees and suitable settings are used.

In scikit-learn, OOB evaluation can be enabled with `oob_score=True` when bootstrap sampling is enabled.

---

## 5. Understanding Random Feature Selection

Random Forest does not necessarily allow every feature to compete at every split.

Instead, it considers a random subset of features and chooses the best split among those candidates.

For example, suppose the dataset has these features:

```text
cylinders
displacement
horsepower
weight
acceleration
model_year
origin
```

At one split, the candidate features might be:

```text
horsepower, weight, model_year
```

At another split, the candidate features might be:

```text
cylinders, acceleration, origin
```

The model evaluates the candidate splits and selects the one that best reduces the regression error.

**Important:** The model does not randomly choose the final feature and threshold. Randomness determines which features are considered; the algorithm then selects the best split among those candidates.

The exact number of features considered depends on the `max_features` parameter.

---

## 6. How Does a Regression Tree Choose a Split?

Random Forest Regression uses decision trees as its underlying models.

For a regression tree, a split is evaluated based on how much it reduces prediction error, commonly measured using Mean Squared Error (MSE).

For a group of target values:

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\bar{y})^2
\]

Here:

- \(n\) is the number of samples in the node.
- \(y_i\) is an actual target value.
- \(\bar{y}\) is the mean target value in that node.

A split divides a node into child nodes. The tree prefers splits that produce a lower **weighted average of child-node impurity**.

\[
MSE_{\text{split}}=
\frac{n_L}{n}MSE_L+
\frac{n_R}{n}MSE_R
\]

The impurity reduction is:

\[
\Delta MSE=MSE_{\text{parent}}-MSE_{\text{split}}
\]

A larger positive reduction means the split reduces prediction error more.

### What does a leaf predict?

In a standard regression tree, a leaf predicts the mean target value of the training samples that reach that leaf.

For example, if a leaf contains MPG values:

```text
20, 22, 24, 26
```

Its prediction is:

\[
\frac{20+22+24+26}{4}=23
\]

Each tree produces a leaf-based prediction, and Random Forest averages those predictions.

---

## 7. Random Forest Regression Using the Auto MPG Dataset

We will use the same dataset and prediction task as our previous regression experiments.

**Problem:** Predict a car's MPG using its horsepower.

- Input feature: `horsepower`
- Target: `mpg`
- Dataset: Auto MPG
- Algorithm: Random Forest Regression

This experiment deliberately uses only horsepower so that we can compare it with our earlier Linear Regression, Polynomial Regression, and Decision Tree Regression experiments using the same input feature and data split.

### Jupyter Notebook — Cell 1: Import libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
```

### Jupyter Notebook — Cell 2: Load the dataset

```python
columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin"
]

df = pd.read_csv(
    "auto-mpg.csv",
    header=None,
    names=columns,
    na_values="?"
)

df["horsepower"] = pd.to_numeric(
    df["horsepower"],
    errors="coerce"
)

df = df.dropna().copy()

df.head()
```

### Jupyter Notebook — Cell 3: Inspect the dataset

```python
print("Dataset shape:", df.shape)
print(df.info())
print(df.isnull().sum())
```

After cleaning, our dataset contains **392 rows**.

### Jupyter Notebook — Cell 4: Select features and target

```python
X = df[["horsepower"]]
y = df["mpg"]

print("X shape:", X.shape)
print("y shape:", y.shape)
```

Here:

- `X` contains the input feature.
- `y` contains the value we want to predict.

We use double brackets in `df[["horsepower"]]` to preserve `X` as a two-dimensional DataFrame, which scikit-learn expects for input features.

### Jupyter Notebook — Cell 5: Split the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
```

Expected split:

```text
Training samples: 313
Testing samples: 79
```

The test set is held out for evaluation after training.

### Jupyter Notebook — Cell 6: Create the Random Forest model

```python
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

Explanation:

- `n_estimators=100`: Train 100 decision trees.
- `random_state=42`: Make the randomized training process reproducible for the same data and environment.
- `max_depth=None` by default: Trees can continue splitting until other stopping conditions apply.

### Jupyter Notebook — Cell 7: Train the model

```python
rf_model.fit(X_train, y_train)
```

During training, the forest fits multiple randomized regression trees to the training data.

### Jupyter Notebook — Cell 8: Make predictions

```python
y_pred = rf_model.predict(X_test)

print("Actual values:")
print(y_test.head().to_list())

print("Predicted values:")
print(y_pred[:5])
```

Each test sample is passed through the trees, and the forest averages their predictions.

### Jupyter Notebook — Cell 9: Evaluate the model

```python
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
```

In our earlier run with this setup, the Random Forest's test \(R^2\) was approximately **0.5645**. Your exact output can vary with library versions or changes to the data and code.

### Jupyter Notebook — Cell 10: Visualize predictions

```python
x_sorted = X.sort_values("horsepower")

rf_predictions = rf_model.predict(x_sorted)

plt.figure(figsize=(10, 6))

plt.scatter(
    X["horsepower"],
    y,
    alpha=0.6,
    label="Actual data"
)

plt.plot(
    x_sorted["horsepower"],
    rf_predictions,
    color="red",
    label="Random Forest prediction"
)

plt.xlabel("Horsepower")
plt.ylabel("Miles per Gallon (MPG)")
plt.title("Random Forest Regression: Horsepower vs MPG")
plt.legend()
plt.show()
```

The prediction curve is typically piecewise constant because the individual regression trees predict constant values within their leaves. Averaging many trees can make the forest's curve less jagged, but it does not make it a smooth polynomial curve.

---

## 8. Important Random Forest Hyperparameters

Hyperparameters control how the forest is built.

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `n_estimators` | Number of trees | More trees can stabilize predictions but increase training and prediction time |
| `max_depth` | Maximum depth of each tree | Limits tree complexity |
| `min_samples_split` | Minimum samples required to split a node | Larger values restrict splitting |
| `min_samples_leaf` | Minimum samples required in a leaf | Larger values produce less specific leaf predictions |
| `max_features` | Number or proportion of features considered at each split | Controls feature randomness |
| `bootstrap` | Whether to train each tree using bootstrap samples | Adds row-sampling randomness when enabled |
| `max_samples` | Number or proportion of rows drawn for each tree | Controls bootstrap sample size |
| `random_state` | Random seed | Helps make results reproducible |
| `oob_score` | Whether to calculate out-of-bag evaluation | Provides an additional performance estimate when compatible with bootstrap sampling |

### Example: Restricting tree depth

```python
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)
```

These are example settings, not necessarily the best settings for this dataset. Use validation or cross-validation to determine suitable hyperparameters.

---

## 9. Why Does Random Forest Usually Reduce Overfitting?

A single deep Decision Tree can fit the training data very closely, including noise.

Random Forest trains multiple trees using different samples and feature choices. Their prediction errors are not perfectly identical, so averaging can reduce prediction variance.

However, Random Forest is **not guaranteed** to outperform every individual tree or every other algorithm. It can still overfit, and its performance depends on the dataset and hyperparameters.

### Bias–variance intuition

- **Decision Tree:** Can have low bias but high variance when grown deeply.
- **Random Forest:** Often reduces variance by averaging many diverse trees.
- **Too much restriction:** Very shallow trees may underfit.
- **Too little restriction:** Complex trees may still overfit, although averaging often helps.

The goal is to find a balance that generalizes well to unseen data.

---

## 10. Feature Importance

Random Forest can estimate which input features contributed most to reducing impurity across its trees.

### Jupyter Notebook — Cell 11: Extract feature importance

```python
importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)

importance.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.show()
```

Because this experiment uses only `horsepower`, it will have only one feature-importance value.

If we train a separate forest using all the available input features, we can compare their estimated importance values.

**Important limitations:**

- Feature importance is not proof of causation.
- Impurity-based importance can favor certain types of features.
- Correlated features can share or distort apparent importance.
- Permutation importance on held-out data can be a useful complementary method.

---

## 11. Random Forest with All Features

The earlier experiment deliberately used only horsepower. Random Forest can also use multiple features.

For a separate all-feature experiment:

```python
X_all = df.drop(columns=["mpg"])
y_all = df["mpg"]

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42
)

rf_all = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_all.fit(X_train_all, y_train_all)

y_pred_all = rf_all.predict(X_test_all)

print("R2 Score:", r2_score(y_test_all, y_pred_all))
print(
    "RMSE:",
    np.sqrt(mean_squared_error(y_test_all, y_pred_all))
)
```

In our earlier run, this all-feature model achieved an \(R^2\) of approximately **0.8869**.

Do not interpret this as proof that Random Forest is inherently better than the earlier models: this version uses several features, whereas the earlier experiments used horsepower alone. A fair algorithm comparison should use the same dataset, feature set, split, and evaluation metrics.

---

## 12. Hyperparameter Tuning and Data Leakage

A good workflow separates training, validation, and testing.

- **Training set:** Fits model parameters.
- **Validation set:** Helps choose hyperparameters and compare candidate models.
- **Test set:** Estimates final performance after model selection.

For this dataset, one possible split is:

```text
Entire cleaned dataset: 392 rows

Training:   234 rows
Validation:  79 rows
Testing:     79 rows
```

The exact row counts can depend on the splitting method and rounding.

Do not repeatedly inspect test performance to choose hyperparameters. Doing so gradually leaks information from the test set into model selection.

For smaller datasets, cross-validation on the training portion is often a more efficient way to compare hyperparameters.

---

## 13. Evaluation Metrics

Random Forest Regression uses the same common regression metrics as Linear Regression and Decision Tree Regression.

### R² Score

\[
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
\]

- Higher is generally better.
- \(1\) represents perfect predictions.
- \(0\) means performance equivalent to predicting the test-set target mean, under the usual definition.
- A negative value means worse performance than that baseline.

### Mean Absolute Error (MAE)

\[
MAE=\frac{1}{n}\sum_i|y_i-\hat{y}_i|
\]

Measures average absolute prediction error.

### Mean Squared Error (MSE)

\[
MSE=\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2
\]

Penalizes larger errors more heavily because errors are squared.

### Root Mean Squared Error (RMSE)

\[
RMSE=\sqrt{MSE}
\]

Expressed in the same units as the target, making it easier to interpret.

---

## 14. Advantages and Disadvantages

### Advantages

- Often generalizes better than a single deep Decision Tree.
- Can capture nonlinear relationships.
- Handles feature interactions automatically.
- Usually requires little feature scaling.
- Can estimate feature importance.
- Works for both regression and classification through different estimator classes.

### Disadvantages

- More computationally expensive than one Decision Tree.
- Can use substantial memory when many large trees are trained.
- Less interpretable than a single tree.
- Does not naturally extrapolate trends beyond the ranges represented by training leaves.
- Feature importance can be misleading when features are correlated or have different measurement characteristics.
- More trees do not always produce meaningfully better performance.

---

## 15. Random Forest vs Other Regression Algorithms

| Algorithm | Main idea | Typical prediction behavior |
|---|---|---|
| Linear Regression | Fits a linear relationship | Straight-line prediction for one feature |
| Polynomial Regression | Fits polynomial features with a linear model | Smooth polynomial curve |
| Decision Tree Regression | Recursively splits the feature space | Piecewise-constant predictions |
| Random Forest Regression | Averages many randomized trees | Piecewise-constant, averaged predictions |

### Our Auto MPG experiments

These are the approximate results from the earlier runs using horsepower to predict MPG.

| Model | Test R² |
|---|---:|
| Linear Regression | 0.566 |
| Polynomial Regression, degree 2 | 0.639 |
| Decision Tree Regression, `max_depth=3` | 0.610 |
| Random Forest Regression | 0.564 |

The Random Forest did not outperform the polynomial model in this particular one-feature experiment. That is completely possible: ensemble models are powerful tools, not guaranteed winners for every dataset or feature set.

The all-feature Random Forest result of approximately 0.887 belongs to a different experiment and should not be directly compared with this table.

---

## 16. Common Mistakes

1. **Assuming more trees always improve accuracy:** More trees can stabilize predictions, but improvements eventually diminish.
2. **Assuming Random Forest always beats a Decision Tree:** It often reduces variance, but results depend on the data and settings.
3. **Scaling unnecessarily:** Tree-based models generally do not need standardization or normalization.
4. **Using the test set for hyperparameter selection:** Keep it for final evaluation.
5. **Treating feature importance as causation:** Importance measures predictive contribution under a particular method, not causal effect.
6. **Comparing models trained on different features:** Use the same data and features for a fair comparison.
7. **Assuming Random Forest extrapolates like Linear Regression:** Tree-based models generally predict using learned leaf values, so extrapolation is limited.

---

## 17. Interview Questions and Answers

### Q1. What is Random Forest Regression?

It is an ensemble algorithm that trains multiple randomized decision trees and averages their predictions to predict continuous values.

### Q2. Why is it called an ensemble algorithm?

Because it combines predictions from multiple models to produce one final prediction.

### Q3. What is bootstrap sampling?

It is sampling rows with replacement to create training samples for individual trees.

### Q4. Why are random features considered at each split?

To increase diversity among trees and reduce correlation between their predictions.

### Q5. How does Random Forest make a regression prediction?

It averages the predictions of all its trees.

### Q6. How does Random Forest reduce variance?

Averaging predictions from sufficiently diverse trees can reduce the variability of the overall prediction.

### Q7. Does Random Forest require feature scaling?

Generally, no. Decision-tree splits depend on feature thresholds and ordering, so scaling is usually unnecessary.

### Q8. What is `n_estimators`?

It specifies the number of trees in the forest.

### Q9. What is `max_depth`?

It limits the maximum depth of each tree.

### Q10. What are out-of-bag samples?

They are training rows omitted from a particular tree's bootstrap sample. They can help estimate performance when enough trees are used.

### Q11. Can Random Forest overfit?

Yes. Although it often reduces variance compared with a single tree, its performance still depends on model complexity, data quality, and hyperparameters.

### Q12. What is the difference between Random Forest and Gradient Boosting?

Random Forest typically trains trees independently using randomized samples and features, then averages their predictions. Gradient Boosting trains trees sequentially, with later trees attempting to correct errors made by the existing ensemble.

### Q13. Can Random Forest handle nonlinear relationships?

Yes. Tree splits allow it to model nonlinear patterns and interactions without explicitly creating polynomial features.

### Q14. Why might a Random Forest perform worse than Polynomial Regression?

The dataset may have a simple smooth relationship that a polynomial model captures effectively, while the forest's piecewise-constant predictions may be less suitable. Model performance depends on the dataset and feature set.

---

## 18. Final Revision Summary

Remember these key points:

- Random Forest is an **ensemble of decision trees**.
- It uses bootstrap sampling and random feature selection to create diverse trees.
- For regression, it averages the trees' predictions.
- `n_estimators` controls the number of trees.
- `max_depth` and `min_samples_leaf` help control tree complexity.
- Tree-based models generally do not require feature scaling.
- Feature importance is not causation.
- Use validation or cross-validation for tuning and preserve a held-out test set.
- Always compare models using the same dataset, feature set, split, and metrics.

**One-line interview answer:** Random Forest Regression is an ensemble learning method that trains multiple decision trees using randomized samples and feature subsets, then averages their predictions to improve stability and often reduce overfitting.