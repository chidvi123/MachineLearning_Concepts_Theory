# Decision Tree Regression — Complete Notes

## 1. What Is Decision Tree Regression?

Decision Tree Regression is a supervised Machine Learning algorithm used to predict continuous numerical values.

It learns a series of decision rules that split the dataset into smaller groups. Each final group is called a **leaf**, and the model predicts a numerical value for observations that reach that leaf.

Unlike Linear Regression, which fits a straight line, a Decision Tree can learn nonlinear relationships by splitting the input space into regions.

### Our Practice Example: Auto MPG Dataset

We used the Auto MPG dataset to predict fuel efficiency.

- `horsepower` — input feature (`X`)
- `mpg` — target (`y`)
- `max_depth=3` — limits the tree's depth

Our goal is to predict MPG based on horsepower.

## 2. How Does a Decision Tree Work?

A Decision Tree repeatedly asks questions about feature values.

For example, a learned tree might ask:

- Is horsepower less than or equal to a threshold?
- If yes, follow the left branch.
- If no, follow the right branch.
- Continue until a leaf is reached.
- Return the prediction stored in that leaf.

The actual thresholds are learned from the training data. They are not manually specified.

A regression tree typically predicts the **mean target value of the training samples in the leaf**.

## 3. Load and Prepare the Dataset

### Jupyter Notebook — Cell 1

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
```

### Jupyter Notebook — Cell 2

```python
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

df["horsepower"] = pd.to_numeric(
    df["horsepower"],
    errors="coerce"
)

df = df.dropna().copy()

print("Dataset shape:", df.shape)
df.head()
```

After cleaning, our dataset contains 392 rows.

## 4. Select the Feature and Target

### Jupyter Notebook — Cell 3

```python
X = df[["horsepower"]]
y = df["mpg"]

print("X shape:", X.shape)
print("y shape:", y.shape)
```

We use double brackets for `X` because scikit-learn expects a two-dimensional feature matrix.

## 5. Split the Data

### Jupyter Notebook — Cell 4

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

The split gives 313 training samples and 79 test samples.

The tree learns its splitting rules from the training data. The test data is reserved for evaluating its predictions.

## 6. Train the Decision Tree Regressor

### Jupyter Notebook — Cell 5

```python
dt_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
```

In our practice, this model achieved an \(R^2\) score of approximately **0.6104**.

Your exact result should be checked by running the notebook with the same dataset and split.

### What does `max_depth=3` mean?

It limits the maximum number of edges from the root to a leaf to three.

- Smaller depth: simpler model, potentially underfits.
- Larger depth: more complex model, potentially overfits.
- No explicit depth limit: the tree can grow until other stopping conditions apply.

`max_depth` is a **hyperparameter** chosen before fitting.

## 7. How Does the Tree Choose a Split?

For regression, scikit-learn's default `criterion="squared_error"` evaluates splits using the reduction in squared-error impurity.

At a node, the algorithm considers candidate features and thresholds. For each candidate split, it calculates the weighted impurity of the resulting child nodes.

A good split creates child groups whose target values are more consistent than those in the parent.

The algorithm selects a split that maximizes the reduction in impurity, subject to the tree's constraints.

### Mean Squared Error at a Node

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\bar y)^2
\]

Where:
- \(y_i\) = actual MPG for a sample.
- \(\bar y\) = mean MPG of the samples in that node.
- \(n\) = number of samples in the node.

### Weighted Impurity After a Split

\[
I_{\text{split}}=
\frac{n_L}{n}I_L+
\frac{n_R}{n}I_R
\]

Where:
- \(n_L\), \(n_R\) = sample counts in the left and right children.
- \(I_L\), \(I_R\) = their impurity values.
- \(n=n_L+n_R\).

The split is useful when the weighted child impurity is lower than the parent's impurity.

## 8. Visualize the Decision Tree

### Jupyter Notebook — Cell 6

```python
plt.figure(figsize=(18, 9))

plot_tree(
    dt_model,
    feature_names=["horsepower"],
    filled=True,
    rounded=True,
    precision=2
)

plt.title("Decision Tree Regressor")
plt.show()
```

### How to read the tree

Each node can display:

- **`horsepower <= threshold`**: the split condition.
- **`squared_error`**: impurity at that node.
- **`samples`**: number of training observations reaching that node.
- **`value`**: mean MPG of the training observations in that node.

The root is the first node. The leaves are the terminal nodes that produce predictions.

For an observation, follow the split rules until reaching a leaf. The leaf's `value` is the predicted MPG.

## 9. Visualize Predictions

### Jupyter Notebook — Cell 7

```python
x_sorted = X.sort_values("horsepower")

plt.scatter(X["horsepower"], y, alpha=0.5, label="Actual data")

plt.plot(
    x_sorted["horsepower"],
    dt_model.predict(x_sorted),
    color="red",
    label="Decision Tree prediction"
)

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Decision Tree Regression: Horsepower vs MPG")
plt.legend()
plt.show()
```

The Decision Tree produces a **step-like prediction curve**, not a smooth line. Within each leaf's interval, the prediction is constant.

## 10. Evaluate the Model

### Jupyter Notebook — Cell 8

```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", rmse)
```

- **R²:** measures performance relative to the test-set mean-prediction baseline.
- **MAE:** average absolute error in MPG.
- **MSE:** average squared error.
- **RMSE:** square root of MSE, expressed in MPG.

A higher R² and lower error metrics are generally preferable when comparing models on the same test set.

## 11. Experiment with Tree Depth

### Jupyter Notebook — Cell 9

```python
for depth in [1, 2, 3, 4, 5, 8, None]:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"Max depth: {depth}")
    print("Train R2:", r2_score(y_train, train_pred))
    print("Test R2:", r2_score(y_test, test_pred))
    print("-" * 30)
```

### How to interpret the results

- Shallow trees may miss important patterns.
- Deeper trees can fit the training data more closely.
- Very deep trees may memorize noise and generalize poorly.
- The best depth depends on the dataset.

For rigorous hyperparameter selection, use a validation set or cross-validation instead of repeatedly selecting the best result on the test set.

## 12. Important Hyperparameters

| Hyperparameter | Purpose |
|---|---|
| `max_depth` | Maximum depth of the tree |
| `min_samples_split` | Minimum number of samples required to split a node |
| `min_samples_leaf` | Minimum number of training samples required in a leaf |
| `max_leaf_nodes` | Limits the number of leaves |
| `criterion` | Impurity measure used to evaluate splits |
| `random_state` | Controls randomness for reproducibility |

These are configuration choices. The splitting thresholds and leaf predictions are learned during training.

## 13. Advantages and Limitations

### Advantages

- Captures nonlinear relationships.
- Does not require feature standardization for this tree-based model.
- Learns decision rules automatically.
- Can model interactions through sequential splits.
- Easy to visualize when the tree is small.

### Limitations

- Deep trees can overfit.
- Small data changes can produce a different tree.
- Predictions are piecewise constant, so the curve is not smooth.
- Individual trees can be less accurate than ensemble models.

## 14. Common Interview Questions

**1. What is Decision Tree Regression?**

A regression algorithm that recursively partitions the feature space and predicts a numerical value in each leaf.

**2. How does a regression tree choose a split?**

It evaluates candidate feature-threshold splits and chooses one that reduces weighted impurity.

**3. What does a leaf predict?**

With the default squared-error criterion, the mean target value of the training samples reaching that leaf.

**4. What does `max_depth` control?**

The maximum distance from the root to a leaf.

**5. Does a Decision Tree need feature scaling?**

Usually not. Its threshold comparisons are generally unaffected by ordinary scaling of a feature.

**6. Why do Decision Trees overfit?**

Deep trees can create narrow regions that fit individual training observations or noise.

**7. Why are tree predictions step-like?**

All observations reaching the same leaf receive the same prediction.

**8. What is the difference between a Decision Tree and Random Forest?**

A Decision Tree uses one tree. A Random Forest combines predictions from many trees to improve robustness and often generalization.

**9. What is `squared_error` in a tree plot?**

It is the node's mean squared-error impurity, based on target variation among training samples in that node.

## 15. Quick Revision

- We used Auto MPG to predict `mpg` from `horsepower`.
- We cleaned the dataset to 392 rows.
- We split it into 313 training and 79 test samples.
- We trained `DecisionTreeRegressor(max_depth=3)`.
- Our observed test R² was approximately 0.6104.
- The tree learns feature thresholds during training.
- Regression leaves predict the mean target under the default squared-error criterion.
- `max_depth` controls complexity.
- Tree predictions are piecewise constant.
- Deep trees can overfit.
- Evaluate models on unseen data and tune hyperparameters using validation data.
