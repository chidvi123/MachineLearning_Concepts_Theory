# Decision Tree Regressor --- Complete Notes

## 1. What is a Decision Tree Regressor?

A Decision Tree predicts a continuous target by repeatedly splitting the
data into smaller groups using feature conditions.

Example: - Is `horsepower <= 84.5`? - If yes, follow the left branch. -
If no, follow the right branch.

The thresholds are learned from the training data; we do not need to
specify them manually.

## 2. How does the tree choose a split?

At a node, the tree considers candidate features and thresholds. For
regression, it commonly chooses the split that produces the greatest
reduction in squared error, measured through weighted mean squared error
(MSE).

Conceptually: 1. Try a candidate feature and threshold. 2. Divide rows
into left and right groups. 3. Calculate the weighted MSE of the child
groups. 4. Compare candidate splits. 5. Choose the split with the lowest
weighted child error. 6. Repeat at the child nodes until a stopping rule
is reached.

For example, the tree may test `horsepower <= 70`, `horsepower <= 80`,
and `horsepower <= 90`. It evaluates these candidate thresholds and
selects the best available split.

## 3. How does a leaf make a prediction?

When a new row reaches a leaf, a regression tree usually predicts the
**average target value** of the training samples in that leaf.

If a leaf contains training MPG values 20, 22, and 24, its prediction
is:

\[ (20 + 22 + 24) / 3 = 22 \]

## 4. Train a Decision Tree Regressor

``` python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

X = df[["horsepower"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("Decision Tree R²:", r2_score(y_test, y_pred))
```

`max_depth=3` limits the tree to at most three levels of splits (depth
is counted from the root according to scikit-learn's convention).

## 5. Visualize the tree

``` python
import matplotlib.pyplot as plt
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

Important values in a regression tree plot: - `feature <= threshold`:
condition used at that node. - `samples`: number of training rows
reaching that node. - `value`: average target value of those samples. -
`squared_error`: variation in target values at that node (impurity for
squared-error regression).

## 6. Plot the predictions with one feature

``` python
X_sorted = X.sort_values("horsepower")

plt.scatter(X["horsepower"], y, alpha=0.6)
plt.plot(
    X_sorted["horsepower"],
    dt_model.predict(X_sorted),
    color="red"
)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Decision Tree Regression")
plt.show()
```

A tree's prediction curve is step-like because all inputs in a leaf
receive the same prediction.

## 7. Main hyperparameters

-   `max_depth`: limits tree depth.
-   `min_samples_split`: minimum rows required to split a node.
-   `min_samples_leaf`: minimum rows allowed in a leaf.
-   `max_features`: number or fraction of features considered when
    searching for splits.

## 8. Overfitting

A very deep tree can create many small leaves and fit training data
closely. This can lead to poor predictions on unseen data.

Ways to control complexity: - Limit `max_depth`. - Increase
`min_samples_leaf`. - Increase `min_samples_split`. - Select settings
using validation data or cross-validation.

## 9. Tune `max_depth` using validation data

``` python
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
best_depth = None
best_score = float("-inf")

for depth in depths:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    score = r2_score(y_val, val_pred)

    print(f"Depth={depth}, validation R²={score:.4f}")

    if score > best_score:
        best_score = score
        best_depth = depth

print("Best depth:", best_depth)
print("Best validation R²:", best_score)
```

This code assumes `X_train`, `X_val`, `y_train`, and `y_val` were
created using the train/validation/test workflow. Do not use `X_test` to
choose the depth.

After choosing the depth, retrain using the combined training +
validation data, then evaluate once on the test set.

## 10. Strengths and limitations

**Strengths** - Easy to explain and visualize. - Captures nonlinear
relationships. - Does not require feature scaling for ordinary tree
splits.

**Limitations** - A single tree can be unstable and overfit. -
Predictions are piecewise constant. - Small data changes can produce a
different tree.
