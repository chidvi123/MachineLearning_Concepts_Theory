# Linear Regression --- Complete Notes

## 1. What is Linear Regression?

Linear Regression predicts a continuous target by learning a linear
relationship between input features and the target.

With one feature:

\[ `\hat{y}`{=tex} = b_0 + b_1x \]

For Auto MPG: - (x) can be horsepower. - (`\hat{y}`{=tex}) is predicted
MPG. - (b_0) is the intercept. - (b_1) is the learned coefficient.

The model finds coefficients that minimize the sum of squared residuals
on the training data.

A **residual** is:

\[ `\text{residual}`{=tex} = y\_{`\text{actual}`{=tex}} -
y\_{`\text{predicted}`{=tex}} \]

## 2. Fit a model using horsepower only

``` python
from sklearn.model_selection import train_test_split
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
print("Test R²:", r2_score(y_test, y_pred))
```

### What each line does

-   `LinearRegression()` creates the model.
-   `.fit(X_train, y_train)` learns the intercept and coefficient from
    training data.
-   `.predict(X_test)` predicts MPG for unseen test rows.
-   `r2_score(y_test, y_pred)` compares predictions with actual MPG
    values.

The coefficient describes the fitted change in predicted MPG for a
one-unit change in horsepower in this simple model. It is an association
learned from the data, not proof of causation.

## 3. Linear Regression with multiple features

``` python
X = df.drop(columns=["mpg"])
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)
print("Test R²:", r2_score(y_test, y_pred))
```

With multiple features, the model combines a coefficient for each
feature with an intercept.

## 4. Plot a one-feature linear model

``` python
import matplotlib.pyplot as plt

X = df[["horsepower"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

X_sorted = X.sort_values("horsepower")
plt.scatter(X["horsepower"], y, alpha=0.6)
plt.plot(
    X_sorted["horsepower"],
    linear_model.predict(X_sorted),
    color="red"
)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Linear Regression: Horsepower vs MPG")
plt.show()
```

The scatter points show observed cars; the line shows the model's
predictions across horsepower values.

## 5. Strengths and limitations

**Strengths** - Simple, fast, and interpretable. - Useful as a baseline
model. - Coefficients can help describe the fitted relationship.

**Limitations** - A straight-line model may not capture curved
relationships. - Outliers can strongly affect the fitted line. - With
multiple features, correlated inputs can make coefficients harder to
interpret.

## 6. When to use it

Use Linear Regression as a starting point for continuous targets.
Compare it against nonlinear models if the relationship is not
adequately represented by a linear function.

## 7. Common mistakes

-   Accidentally putting `mpg` in `X`.
-   Passing a one-dimensional list where a two-dimensional feature
    matrix is expected; `df[["horsepower"]]` avoids this.
-   Evaluating on training data and assuming that is test performance.
-   Comparing models using different test rows.
