# Polynomial Regression --- Complete Notes

## 1. What is Polynomial Regression?

Polynomial Regression extends linear regression by creating powers of
the input features. This lets the model fit curved relationships.

For one feature:

\[ `\hat{y}`{=tex} = b_0 + b_1x + b_2x\^2 \]

For degree 2, the model uses both (x) and (x\^2). The model is still
linear in its learned coefficients, but its curve can be nonlinear in
the original input.

## 2. Why use it?

A straight line may not capture the relationship between horsepower and
MPG. A polynomial model can represent a curve.

-   Degree 1: straight line (similar to ordinary Linear Regression).
-   Degree 2: quadratic curve.
-   Degree 3: cubic curve.
-   Higher degrees: more flexible curves, but also a greater risk of
    overfitting.

## 3. Fit Polynomial Regression of degree 2

``` python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
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

### What the pipeline does

1.  `PolynomialFeatures(degree=2)` creates `horsepower` and
    `horsepower²`.
2.  `include_bias=False` avoids adding a constant column because
    `LinearRegression` already learns an intercept.
3.  `LinearRegression()` learns coefficients from the expanded features.
4.  `make_pipeline(...)` applies these steps in order during fitting and
    prediction.

Using a pipeline helps prevent mistakes: the polynomial transformation
is fitted and applied consistently.

## 4. Plot the fitted curve

``` python
import matplotlib.pyplot as plt

X_sorted = X.sort_values("horsepower")

plt.scatter(X["horsepower"], y, alpha=0.6)
plt.plot(
    X_sorted["horsepower"],
    poly_model.predict(X_sorted),
    color="red"
)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression (Degree 2)")
plt.show()
```

Sorting the horsepower values before plotting ensures the line is drawn
from low horsepower to high horsepower rather than jumping around in row
order.

## 5. Choosing the degree

Treat degree as a hyperparameter: 1. Fit candidate degrees on training
data. 2. Compare validation scores. 3. Choose the degree with the best
validation performance, considering model simplicity too. 4. Evaluate
the selected model once on the test set.

A higher degree is not automatically better. It may fit training data
closely but generalize poorly.

## 6. Common mistakes

-   Creating polynomial features before splitting the data in workflows
    where the transformer learns data-dependent properties. Prefer a
    pipeline.
-   Forgetting to sort `X` before plotting the prediction line.
-   Assuming degree 2 must outperform degree 1 on unseen data.
-   Using a high degree without checking overfitting.
