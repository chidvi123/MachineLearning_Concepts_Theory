# Multiple Linear Regression — Complete Notes

## 1. What Is Multiple Linear Regression?

Multiple Linear Regression is a supervised Machine Learning algorithm used to predict a continuous numerical target using **two or more input features**.

In Simple Linear Regression, we use one input feature.

In Multiple Linear Regression, we use multiple input features to predict the target.

For example, salary may depend on several factors:

- Years of experience
- Education level
- Age
- Skills or qualifications

The model learns the relationship between the available input features and salary from the training data.

**Important:** These are illustrative possible features. We should use only columns that actually exist in our chosen dataset.

## 2. Simple vs Multiple Linear Regression

| Simple Linear Regression | Multiple Linear Regression |
|---|---|
| Uses one feature | Uses two or more features |
| Fits a line in two-dimensional input-output space | Fits a hyperplane in a higher-dimensional space |
| Equation: \(\hat y=b_0+b_1x_1\) | Equation: \(\hat y=b_0+b_1x_1+\cdots+b_px_p\) |
| One coefficient for the input feature | One coefficient for each input feature |

## 3. Mathematical Foundation

The equation for Multiple Linear Regression is:

\[
\hat y=b_0+b_1x_1+b_2x_2+\cdots+b_px_p
\]

Where:

- \(\hat y\) = predicted target.
- \(b_0\) = intercept.
- \(b_1,b_2,\ldots,b_p\) = coefficients.
- \(x_1,x_2,\ldots,x_p\) = input features.
- \(p\) = number of features.

Each coefficient represents the model's predicted change in the target for a one-unit increase in that feature, **holding the other features constant**.

For example, a positive coefficient for experience means the model predicts a higher salary as experience increases, assuming the other included features remain fixed. This describes the fitted relationship, not necessarily a causal effect.

## 4. Features and Target

In Pandas, we select multiple columns for `X` and one column for `y`.

```python
X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]
```

- `X` is a two-dimensional DataFrame.
- `y` is typically a one-dimensional Series.
- Each row is an observation.
- Each column in `X` is an input feature.

Replace the placeholder names with actual columns from the dataset you use.

## 5. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

- `test_size=0.2` reserves 20% of rows for testing.
- `random_state=42` makes the random split reproducible.
- The model learns from `X_train` and `y_train`.
- `X_test` is used to make predictions for evaluation against `y_test`.

For reliable evaluation, preprocessing steps that learn statistics from data should be fitted using training data only.

## 6. Train Multiple Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

`fit()` learns the coefficients and intercept that minimize the sum of squared residuals for the training data.

Inspect the learned parameters:

```python
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Features:", X.columns.tolist())
```

The coefficient order corresponds to the feature order in `X`.

## 7. Make Predictions

```python
y_pred = model.predict(X_test)

print("Actual values:")
print(y_test.head())

print("Predicted values:")
print(y_pred[:5])
```

For a new observation, supply all the features used during training, in the same column order.

```python
new_data = pd.DataFrame({
    "feature_1": [value_1],
    "feature_2": [value_2],
    "feature_3": [value_3]
})

prediction = model.predict(new_data)
print(prediction)
```

The names and values above are placeholders; replace them with the real feature names and appropriate values from your dataset.

## 8. Cost Function

Multiple Linear Regression commonly minimizes Mean Squared Error:

\[
J(\mathbf b)=\frac{1}{n}\sum_{i=1}^{n}
(\hat y_i-y_i)^2
\]

The prediction for observation \(i\) is:

\[
\hat y_i=b_0+\sum_{j=1}^{p}b_jx_{ij}
\]

The objective is to find coefficients that minimize the squared differences between actual and predicted target values.

## 9. Gradient Descent

Gradient Descent can optimize the coefficients of Multiple Linear Regression.

For the MSE cost function:

\[
\frac{\partial J}{\partial b_j}
=\frac{2}{n}\sum_{i=1}^{n}
(\hat y_i-y_i)x_{ij}
\]

For the intercept:

\[
\frac{\partial J}{\partial b_0}
=\frac{2}{n}\sum_{i=1}^{n}
(\hat y_i-y_i)
\]

Parameter updates are:

\[
b_j\leftarrow b_j-\alpha
\frac{\partial J}{\partial b_j}
\]

\[
b_0\leftarrow b_0-\alpha
\frac{\partial J}{\partial b_0}
\]

Here, \(\alpha\) is the learning rate.

The process repeats:

1. Calculate predictions using all features.
2. Calculate errors and cost.
3. Calculate the gradient for every coefficient and the intercept.
4. Update all parameters.
5. Repeat until the cost converges or the iteration limit is reached.

### Ordinary Least Squares vs Gradient Descent

Scikit-learn's standard `LinearRegression()` estimator uses Ordinary Least Squares, not iterative Gradient Descent. Both aim to minimize squared residuals, but they use different optimization approaches.

Gradient Descent is particularly useful to understand when learning optimization and training algorithms for larger machine learning models.

## 10. Model Evaluation

```python
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
```

### R² Score

Measures performance relative to predicting the mean of the test targets.

- \(R^2=1\): perfect predictions.
- \(R^2=0\): equivalent to the mean-prediction baseline.
- \(R^2<0\): worse than that baseline.

### MAE

Average absolute prediction error, expressed in the target's units.

### MSE

Average squared prediction error. Larger errors receive greater penalties.

### RMSE

Square root of MSE, expressed in the target's units.

For comparisons on the same test set, lower error metrics and a higher R² are generally preferred.

## 11. Assumptions

1. The relationship between predictors and the target is adequately represented by a linear model.
2. Errors are independent in the relevant sampling setup.
3. Error variance is approximately constant for standard inference.
4. Errors are approximately normal when required for certain small-sample statistical inference.
5. Strong multicollinearity should be checked because it can make individual coefficient estimates unstable.

Violations do not always prevent prediction, but they can affect reliability and interpretation.

## 12. Multicollinearity

Multicollinearity occurs when input features are strongly correlated with one another.

For example, two measurements that contain nearly the same information can make it difficult to determine their individual coefficients reliably.

Potential effects:

- Coefficients become unstable.
- Small data changes can lead to large coefficient changes.
- Individual coefficient interpretations become less reliable.

Possible approaches include checking correlations, calculating Variance Inflation Factors (VIF), removing redundant features when justified, or using regularized models such as Ridge Regression.

## 13. Advantages and Limitations

### Advantages

- Uses multiple predictors.
- Can model more complex relationships than a single-feature linear model.
- Coefficients are relatively interpretable.
- Provides a strong baseline for regression problems.

### Limitations

- Assumes a linear relationship in the predictors.
- Sensitive to outliers.
- Multicollinearity can destabilize coefficients.
- Adding irrelevant features can hurt generalization.
- Predictions outside the training data range may be unreliable.

## 14. Common Interview Questions

**1. What is Multiple Linear Regression?**

A supervised learning algorithm that predicts a continuous target using two or more features.

**2. How does it differ from Simple Linear Regression?**

It uses multiple input features rather than one.

**3. What does a coefficient mean?**

The predicted change in the target for a one-unit increase in a feature while holding the other features constant.

**4. What is multicollinearity?**

Strong correlation between input features, which can make coefficient estimates unstable.

**5. Can Multiple Linear Regression use categorical variables?**

Yes, after appropriate encoding, such as one-hot encoding for nominal categories.

**6. Does adding more features always improve the model?**

Training fit may improve, but test performance does not necessarily improve. Additional features can introduce noise or overfitting.

**7. Does `LinearRegression()` use Gradient Descent?**

No. Scikit-learn's standard implementation uses Ordinary Least Squares.

**8. What is the difference between R² and adjusted R²?**

R² measures fit relative to a mean baseline. Adjusted R² accounts for the number of predictors and can help compare models with different numbers of features.

## 15. Quick Revision

- Multiple Linear Regression predicts a continuous target using multiple features.
- Each feature has a learned coefficient.
- The intercept is the prediction when all features are zero.
- `X` is a two-dimensional feature matrix; `y` is the target.
- The model minimizes squared residuals.
- `LinearRegression()` uses Ordinary Least Squares by default.
- Gradient Descent is an alternative optimization method.
- MAE, MSE, RMSE, and R² help evaluate predictions.
- Multicollinearity affects coefficient stability.
- More features do not automatically mean better test performance.

