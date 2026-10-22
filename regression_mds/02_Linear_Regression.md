# Linear Regression — Complete Notes

## 1. What Is Linear Regression?

Linear Regression is a **supervised Machine Learning algorithm** used to predict a continuous numerical value by learning the relationship between input features and a target variable.

### Our practice example: Salary Prediction

We use the **Salary vs Years of Experience** dataset.

- `YearsExperience` — independent variable (input feature).
- `Salary` — dependent variable (target).
- Goal — predict a person's salary based on their years of experience.

For example, the model learns the relationship between experience and salary from the training data and uses that relationship to predict salaries for unseen inputs.

### Regression vs Classification

| Regression | Classification |
|---|---|
| Predicts continuous numerical values | Predicts discrete classes |
| Example: predicting salary | Example: predicting whether an email is spam |
| Metrics: MAE, MSE, RMSE, R² | Metrics: accuracy, precision, recall, F1 |

### Types of Linear Regression

**Simple Linear Regression:** Uses one input feature.

\[
\hat{y}=wx+b
\]

**Multiple Linear Regression:** Uses two or more input features.

\[
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
\]

Here:
- \(\hat{y}\) = predicted salary.
- \(x\) = years of experience.
- \(w\) = weight or coefficient.
- \(b\) = bias or intercept.

---

## 2. Import Libraries

### Jupyter Notebook — Cell 1

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
```

## 3. Load and Explore the Dataset

### Jupyter Notebook — Cell 2

```python
df = pd.read_csv("Salary_Data.csv")

df.head()
```

### Jupyter Notebook — Cell 3

```python
print(df.shape)
print(df.info())
print(df.isnull().sum())
```

The exact dataset filename should match the file we used during practice. Here, `Salary_Data.csv` is the filename assumed in this notebook.

## 4. Separate Features and Target

### Jupyter Notebook — Cell 4

```python
X = df[["YearsExperience"]]
y = df["Salary"]

print(X.head())
print(y.head())
```

### Why do we use double brackets for X?

```python
X = df[["YearsExperience"]]
```

Double brackets select a DataFrame with shape `(number_of_rows, 1)`.

```python
y = df["Salary"]
```

Single brackets select a Pandas Series containing the target values.

Scikit-learn generally expects the feature matrix `X` to be two-dimensional and the target `y` to be one-dimensional for this regression task.

- `X` contains the input used to make predictions.
- `y` contains the correct answers the model learns from.

## 5. Split the Dataset

### Jupyter Notebook — Cell 5

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

### Understanding train-test split

- `test_size=0.2` assigns 20% of the rows to the test set and 80% to the training set.
- `random_state=42` makes the random split reproducible.
- The model learns from the training set.
- The test set evaluates how well the model predicts unseen data.

The model must not learn from the test targets during training.

---

## 6. Mathematical Foundation

Linear Regression tries to find the best-fitting straight line through the training data.

\[
\hat{y}=wx+b
\]

For our salary prediction problem:

\[
\widehat{\text{Salary}}=w(\text{YearsExperience})+b
\]

### Weight (slope)

The weight describes how much the predicted salary changes when years of experience increases by one unit.

For example, if the learned coefficient were ₹50,000 per year, an additional year of experience would increase the model's prediction by ₹50,000, assuming the same linear relationship. This is only an illustrative value, not our measured result.

### Bias (intercept)

The intercept is the model's predicted salary when years of experience is zero.

It is a mathematical starting point for the line and should not automatically be interpreted as a realistic starting salary.

### Prediction error and residual

\[
e_i=y_i-\hat{y}_i
\]

- \(y_i\) = actual salary.
- \(\hat{y}_i\) = predicted salary.
- \(e_i\) = residual or prediction error.

A positive residual means the actual salary is higher than predicted. A negative residual means the actual salary is lower than predicted.

---

## 7. Cost Function: Mean Squared Error

Linear Regression needs a way to measure how far its predictions are from the actual targets.

Mean Squared Error (MSE) is:

\[
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}
\left(\hat{y}_i-y_i\right)^2
\]

Where:
- \(n\) = number of training samples.
- \(y_i\) = actual salary.
- \(\hat{y}_i\) = predicted salary.
- \(J(w,b)\) = cost for the current values of weight and bias.

### Why square the errors?

1. Squaring prevents positive and negative errors from cancelling each other.
2. Larger errors receive a larger penalty.
3. The squared-error cost is differentiable, allowing us to calculate gradients.

**Objective:** Find the weight and bias that minimize the cost function.

A lower training MSE means the predictions are closer to the training targets on average in squared-error terms. It does not guarantee better performance on unseen data.

---

# 8. Gradient Descent — The Core Concept

## 8.1 What Is Gradient Descent?

Gradient Descent is an **optimization algorithm** that iteratively updates model parameters to reduce a cost function.

In Linear Regression, the parameters are:
- Weight \(w\)
- Bias \(b\)

The cost function tells us how poorly the model is performing. The gradient tells us how the cost changes when the parameters change.

Gradient Descent uses that information to update the parameters in a direction that reduces the cost.

### Intuition

Imagine standing on a hill in fog and trying to reach the lowest point.

1. Estimate the direction in which the hill slopes upward.
2. Move in the opposite direction.
3. Repeat until further updates make very little improvement.

In Linear Regression, the hill represents the cost function, and the position represents the current weight and bias.

## 8.2 What Is a Gradient?

A gradient contains the partial derivatives of the cost function with respect to the parameters.

For our two parameters:

\[
\nabla J(w,b)=
\left[
\frac{\partial J}{\partial w},
\frac{\partial J}{\partial b}
\right]
\]

- \(\frac{\partial J}{\partial w}\) tells us how the cost changes as the weight changes.
- \(\frac{\partial J}{\partial b}\) tells us how the cost changes as the bias changes.

A positive gradient means increasing that parameter locally increases the cost. A negative gradient means increasing that parameter locally decreases the cost.

Gradient Descent moves in the opposite direction of the gradient.

## 8.3 Deriving the Gradient

For one input feature, the prediction is:

\[
\hat{y}_i=wx_i+b
\]

The cost function is:

\[
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}
(wx_i+b-y_i)^2
\]

### Derivative with respect to weight

Using the chain rule:

\[
\frac{\partial J}{\partial w}
=\frac{2}{n}\sum_{i=1}^{n}
(wx_i+b-y_i)x_i
\]

### Derivative with respect to bias

\[
\frac{\partial J}{\partial b}
=\frac{2}{n}\sum_{i=1}^{n}
(wx_i+b-y_i)
\]

The derivatives measure how the average squared-error cost changes with each parameter.

## 8.4 Parameter Update Equations

Gradient Descent updates the parameters as follows:

\[
w_{\text{new}}=w_{\text{old}}-
\alpha\frac{\partial J}{\partial w}
\]

\[
b_{\text{new}}=b_{\text{old}}-
\alpha\frac{\partial J}{\partial b}
\]

Here, \(\alpha\) is the **learning rate**.

The minus sign is important: we move opposite to the gradient to reduce the cost.

Each iteration performs these steps:

1. Predict salaries using the current weight and bias.
2. Calculate the cost.
3. Calculate both gradients.
4. Update the weight and bias.
5. Repeat.

Both parameters are updated using the gradients calculated from the same current parameter values.

## 8.5 Learning Rate

The learning rate controls the size of each parameter update.

| Learning rate | Possible behavior |
|---|---|
| Too small | Learning is slow and needs many iterations |
| Appropriate | Cost generally decreases and converges |
| Too large | Updates may overshoot the minimum or diverge |

A larger learning rate is not necessarily better. We choose it based on how training behaves.

## 8.6 Implement Gradient Descent from Scratch

For this demonstration, we will use the same Salary vs Years of Experience dataset. Since the numerical scales of salary and experience differ substantially, we standardize both arrays for numerical stability.

This scratch implementation optimizes MSE for the standardized target. The learned weight and bias are therefore in standardized units, not directly in salary units.

### Jupyter Notebook — Cell 6

```python
# Convert the training and test data to NumPy arrays
x_train = X_train["YearsExperience"].to_numpy(dtype=float)
x_test = X_test["YearsExperience"].to_numpy(dtype=float)

y_train_array = y_train.to_numpy(dtype=float)
y_test_array = y_test.to_numpy(dtype=float)

# Standardize using training-set statistics only
x_mean = x_train.mean()
x_std = x_train.std()

y_mean = y_train_array.mean()
y_std = y_train_array.std()

x_train_scaled = (x_train - x_mean) / x_std
x_test_scaled = (x_test - x_mean) / x_std

y_train_scaled = (y_train_array - y_mean) / y_std
y_test_scaled = (y_test_array - y_mean) / y_std
```

### Jupyter Notebook — Cell 7

```python
# Initialize parameters
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 2000
n = len(x_train_scaled)

cost_history = []

for epoch in range(epochs):
    # Predictions
    y_pred_scaled = w * x_train_scaled + b

    # Errors and cost
    errors = y_pred_scaled - y_train_scaled
    cost = np.mean(errors ** 2)
    cost_history.append(cost)

    # Gradients
    dw = (2 / n) * np.sum(errors * x_train_scaled)
    db = (2 / n) * np.sum(errors)

    # Update both parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Learned standardized weight:", w)
print("Learned standardized bias:", b)
print("Final training MSE:", cost_history[-1])
```

### Jupyter Notebook — Cell 8

```python
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Gradient Descent: Cost vs Iterations")
plt.show()
```

### What should the graph show?

The cost should generally decrease as iterations progress and then level off near a minimum. Small fluctuations can occur in other optimization settings, but this full-batch implementation on the standardized linear regression problem should normally show a smooth decrease.

### Jupyter Notebook — Cell 9

```python
# Predict in standardized units
y_pred_scaled = w * x_test_scaled + b

# Convert predictions back to salary units
y_pred_gd = y_pred_scaled * y_std + y_mean

print("First 5 actual salaries:", y_test_array[:5])
print("First 5 predicted salaries:", y_pred_gd[:5])
```

## 8.7 A Small Numerical Example

Suppose, for illustration, the current weight is \(w=2\), the current bias is \(b=1\), and the gradients are:

\[
\frac{\partial J}{\partial w}=4,\qquad
\frac{\partial J}{\partial b}=2
\]

If the learning rate is \(0.1\):

\[
w_{\text{new}}=2-(0.1)(4)=1.6
\]

\[
b_{\text{new}}=1-(0.1)(2)=0.8
\]

These values are only an illustration of the update rule, not the results from our salary dataset.

## 8.8 Types of Gradient Descent

### Batch Gradient Descent

Uses all training examples to calculate the gradient for each update.

- Stable, consistent updates for a fixed dataset.
- Can be expensive for very large datasets.

Our from-scratch implementation above uses Batch Gradient Descent.

### Stochastic Gradient Descent (SGD)

Uses one training example to calculate each update.

- Frequent updates.
- Noisy optimization path.
- Can be useful for very large datasets and online learning.

### Mini-batch Gradient Descent

Uses a small batch of examples for each update.

- Balances computational efficiency and update frequency.
- Commonly used to train neural networks.

## 8.9 Convergence and Common Problems

**Convergence:** The cost stops improving significantly or parameter updates become sufficiently small.

**Learning rate too high:** The cost may oscillate or increase.

**Learning rate too low:** The cost may decrease very slowly.

**Feature scale differences:** Very different numerical scales can make optimization poorly conditioned. Standardization often helps.

**Local minima:** For ordinary Linear Regression with squared-error loss, the cost is a convex quadratic function. Gradient Descent with a suitable learning rate can reach a global minimum. The non-convex local-minimum difficulties encountered in other models do not apply in the same way here.

---

## 9. How Does Scikit-Learn Train Linear Regression?

### Jupyter Notebook — Cell 10

```python
model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
```

`LinearRegression()` in scikit-learn uses **Ordinary Least Squares (OLS)** as its standard fitting method. It does not use the iterative Gradient Descent implementation shown above.

OLS finds coefficients that minimize the sum of squared residuals. In typical dense-data settings, scikit-learn solves this using numerical linear algebra.

### OLS vs Gradient Descent

| Ordinary Least Squares | Gradient Descent |
|---|---|
| Solves the least-squares problem numerically | Iteratively updates parameters |
| Standard method used by `LinearRegression()` | Must be implemented or selected separately |
| Does not require a learning rate | Requires a learning rate |
| Convenient for many small and medium linear regression problems | Useful for iterative optimization and large-scale learning |

Both methods aim to minimize squared prediction errors, but they reach the solution differently.

---

## 10. Make Predictions

### Jupyter Notebook — Cell 11

```python
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "YearsExperience": X_test["YearsExperience"].to_numpy(),
    "ActualSalary": y_test.to_numpy(),
    "PredictedSalary": y_pred
})

results.head()
```

`predict()` uses the learned coefficient and intercept to produce predictions for new feature values.

### Predict a salary for a new input

### Jupyter Notebook — Cell 12

```python
new_experience = pd.DataFrame({
    "YearsExperience": [5.0]
})

predicted_salary = model.predict(new_experience)

print("Predicted salary:", predicted_salary[0])
```

This is the model's estimate for five years of experience. It is not a guarantee of the salary a real person will earn.

---

## 11. Evaluate the Model

### Jupyter Notebook — Cell 13

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

### R² Score — Coefficient of Determination

R² compares the model against predicting the mean of the test targets.

\[
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
\]

- \(R^2=1\): perfect predictions.
- \(R^2=0\): no improvement over predicting the test-set mean.
- \(R^2<0\): worse than predicting the test-set mean.

R² is not a percentage accuracy score.

### MAE — Mean Absolute Error

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

It gives the average absolute prediction error in the target's units.

### MSE — Mean Squared Error

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

It penalizes larger errors more heavily because errors are squared.

### RMSE — Root Mean Squared Error

\[
RMSE=\sqrt{MSE}
\]

RMSE is expressed in the target's units, making it easier to compare with salary values.

**Important:** A lower MAE, MSE, or RMSE is generally better when comparing models on the same test set. A higher R² is generally better.

---

## 12. Visualize the Regression Line

### Jupyter Notebook — Cell 14

```python
plt.scatter(
    X_test["YearsExperience"],
    y_test,
    label="Actual salary"
)

x_line = np.linspace(
    X["YearsExperience"].min(),
    X["YearsExperience"].max(),
    100
)

x_line_df = pd.DataFrame({"YearsExperience": x_line})
y_line = model.predict(x_line_df)

plt.plot(x_line, y_line, color="red", label="Regression line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.legend()
plt.show()
```

### How to interpret the graph

- Each point represents an actual observation.
- The line represents the model's predicted salary at each experience value.
- Points closer to the line generally have smaller prediction errors.
- The line is the best fit according to the least-squares objective, not necessarily a perfect fit for every point.

---

## 13. Parameters vs Hyperparameters

### Parameters

Learned from the training data.

For Linear Regression:
- Coefficient \(w\)
- Intercept \(b\)

### Hyperparameters

Chosen before or during model configuration rather than directly learned as ordinary model coefficients.

Examples in this workflow:
- `test_size`
- `random_state`

These two control the data split, not the Linear Regression equation itself. The standard `LinearRegression()` estimator has relatively few hyperparameters to tune for ordinary use.

In Gradient Descent, `learning_rate` and `epochs` are hyperparameters.

---

## 14. Assumptions of Linear Regression

1. **Linearity:** The conditional relationship between the predictors and target is adequately represented by a linear model.
2. **Independence:** Observations or errors are independent in the relevant sampling setup.
3. **Homoscedasticity:** Error variance is approximately constant across the range of predictions.
4. **Error normality:** Normal errors are mainly important for exact small-sample statistical inference, not for simply calculating predictions.
5. **Low multicollinearity:** For multiple regression, highly correlated predictors can make individual coefficient estimates unstable.

A model can still produce predictions if some assumptions are violated, but interpretation and statistical inference may be affected.

---

## 15. Underfitting and Overfitting

### Underfitting

The model is too simple to capture the important pattern.

Possible signs:
- Poor training performance.
- Poor test performance.

### Overfitting

The model captures noise or peculiarities of the training set instead of general patterns.

Possible signs:
- Very good training performance.
- Much worse test performance.

Linear Regression can underfit nonlinear relationships. More complex models may fit them better, but complexity must be evaluated on unseen data.

---

## 16. Advantages and Limitations

### Advantages

- Simple and easy to interpret.
- Fast to train for many datasets.
- Coefficients help describe the fitted relationship.
- Provides a strong baseline for regression problems.

### Limitations

- A straight-line relationship may be too restrictive.
- Sensitive to outliers because squared errors penalize large deviations.
- Correlated features can make coefficient interpretation unstable.
- Extrapolating far beyond the training data can produce unreliable predictions.

---

## 17. Common Interview Questions

**1. What is Linear Regression?**

A supervised learning algorithm that models a continuous target as a linear function of input features.

**2. What is the difference between simple and multiple Linear Regression?**

Simple Linear Regression uses one feature; Multiple Linear Regression uses multiple features.

**3. What are weight and bias?**

The weight controls the contribution of a feature; the bias is the intercept.

**4. What is a cost function?**

A function that measures prediction error for the current model parameters.

**5. What is Gradient Descent?**

An iterative optimization algorithm that updates parameters in the direction opposite to the cost gradient.

**6. Why is there a minus sign in the update rule?**

Because we move opposite to the gradient to reduce the cost.

**7. What is the learning rate?**

A hyperparameter that controls the size of parameter updates.

**8. What happens if the learning rate is too high?**

Updates may overshoot the minimum, causing oscillation or divergence.

**9. Does scikit-learn's `LinearRegression()` use Gradient Descent?**

No. Its standard fitting method is Ordinary Least Squares.

**10. What is the difference between MAE and MSE?**

MAE averages absolute errors; MSE averages squared errors and penalizes large errors more heavily.

**11. Can R² be negative?**

Yes. On evaluation data, the model can perform worse than the mean-prediction baseline.

**12. Why do we split the data into training and testing sets?**

To evaluate performance on data that was not used to fit the model.

**13. What is the difference between a parameter and a hyperparameter?**

Parameters are learned from the training data; hyperparameters control configuration or the learning process.

**14. Why can feature scaling help Gradient Descent?**

It can improve the numerical conditioning of optimization and make updates more effective when features have very different scales.

**15. What is the difference between Batch, Stochastic, and Mini-batch Gradient Descent?**

They use all training samples, one sample, or a small group of samples per parameter update, respectively.

---

## 18. Quick Revision

- Linear Regression predicts a continuous numerical target.
- Simple Linear Regression uses one feature.
- The model equation is \(\hat{y}=wx+b\).
- The coefficient is the slope; the intercept is the bias.
- MSE measures average squared prediction error.
- Gradient Descent calculates gradients and updates parameters iteratively.
- The learning rate controls the size of each update.
- Batch Gradient Descent uses all training samples per update.
- Scikit-learn's `LinearRegression()` uses Ordinary Least Squares by default.
- `fit()` learns model parameters; `predict()` generates predictions.
- R² evaluates performance relative to a mean-prediction baseline.
- MAE and RMSE are expressed in the target's units.
- Always evaluate on data that was not used for training.

**Remember:** Gradient Descent is a general optimization algorithm, while Linear Regression is a model. Gradient Descent can train a Linear Regression model, but the standard scikit-learn `LinearRegression()` implementation uses OLS instead.
