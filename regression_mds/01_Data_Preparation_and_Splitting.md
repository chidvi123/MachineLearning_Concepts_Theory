# Auto MPG --- Data Preparation and Data Splitting

## 1. What are we predicting?

We want to predict `mpg` (miles per gallon), a continuous numeric value.
Therefore, this is a **regression** problem.

The dataset has these columns:

-   `mpg`: target
-   `cylinders`
-   `displacement`
-   `horsepower`
-   `weight`
-   `acceleration`
-   `model_year`
-   `origin`

The classic Auto MPG CSV may have no header row and uses `?` for missing
horsepower values.

## 2. Load and clean the data

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

# Ensure horsepower is numeric and remove rows with missing values.
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna().copy()

df.info()
print(df.head())
```

### Why clean `horsepower`?

In this dataset, missing horsepower values may be written as the text
`"?"`. Machine-learning estimators expect numeric inputs, so we convert
horsepower to numeric and remove rows that still contain missing values.

If the file was already loaded without `na_values="?"`, use:

``` python
df = df[df["horsepower"] != "?"].copy()
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna().copy()
```

Do not apply both alternatives blindly; use the version that matches how
you loaded the CSV.

## 3. Choose features and target

### Experiment A: horsepower only

``` python
X = df[["horsepower"]]
y = df["mpg"]
```

`X` uses double brackets because it remains a DataFrame with one
feature. `y` is a Series containing the target.

This experiment asks: how well can we predict MPG using only horsepower?

### Experiment B: all available features

``` python
X = df.drop(columns=["mpg"])
y = df["mpg"]
```

Now the model can use cylinders, displacement, horsepower, weight,
acceleration, model year, and origin.

The target must not be included in `X`; otherwise, the model would be
given the answer it is supposed to predict.

## 4. Simple train/test split

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

-   `test_size=0.2`: approximately 20% of rows go to the test set.
-   `random_state=42`: makes the random split reproducible.
-   `X_train` and `y_train` contain matching rows; each input row stays
    paired with its correct target.
-   `X_test` and `y_test` contain the held-out rows used for evaluation.

For 392 clean rows, scikit-learn assigns 79 rows to the test set and 313
to the training set when `test_size=0.2`.

## 5. Train/validation/test split for tuning

Use validation data when choosing hyperparameters. Keep the test set
separate until the final evaluation.

``` python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)
```

Conceptually: - 80% is kept for training + validation. - 20% is reserved
for testing. - Of the remaining 80%, 25% becomes validation.

For 392 rows, the approximate counts are: - Train: 234 - Validation:
79 - Test: 79

Scikit-learn rounds fractional test sizes up to a whole number, which
explains the exact counts.

## 6. Why use separate sets?

-   **Train:** learn model patterns.
-   **Validation:** choose settings and compare candidate models.
-   **Test:** estimate final performance on data that was not used to
    make those choices.

If you use the test score to repeatedly choose settings, the test set
gradually influences your choices and is no longer a clean final check.
