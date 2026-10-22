# Regression Models --- Notebook Roadmap

## What we studied

This notebook collection uses the **Auto MPG** dataset to predict a
car's fuel efficiency (`mpg`) from its features.

Models covered: 1. Linear Regression 2. Polynomial Regression 3.
Decision Tree Regressor 4. Random Forest Regressor

The main workflow is:

1.  Load and clean the dataset.
2.  Choose input features `X` and target `y`.
3.  Split data into training, validation (when tuning), and test sets.
4.  Fit a model using training data.
5.  Predict on unseen data.
6.  Evaluate with R², MAE, and RMSE.
7.  Tune hyperparameters using validation data, not the test set.
8.  Compare models using the same test set.

## Core terms

-   **Feature / input (`X`)**: information used to make a prediction,
    such as horsepower or weight.
-   **Target (`y`)**: the value we want to predict, here `mpg`.
-   **Training set**: data used to learn model parameters.
-   **Validation set**: data used to choose models or hyperparameters.
-   **Test set**: held-out data used for the final evaluation.
-   **Parameter**: learned by the model during training.
-   **Hyperparameter**: a setting chosen before or around training, such
    as `max_depth` or `n_estimators`.
-   **Overfitting**: model learns training-specific details that do not
    generalize well.
-   **Generalization**: how well a model predicts new, unseen examples.

## Suggested notebook order

1.  Run the shared data preparation in
    `01_Data_Preparation_and_Splitting.md`.
2.  Study each model's individual notebook.
3.  Use `05_Model_Comparison_and_Tuning.md` to compare and tune models.
4.  Keep `06_Complete_Regression_Guide_A_to_Z.md` as the complete
    reference.

## Important consistency rule

For a fair model comparison, use the **same train/test rows** for each
model. If you change the features, recreate the split. Do not tune
repeatedly against the test set.
