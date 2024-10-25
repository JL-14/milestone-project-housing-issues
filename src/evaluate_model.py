import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# def regression_performance(X_train, y_train, X_test, y_test, pipeline):
#     st.write("Model Evaluation\n")
#     st.write("* Train Set")
#     regression_evaluation(X_train, y_train, pipeline)
#     st.write("* Test Set")
#     regression_evaluation(X_test, y_test, pipeline)

# from sklearn.metrics import r2_score

# def regression_performance(X_train, y_train, X_test, y_test, model):
#     # Predictions
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)

#     # Calculate performance metrics
#     train_r2 = r2_score(y_train, y_train_pred)
#     test_r2 = r2_score(y_test, y_test_pred)

#     # Return a dictionary with the performance metrics
#     return {
#         "Train R2 Score": train_r2,
#         "Test R2 Score": test_r2
#     }

# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import numpy as np

def regression_performance(X_train, y_train, X_test, y_test, model):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate performance metrics for Train Set
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    # Calculate performance metrics for Test Set
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    # Return a dictionary with the performance metrics
    return {
        "Train R2 Score": train_r2,
        "Train Mean Absolute Error": train_mae,
        "Train Mean Squared Error": train_mse,
        "Train Root Mean Squared Error": train_rmse,
        "Test R2 Score": test_r2,
        "Test Mean Absolute Error": test_mae,
        "Test Mean Squared Error": test_mse,
        "Test Root Mean Squared Error": test_rmse
    }

def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    print('R2 Score:', r2_score(y, prediction).round(3))
    print('Mean Absolute Error:', mean_absolute_error(y, prediction).round(3))
    print('Mean Squared Error:', mean_squared_error(y, prediction).round(3))
    print('Root Mean Squared Error:', np.sqrt(
        mean_squared_error(y, prediction)).round(3))
    print("\n")


# def regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5):
#     pred_train = pipeline.predict(X_train)
#     pred_test = pipeline.predict(X_test)

#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
#     sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0])
#     sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0])
#     axes[0].set_xlabel("Actual")
#     axes[0].set_ylabel("Predictions")
#     axes[0].set_title("Train Set")

#     sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1])
#     sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[1])
#     axes[1].set_xlabel("Actual")
#     axes[1].set_ylabel("Predictions")
#     axes[1].set_title("Test Set")

#     plt.show()



# def model_evaluation(X_train=X_train, y_train=y_train,
#                         X_test=X_test, y_test=y_test,
#                         pipeline=pipeline_best):
#     st.write: "Train Set"
#     regression_evaluation(X_train, y_train, pipeline_best)

#     st.wrire: "Test Set"