import mlflow
import statsmodels.api as sm

from health_insurance_plan_classification.utils import (
    feature_selection,
    get_train_data,
    load_data,
)

# Importing data
data = load_data("../data/processed/planosaude_processed.csv", sep_char=",")

# Separating the features and the target variable
X, y = feature_selection(dataframe=data, target="medexpense", cutoff=0.1)

X_train, y_train = get_train_data(X, y, 0.25, 42)

# Initializing the regression model
X_train = sm.add_constant(X_train)

regression_model_1 = sm.OLS(y_train, X_train).fit()


# Adding to mlflow the results from the firts linear regression model
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Linear Regression Experiment_V1")


with mlflow.start_run(run_name="First Linear Model"):

    # Saving the model summary as a txt file
    with open("regression_model_1_summary.txt", "w") as file:
        file.write(regression_model_1.summary().as_text())

    mlflow.log_artifact("regression_model_1_summary.txt")

    # Sent the metrics outputs from the model summary
    mlflow.log_metric("r_squared", regression_model_1.rsquared)
    mlflow.log_metric("r_squared_adj", regression_model_1.rsquared_adj)
    mlflow.log_metric("f_statistic", regression_model_1.fvalue)
    mlflow.log_metric("f_pvalue", regression_model_1.f_pvalue)
    mlflow.log_metric("aic", regression_model_1.aic)
    mlflow.log_metric("bic", regression_model_1.bic)
    mlflow.log_metric("log_likelihood", regression_model_1.llf)

    for i in range(len(regression_model_1.pvalues.index) - 1):
        mlflow.log_metric(
            f"pvalue_{regression_model_1.pvalues.index[i]}",
            regression_model_1.pvalues[i],
        )

    for param, value in regression_model_1.params.items():
        mlflow.log_param(f"coef_{param}", value)

    # Sent the model
    mlflow.statsmodels.log_model(
        regression_model_1, artifact_path="Linear_Regression_model"
    )
