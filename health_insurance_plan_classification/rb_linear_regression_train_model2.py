import mlflow
import statsmodels.api as sm

from health_insurance_plan_classification.utils import (
    breusch_pagan_test,
    feature_selection,
    get_train_data,
    load_data,
    pvalue_test,
    shapiro_francia_test,
)

# Importing data
data = load_data("../data/processed/planosaude_processed.csv", sep_char=",")

# Separating the features and the target variable
X, y = feature_selection(dataframe=data, target="medexpense", cutoff=0.1)

X_train, y_train = get_train_data(X, y, 0.25, 42)

# Initializing the regression model
X_train = sm.add_constant(X_train)
regression_model = sm.OLS(y_train, X_train).fit()

# Evalueting the pvalues and creating a new X_train set
X_train_ajusted = pvalue_test(regression_model, X_train)

# Performing the second linear model version
regression_model_2 = sm.OLS(y_train, X_train_ajusted).fit()


# Shapiro-Francia Test checking the residuals normality
method, statistics_W, statistics_Z, Pvalue_Normality = shapiro_francia_test(
    regression_model_2
)

# Breusch-Pagan heteroskedasticity test
chisq_bp, pvalue_bp = breusch_pagan_test(regression_model_2)


# Creating Experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.create_experiment("Linear Regression Experiment_V2")

# Adding to mlflow the results from the second linear regression model
mlflow.set_experiment("Linear Regression Experiment_V2")


with mlflow.start_run(run_name="Second Linear Model"):
    # Saving the model summary as a txt file
    with open("regression_model_2_summary.txt", "w") as file:
        file.write(regression_model_2.summary().as_text())

    mlflow.log_artifact("regression_model_2_summary.txt")

    # Sent the metrics outputs from the model summary
    mlflow.log_metric("r_squared", regression_model_2.rsquared)
    mlflow.log_metric("r_squared_adj", regression_model_2.rsquared_adj)
    mlflow.log_metric("f_statistic", regression_model_2.fvalue)
    mlflow.log_metric("f_pvalue", regression_model_2.f_pvalue)
    mlflow.log_metric("aic", regression_model_2.aic)
    mlflow.log_metric("bic", regression_model_2.bic)
    mlflow.log_metric("log_likelihood", regression_model_2.llf)

    for i in range(len(regression_model_2.pvalues.index) - 1):
        mlflow.log_metric(
            f"pvalue_{regression_model_2.pvalues.index[i]}",
            regression_model_2.pvalues[i],
        )

    for param, value in regression_model_2.params.items():
        mlflow.log_param(f"coef_{param}", value)

    mlflow.statsmodels.log_model(
        regression_model_2, artifact_path="Linear_Regression_model"
    )

    # Sent the Shapiro-Francia normality test
    mlflow.log_metric("Shapiro-Francia Normality - pvalue", Pvalue_Normality)
    mlflow.log_metric("Shapiro-Francia Normality - Z", statistics_Z)
    mlflow.log_metric("Shapiro-Francia Normality - W", statistics_W)

    # Sent the Breusch-Pagan heteroskedasticity test
    mlflow.log_metric("Breusch-Pagan p-value", pvalue_bp)
    mlflow.log_metric("Breusch-Pagan chi2", chisq_bp)
