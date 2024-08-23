import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sfrancia import shapiroFrancia
from sklearn.model_selection import train_test_split


def load_data(file_path, sep_char):
    try:
        df = pd.read_csv(file_path, sep=sep_char)
        return df

    except Exception as e:
        raise ValueError(f"Erro na leitura dos dados - {e}")


def feature_selection(dataframe, target: str, cutoff=0.1):
    correlation = dataframe.corr()[target]
    selected_feature = correlation[
        abs(dataframe.corr()[target]) > cutoff
    ].index.tolist()

    selected_feature.remove(target)
    df_selected = dataframe[selected_feature]
    target_series = dataframe[target]

    return df_selected, target_series


def get_train_data(X, y, train_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )

    return X_train, y_train


def get_test_data(X, y, train_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )

    return X_test, y_test


def pvalue_test(model, X):
    variables_with_acceptable_pvalue = []

    for i in range(len(model.pvalues.index)):
        if model.pvalues[i] < 0.05:
            variables_with_acceptable_pvalue.append(model.pvalues.index[i])

    return X[variables_with_acceptable_pvalue]


def shapiro_francia_test(model_residual):
    method, statistics_W, statistics_Z, Pvalue = shapiroFrancia(
        model_residual.resid
    ).values()

    return method, statistics_W, statistics_Z, Pvalue


def breusch_pagan_test(model):
    df = pd.DataFrame({"yhat": model.fittedvalues, "resid": model.resid})

    df["up"] = (np.square(df.resid)) / np.sum(
        ((np.square(df.resid)) / df.shape[0])
    )

    modelo_aux = sm.OLS.from_formula("up ~ yhat", df).fit()

    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)

    anova_table["sum_sq"] = anova_table["sum_sq"] / 2

    chisq = anova_table["sum_sq"].iloc[0]

    p_value = stats.chi2.pdf(chisq, 1) * 2

    return chisq.round(4), p_value
