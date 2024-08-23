import pandas as pd
import pytest


@pytest.fixture
def sample_test_data():
    data = {
        "id": [0, 1, 2, 3, 4],
        "medexpense": [206.84, 184.19, 175.68, 180.04, 218.86],
        "age": [57, 67, 62, 67, 68],
        "dcron": [5, 4, 4, 5, 5],
        "income": [57.200001, 67.199997, 62.000000, 67.199997, 68.000000],
        "plan_emerald": [False, False, False, False, False],
        "plan_golden": [False, False, False, False, False],
    }

    return pd.DataFrame(data)


@pytest.fixture
def route_files():
    file_path = "/Users/rabernar/Visual_Studio/Git_DataScience_REPOs/Health_Insurance_Plan_Classification/data/processed/planosaude_processed.csv"
    sep_char = ","

    return file_path, sep_char
