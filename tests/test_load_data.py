import pandas as pd
import pytest

from health_insurance_plan_classification.utils import load_data


def test_load_data(route_files):
    file_path, sep_char = route_files

    df = load_data(file_path=file_path, sep_char=sep_char)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_not_found():
    file_path = "../data/no_file/unfound.csv"
    sep_char = ","

    with pytest.raises(ValueError):
        load_data(file_path, sep_char)
