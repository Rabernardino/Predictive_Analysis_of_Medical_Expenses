import pandas as pd

from health_insurance_plan_classification.utils import feature_selection


def test_feature_selection(sample_test_data):
    selected_feature_df, target_series = feature_selection(
        sample_test_data, target="medexpense", cutoff=0.1
    )

    assert isinstance(selected_feature_df, pd.DataFrame)
    assert "medexpense" not in list(selected_feature_df.columns)
    assert "medexpense" == target_series.name
