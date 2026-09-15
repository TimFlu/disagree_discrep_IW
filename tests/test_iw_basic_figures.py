import numpy as np
import pandas as pd
import pytest

from src.plot.iw_basic_figures import summarize


def test_metrics_exclude_unsupported_and_preserve_thresholds():
    data = pd.DataFrame({
        'prediction_method': ['iw_overlap_critic'] * 5,
        'iw_threshold': [1., 1., 1., 2., 2.],
        'status': ['ok', 'ok', 'unsupported', 'unsupported', 'ok'],
        'lower_bound': [.4, .8, .9, np.nan, .5],
        'trg_accuracy': [.5, .6, .1, .7, np.nan],
    })
    table = summarize(data).set_index('iw_threshold')
    assert table.loc[1., 'n_total'] == 3
    assert table.loc[1., 'n_evaluated'] == 2
    assert table.loc[1., 'n_unsupported'] == 1
    assert table.loc[1., 'mae'] == pytest.approx(.15)
    assert table.loc[1., 'empirical_coverage'] == .5
    assert table.loc[2., 'n_evaluated'] == 0
    assert np.isnan(table.loc[2., 'empirical_coverage'])


def test_dis2_without_threshold_and_equal_prediction_is_covered():
    data = pd.DataFrame({
        'prediction_method': ['dis2_historical'], 'iw_threshold': [np.nan],
        'status': ['ok'], 'lower_bound': [.5], 'trg_accuracy': [.5],
    })
    row = summarize(data).iloc[0]
    assert row.n_evaluated == 1
    assert row.mae == 0
    assert row.empirical_coverage == 1
