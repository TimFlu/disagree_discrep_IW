import io
import inspect
import pickle

import numpy as np
import pandas as pd
import pytest

from src.plot.compare_iw_hybrids import _NumpyCompatUnpickler, _read_results_pickle, load_results


def test_numpy2_array_globals_on_numpy1(tmp_path, monkeypatch):
    frame = pd.DataFrame({'lower_bound': [0.4, 0.6], 'trg_accuracy': [0.5, 0.7]})
    payload = pickle.dumps(frame, protocol=0).replace(b'numpy.core', b'numpy._core')
    assert b'numpy._core' in payload
    path = tmp_path / 'numpy2.pkl'
    path.write_bytes(payload)

    def missing_numpy_core(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'numpy._core'", name='numpy._core')

    monkeypatch.setattr(pd, 'read_pickle', missing_numpy_core)
    pd.testing.assert_frame_equal(_read_results_pickle(path), frame)


def test_normal_pickle_and_historical_defaults(tmp_path):
    frame = pd.DataFrame({'lower_bound': [0.4], 'trg_accuracy': [0.5]})
    path = tmp_path / 'normal.pkl'
    frame.to_pickle(path)
    pd.testing.assert_frame_equal(_read_results_pickle(path), frame)
    loaded = load_results([path])
    assert loaded.loc[0, 'prediction_method'] == 'dis2_historical'
    assert np.isnan(loaded.loc[0, 'iw_threshold'])
    assert loaded.loc[0, 'lower_bound'] == 0.4


def test_unrelated_missing_module_is_preserved(tmp_path, monkeypatch):
    def missing_dependency(*args, **kwargs):
        raise ModuleNotFoundError('missing dependency', name='unrelated_dependency')

    monkeypatch.setattr(pd, 'read_pickle', missing_dependency)
    with pytest.raises(ModuleNotFoundError, match='missing dependency'):
        _read_results_pickle(tmp_path / 'unused.pkl')


@pytest.mark.skipif('na_value' in inspect.signature(pd.StringDtype).parameters,
                    reason='Compatibility adapter is only needed for older pandas')
def test_newer_pandas_string_state():
    loader = _NumpyCompatUnpickler(io.BytesIO())
    dtype = loader.find_class('pandas', 'StringDtype')('python', np.nan)
    array_type = loader.find_class('pandas.arrays', 'StringArray')
    array = array_type.__new__(array_type)
    array.__setstate__((dtype, np.array(['logits', 'PCA1'], dtype=object)))
    assert array.tolist() == ['logits', 'PCA1']
