import pandas as pd
from pandas.testing import assert_frame_equal
from functools import partial

import config
DATA_DIR = config.DATA_DIR

import bsm_pricer

def test_bsm_call_price():
    kwargs = {
        "S": 100,
        "K": 5,
        "r": 0.05,
        "T": 1.75,
        "sigma": 0.65,
    }

    # Expected output:
    # European call option price: 95.41960799088974
    expected_output = 95.41960799088974
    
    assert (bsm_pricer.european_call_price(**kwargs) - expected_output) < 1e-10
    
def test_bsm_put_price():
    kwargs = {
        "S": 100,
        "K": 5,
        "r": 0.05,
        "T": 1.75,
        "sigma": 0.65,
    }

    # Expected output:
    # European put option price: 0.0007023491441319144
    expected_output = 0.0007023491441319144
    
    assert (bsm_pricer.european_put_price(**kwargs) - expected_output) < 1e-10
    