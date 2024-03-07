"""
This module contains unit tests for the Level 3 filters in filter_option_data_03.py.
"""
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import config
import filter_option_data_03
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)



def test_level_3_final_output():
    """
    Test final output of the level 3 filter.
    """
    
    l3_data_iv_only, l3_filtered_options = filter_option_data_03.run_filter(_df=None, date_range='1996-01_2012-01', iv_only=False)    
    
    iv_only_remaining_options = len(l3_data_iv_only)
    iv_and_pcp_remaining_options = len(l3_filtered_options)
    pcp_filter_dropped_count = iv_and_pcp_remaining_options - iv_only_remaining_options
    
    '''
    Expected output:
    Remaining options after IV filter == 593431 - 38568 = 554863
    Remaining options after IV and PCP filter ==  461890
    PCP filter drop count == 92973
    ''' 
    
    assert (iv_only_remaining_options==554863, iv_and_pcp_remaining_options==461890, pcp_filter_dropped_count==92973)