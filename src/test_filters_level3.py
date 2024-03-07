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
    
    filtered_df = filter_option_data_03.run_filter()

    check_results = pd.DataFrame(index=pd.MultiIndex.from_product([['Level 3 filters'], ['IV filter', 'Put-call parity filter', 'All']]),
                             columns=pd.MultiIndex.from_product([['Berkeley', 'OptionMetrics'], ['Deleted', 'Remaining']]))
    check_results.loc[['Level 3 filters'], ['Berkeley', 'OptionMetrics']] = [
        [10865, np.nan, 67850, np.nan],
        [10298, np.nan,46138, np.nan],
        [np.nan, 173500,np.nan, 962784]]
    
    '''
    Expected output:
    
                                            Berkeley           OptionMetrics          
                                            Deleted Remaining       Deleted Remaining
    Level 3 filters IV filter                 10865       NaN         67850       NaN
                    Put-call parity filter    10298       NaN         46138       NaN
                    All                         NaN    173500           NaN    962784
    ''' 
    
    assert_frame_equal(filtered_df, check_results, atol=1e-2)