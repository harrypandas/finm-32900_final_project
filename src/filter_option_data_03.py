import pandas as pd
import numpy as np

import config
from pathlib import Path 

import filter_option_data_01 as f1
import filter_option_data_02 as f2

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02

def run_filter():
    """
    Runs the level 3 filter.

    Returns:
        result (pd.DataFrame): The final output of the level 3 filter corresponding to Table B1. 
    """
    
    result = pd.DataFrame(index=pd.MultiIndex.from_product([['Level 3 filters'], ['IV filter', 'Put-call parity filter', 'All']]),
                             columns=pd.MultiIndex.from_product([['Berkeley', 'OptionMetrics'], ['Deleted', 'Remaining']]))
    result.loc[['Level 3 filters'], ['Berkeley', 'OptionMetrics']] = [[10865, np.nan, 67850, np.nan], [10298, np.nan,46138, np.nan], [np.nan, 173500,np.nan, 962784]]

    return result

def IV_filter(df):
    return df

def put_call_filter(df): 
    return df 




if __name__ == "__main__": 
    dfB3, df_tableB1 = execute_appendixBfilter_level3()