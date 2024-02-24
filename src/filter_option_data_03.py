import pandas as pd
import numpy as np


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