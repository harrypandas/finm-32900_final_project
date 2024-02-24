import pandas as pd
import numpy as np
import bsm_pricer as bsm
import filter_option_data_01 as fod1

def calc_time_to_maturity(df):
    ttm = (df['exdate'] - df['date']).dt.days
    return ttm

def calc_time_to_maturity_yrs(df):
    ttm_yrs = calc_time_to_maturity(df) / 365
    return ttm_yrs

def filter_time_to_maturity(df):
    # calculate time to maturity >> df.time_to_maturity
    df['time_to_maturity'] = calc_time_to_maturity(df)

    # calculate time to maturity in years >> df.time_to_matility_yrs
    df['time_to_matility_yrs'] = calc_time_to_maturity_yrs(df)  

    # remove options with less than 7 days to maturity or greater than 180 days to maturity
    df = df.loc[(df['time_to_maturity'] >= 7) & (df['time_to_maturity'] <= 180)].reset_index(drop=True)

    return df

def filter_iv(df):
    
    return df

def filter_moneyness(df):
    # check if moneyness (mnyns) is already in the dataframe
    if 'mnyns' not in df.columns:
        # if not, calculate moneyness
        df = fod1.getSecPrice(df)
    # remove options with moneyness less than 0.8 or greater than 1.2
    df = df.loc[(df['mnyns']>=0.8) & (df['mnyns']<=1.2)].reset_index(drop=True)
    return df

def filter_implied_interest_rate(df):
    return df


if __name__ == "__main__": 
    pass