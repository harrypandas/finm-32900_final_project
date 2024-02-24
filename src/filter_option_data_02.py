import pandas as pd
import numpy as np
import bsm_pricer as bsm
import load_option_data_01 as l1
import filter_option_data_01 as f1
from pathlib import Path
import config

DATA_DIR = Path(config.DATA_DIR)

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
        df = f1.getSecPrice(df)
    # remove options with moneyness less than 0.8 or greater than 1.2
    df = df.loc[(df['mnyns']>=0.8) & (df['mnyns']<=1.2)].reset_index(drop=True)
    return df

def filter_implied_interest_rate(df):
    return df

def apply_l2_filters(df):
    df = filter_time_to_maturity(df)
    df = filter_iv(df)
    df = filter_moneyness(df)
    df = filter_implied_interest_rate(df)
    return df

if __name__ == "__main__": 
    df = l1.load_all_optm_data()
    df_f1, df_sum, df_tableB1 = f1.appendixBfilter_level1(df)
    df_f2 = apply_l2_filters(df_f1)

    startYearMonth = df_f2['date'].min().year*100 + df_f2['date'].min().month
    endYearMonth = df_f2['date'].max().year*100 + df_f2['date'].max().month

    save_path = DATA_DIR.joinpath( f"pulled/data_{startYearMonth}_{endYearMonth}_L2filter.parquet")
    df_f2.to_parquet(save_path)