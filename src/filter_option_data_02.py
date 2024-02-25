import pandas as pd
import numpy as np
import bsm_pricer as bsm
import load_option_data_01 as l1
import filter_option_data_01 as f1
from pathlib import Path
import config

DATA_DIR = Path(config.DATA_DIR)

def calc_time_to_maturity(df):
    """Calculate time to maturity in days.
    """
    ttm = (df['exdate'] - df['date']).dt.days
    return ttm

def calc_time_to_maturity_yrs(df):
    """Calculate time to maturity in years.
    """
    ttm_yrs = calc_time_to_maturity(df) / 365
    return ttm_yrs

def filter_time_to_maturity(df):
    """Days to Maturity <7 or >180 Filter: Filter options 
       based on time to maturity.
    """
    # calculate time to maturity >> df.time_to_maturity
    df['time_to_maturity'] = calc_time_to_maturity(df)

    # calculate time to maturity in years >> df.time_to_matility_yrs
    df['time_to_matility_yrs'] = calc_time_to_maturity_yrs(df)  

    # remove options with less than 7 days to maturity or greater than 180 days to maturity
    df = df.loc[(df['time_to_maturity'] >= 7) & (df['time_to_maturity'] <= 180)].reset_index(drop=True)

    return df

def filter_iv(df):
    """IV<5% or >100% Filter: Filter options based on implied volatility.
    """
    df = df.loc[((df['impl_volatility']>=0.05) 
            & (df['impl_volatility']<=1.00))
           | (df['impl_volatility'].isna())] 
    
    return df

def filter_moneyness(df):
    """Moneyness <0.8 or >1.2 Filter: Filter options based on moneyness.
       We define moneyness as the ratio of the option's strike price to 
       the stock underlying price. Moneyness field must be defined before 
       running this function
    """
    # check if moneyness (mnyns) is already in the dataframe
    if 'mnyns' not in df.columns:
        # if not, calculate moneyness
        df = f1.getSecPrice(df)
    # remove options with moneyness less than 0.8 or greater than 1.2
    df = df.loc[(df['mnyns']>=0.8) & (df['mnyns']<=1.2)].reset_index(drop=True)
    return df

def filter_implied_interest_rate(df):
    """Implied Interest Rate <0% Filter: Filter options based on implied interest rate.
       Implied interest rate field must be defined before running this function.
    """
    return df

def filter_unable_compute_iv(df):
    """Unable to Compute IV Filter: Filter options where implied volatility cannot be computed.
    """
    return df

def apply_l2_filters(df):
    """Apply all level 2 filters to the dataframe.
    """
    df = filter_time_to_maturity(df)
    df = filter_iv(df)
    df = filter_moneyness(df)
    df = filter_implied_interest_rate(df)
    df = filter_unable_compute_iv(df)
    return df

if __name__ == "__main__": 
    """Save the filtered data to a parquet file.
       Must first load all options data, apply level 1 filters, and then apply level 2 filters.
    """
    df = l1.load_all_optm_data()
    df_f1, df_sum, df_tableB1 = f1.appendixBfilter_level1(df)
    df_f2 = apply_l2_filters(df_f1)

    startYearMonth = df_f2['date'].min().year*100 + df_f2['date'].min().month
    endYearMonth = df_f2['date'].max().year*100 + df_f2['date'].max().month

    save_path = DATA_DIR.joinpath( f"pulled/data_{startYearMonth}_{endYearMonth}_L2filter.parquet")
    df_f2.to_parquet(save_path)