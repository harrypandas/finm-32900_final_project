import pandas as pd
import numpy as np
from datetime import datetime,timedelta
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
    df['time_to_maturity_yrs'] = calc_time_to_maturity_yrs(df)  

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
        df = f1.calc_moneyness(df)
    # remove options with moneyness less than 0.8 or greater than 1.2
    df = df.loc[(df['mnyns']>=0.8) & (df['mnyns']<=1.2)].reset_index(drop=True)
    return df

def filter_implied_interest_rate(optm_l2_mny):
    """Implied Interest Rate <0% Filter: Filter options based on implied interest rate.
       Implied interest rate field must be defined before running this function.
    """
    def build_put_call_pairs(call_options, put_options):
        """
        Builds pairs of call and put options based on the same date, expiration date, and moneyness.

        Args:
            call_options (DataFrame): DataFrame containing call options data.
            put_options (DataFrame): DataFrame containing put options data.

        Returns:
            tuple of (matching_calls: pd.DataFrame, matching_puts: pd.DataFrame)
        """
        call_options.set_index(['date', 'exdate', 'strike_price', 'sec_price'], inplace=True)
        put_options.set_index(['date', 'exdate', 'strike_price', 'sec_price'], inplace=True)
        
        # get common indices
        common_index = call_options.index.intersection(put_options.index)

        # Extract the matching entries
        matching_calls = call_options.loc[common_index]
        matching_puts = put_options.loc[common_index]
        
        result = (matching_calls, matching_puts)

        return result

    def calc_implied_interest_rate(matched_options):
        # underlying price
        # if test_price_strike_match(matched_options):
        print(">> Prices and strikes of puts and calls match correctly.")
        S = matched_options['sec_price']
        K = matched_options['strike_price']
        
        # 1/T = 1/time to expiration in years
        T_inv = np.power((matched_options['exdate']-matched_options['date'])/timedelta(days=365),-1)
        T_inv.index=matched_options.index
        
        C_mid = matched_options['best_mid_C']
        P_mid = matched_options['best_mid_P']
        # implied interest rate
        matched_options['impl_int_rate'] = -np.log(-(C_mid-P_mid-S)/K) * T_inv
        return matched_options
    
    optm_l2_mny['best_mid'] = (optm_l2_mny['best_bid'] + optm_l2_mny['best_offer']) / 2

    call_options = optm_l2_mny.loc[optm_l2_mny['cp_flag'] == 'C'].reset_index(drop=True)
    put_options = optm_l2_mny.loc[optm_l2_mny['cp_flag'] == 'P'].reset_index(drop=True)

    matching_calls, matching_puts = build_put_call_pairs(call_options, put_options)

    matched_options = pd.merge(matching_calls, matching_puts, on=['date', 'exdate', 'strike_price', 'sec_price'], suffixes=('_C', '_P')).reset_index()

    matched_options = calc_implied_interest_rate(matched_options)
    neg_int = matched_options.loc[matched_options['impl_int_rate'] < 0][['date', 'exdate', 'strike_price', 'sec_price']].drop_duplicates()
    optm_l2_int = pd.merge(optm_l2_mny, neg_int, on=['date', 'exdate', 'strike_price', 'sec_price'], how='outer', indicator=True)
    optm_l2_int = optm_l2_int.loc[optm_l2_int['_merge'] == 'left_only'].drop(columns='_merge')  
    med_impl_int = matched_options.loc[(matched_options['mnyns_C']>=0.95) & (matched_options['mnyns_C']<=1.05)].groupby(['time_to_maturity_C'])['impl_int_rate'].median().reset_index()
    med_impl_int = med_impl_int.loc[med_impl_int['impl_int_rate']>=0]
    optm_l2_int = pd.merge(optm_l2_int, med_impl_int, left_on='time_to_maturity', right_on='time_to_maturity_C', how='left', indicator=True)
    optm_l2_int['impl_int_rate'] = optm_l2_int['impl_int_rate'].ffill()
    return optm_l2_int

def filter_unable_compute_iv(df):
    """Unable to Compute IV Filter: Filter options where implied volatility cannot be computed.
    """
    df = df.loc[df['impl_volatility'].notna()]
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
    # df = l1.load_all_optm_data()
    # df_f1, df_sum, df_tableB1 = f1.appendixBfilter_level1(df)
    # df_f2 = apply_l2_filters(df_f1)

    # startYearMonth = df_f2['date'].min().year*100 + df_f2['date'].min().month
    # endYearMonth = df_f2['date'].max().year*100 + df_f2['date'].max().month

    # save_path = DATA_DIR.joinpath( f"pulled/data_{startYearMonth}_{endYearMonth}_L2filter.parquet")
    # df_f2.to_parquet(save_path)