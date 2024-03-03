# standard libraries
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# system libraries
import os
import sys
from pathlib import Path
# Add the src directory to the path in order to import config
current_directory = Path.cwd()
src_path = current_directory.parent / "src"
sys.path.insert(0, str(src_path))

# project files
import config
import load_option_data_01 as l1
import filter_option_data_01 as f1
import wrds
import bsm_pricer as bsm

# environment variables
WRDS_USERNAME = Path(config.WRDS_USERNAME)
DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)


# Helper functions
def fit_and_store_curve(group):
    """
    Fit a quadratic curve to the given group of data points and store the fitted values.

    Args:
        group (pandas.DataFrame): The group of data points to fit the curve to.

    Returns:
        pandas.DataFrame: The group of data points with the fitted values stored in the 'fitted_iv' column.
    """
    # Drop rows with NaN in 'moneyness' or 'log_iv'
    group = group.dropna(subset=['mnyns', 'log_iv'])
    if len(group) < 3:  # Need at least 3 points to fit a quadratic curve
        return group
    try:
        # Fit the quadratic curve
        coefficients = np.polyfit(group['mnyns'], group['log_iv'], 2)
        # Calculate fitted values
        group['fitted_iv'] = np.polyval(coefficients, group['mnyns'])
    except np.RankWarning:
        print("Polyfit may be poorly conditioned")
    return group


def calc_relative_distance(series1, series2, method='percent'):
    """
    Calculate the relative distance between the implied volatility and the fitted implied volatility.
    
    Parameters:
        method (str): The method to calculate the relative distance. Options are 'percent', 'manhattan', or 'euclidean'.
        
    Returns:
        numpy.ndarray: The relative distance calculated based on the specified method.
        
    Raises:
        ValueError: If the method is not one of 'percent', 'manhattan', or 'euclidean'.
    """
    
    if method == 'percent':
        result = (series1 - series2) / series2 * 100
    elif method == 'manhattan':
        result = abs(series1 - series2)
    elif method == 'euclidean':
        result = np.sqrt((series1 - series2)**2)
    else:
        raise ValueError("Method must be 'percent', 'manhattan', or 'euclidean'")
    
    result = np.where(np.isinf(result), np.nan, result)
    
    return result

    
def mark_outliers(row, std_devs, outlier_threshold):
    """
    Determines if a data point is an outlier based on its moneyness_bin and relative distance from the fitted curve.
    
    Args:
        row (pandas.Series): A row of data containing the moneyness_bin and rel_distance columns.
        std_devs (pandas.DataFrame): A DataFrame containing the standard deviations for each moneyness_bin.
    
    Returns:
        bool: True if the data point is an outlier, False otherwise.
    """
    
    # Attempt to retrieve the standard deviation for the row's moneyness_bin
    std_dev_row = std_devs.loc[std_devs['mnyns_bin'] == row['mnyns_bin'], 'std_dev']
    
    # Check if std_dev_row is empty (i.e., no matching moneyness_bin was found)
    if not std_dev_row.empty:
        std_dev = std_dev_row.values[0]
        # Calculate how many std_devs away from the fitted curve the IV is
        if abs(row['rel_distance']) > outlier_threshold * std_dev:  # Adjust this threshold as needed
            return True
    else:
        # Handle the case where no matching moneyness_bin was found
        return False
    return False


def build_put_call_pairs(call_options, put_options):
    """
    Builds pairs of call and put options based on the same date, expiration date, and moneyness.

    Args:
        call_options (DataFrame): DataFrame containing call options data.
        put_options (DataFrame): DataFrame containing put options data.

    Returns:
        tuple of (matching_calls: pd.DataFrame, matching_puts: pd.DataFrame)
    """
    call_options.set_index(['date', 'exdate', 'mnyns'], inplace=True)
    put_options.set_index(['date', 'exdate', 'mnyns'], inplace=True)
    
    # get common indices
    common_index = call_options.index.intersection(put_options.index)

    # Extract the matching entries
    matching_calls = call_options.loc[common_index]
    matching_puts = put_options.loc[common_index]
    
    result = (matching_calls, matching_puts)

    return result


def test_price_strike_match(matching_calls_puts):
    """
    Check if the strike prices and security prices of matching calls and puts are equal.

    Parameters:
    matching_calls_puts (DataFrame): DataFrame containing matching calls and puts data.

    Returns:
    bool: True if the strike prices and security prices of matching calls and puts are equal, False otherwise.
    """
    return (np.allclose(matching_calls_puts['strike_price_C'], matching_calls_puts['strike_price_P'])) and (np.allclose(matching_calls_puts['sec_price_C'], matching_calls_puts['sec_price_P']))# and (np.allclose(matching_calls_puts['tb_m3_C'], matching_calls_puts['tb_m3_P']))


def calc_implied_interest_rate(matched_options):
    """
    Calculates the implied interest rate based on the given matched options data.

    Parameters:
    matched_options (DataFrame): DataFrame containing the matched options data.

    Returns:
    DataFrame: DataFrame with an additional column 'pc_parity_int_rate' representing the implied interest rate.
    
    Raises:
    ValueError: If there is a mismatch between the price and strike price of the options.
    """
    
    # underlying price
    if test_price_strike_match(matched_options):
        print(">> Underlying prices, strike prices of put and call options match exactly.")
        S = matched_options['sec_price_C']
        K = matched_options['strike_price_C']  
        
        # 1/T = 1/time to expiration in years
        T_inv = np.power((matched_options.reset_index()['exdate']-matched_options.reset_index()['date'])/datetime.timedelta(days=365), -1)
        T_inv.index=matched_options.index
        T_inv
        
        C_mid = matched_options['mid_price_C']
        P_mid = matched_options['mid_price_P']
        # implied interest rate
        matched_options['pc_parity_int_rate'] = np.log((S-C_mid+P_mid)/K) * T_inv
        return matched_options
    else:
        raise ValueError("Price and strike price mismatch")


def pcp_filter_sensitivity(matched_options, int_rate_rel_distance_func, outlier_threshold):
    """
    Run a sensitivity analysis on the put-call parity filter, using different methods for relative distance and different standard deviation outlier thresholds.cx

    Parameters:
    matched_options (DataFrame): DataFrame containing the matched options data.
    int_rate_rel_distance_func (str): Method to calculate relative distance for interest rate.
    outlier_threshold (float): Threshold for flagging outliers.

    Returns:
    dict: A dictionary containing the number of deleted and remaining options after filtering.
    """
    matched_options['rel_distance_int_rate'] = calc_relative_distance(matched_options['pc_parity_int_rate'], matched_options['daily_median_rate'], method=int_rate_rel_distance_func)
    # fill 3905 nans...
    matched_options['rel_distance_int_rate'] = matched_options['rel_distance_int_rate'].fillna(0.0)

    # calculate the standard deviation of the relative distances
    stdev_int_rate_rel_distance = matched_options['rel_distance_int_rate'].std()

    # flag outliers based on the threshold
    matched_options['is_outlier_int_rate'] = matched_options['rel_distance_int_rate'].abs() > outlier_threshold * stdev_int_rate_rel_distance

    # filter out the outliers
    l3_filtered_options = matched_options[~matched_options['is_outlier_int_rate']]

    # make the dataframe long-form to compare to the level 2 data
    _calls = l3_filtered_options.filter(like='_C').rename(columns=lambda x: x.replace('_C', ''))
    _puts = l3_filtered_options.filter(like='_P').rename(columns=lambda x: x.replace('_P', ''))
    l3_filtered_options = pd.concat((_calls, _puts), axis=0)

    # update the final results for this combination of relative distance method and outlier threshold
    return l3_filtered_options


def iv_filter_sensitivity(l2_data, iv_distance_method, iv_outlier_threshold):
    """
    Filter out outliers based on the sensitivity of the implied volatility (IV) data.

    Args:
        l2_data (DataFrame): DataFrame containing the IV data.
        iv_distance_method (str): Method used to calculate the relative distance between observed and fitted IV.
        iv_outlier_threshold (float): Threshold for flagging outliers based on the standard deviation of relative distances.

    Returns:
        DataFrame: Filtered DataFrame with outliers removed.

    """
    l2_data['rel_distance_iv'] = calc_relative_distance(l2_data['log_iv'], l2_data['fitted_iv'], method=iv_distance_method)

    # Define moneyness bins
    bins = np.arange(0.8, 1.21, 0.05)
    l2_data['mnyns_bin'] = pd.cut(l2_data['mnyns'], bins=bins)

    # Compute standard deviation of relative distances within each moneyness bin
    std_devs = l2_data.groupby('mnyns_bin')['rel_distance_iv'].std().reset_index(name='std_dev')
    std_devs
    l2_data
    l2_data['stdev_iv_mnyns_bin'] = l2_data['mnyns_bin'].map(std_devs.set_index('mnyns_bin')['std_dev'])
    l2_data['stdev_iv_mnyns_bin'].apply(lambda x: x*iv_outlier_threshold).astype(float)
    # flag outliers based on the threshold
    l2_data['is_outlier_iv'] = l2_data['rel_distance_iv'].abs() > l2_data['stdev_iv_mnyns_bin'].apply(lambda x: x*iv_outlier_threshold).astype(float)

    # filter out the outliers
    l3_data_iv_only = l2_data[~l2_data['is_outlier_iv']]
    
    # update the final results for this combination of relative distance method and outlier threshold
    return l3_data_iv_only


# charts and checks
def build_check_results():
    check_results = pd.DataFrame(index=pd.MultiIndex.from_product([['Level 3 filters'], ['IV filter', 'Put-call parity filter', 'All']]),
                             columns=pd.MultiIndex.from_product([['Berkeley', 'OptionMetrics'], ['Deleted', 'Remaining']]))
    check_results.loc[['Level 3 filters'], ['Berkeley', 'OptionMetrics']] = [[10865, np.nan, 67850, np.nan], [10298, np.nan,46138, np.nan], [np.nan, 173500,np.nan, 962784]]

    return check_results.loc[:, 'OptionMetrics']

def build_l2_data_chart(l2_data, date_range):
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    ax[0,0].hist(l2_data['impl_volatility'], bins=250, color='darkblue')
    ax[0,0].set_xlabel('IV')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].set_title('Distribution of IV')
    ax[0,0].grid()

    ax[0,1].hist(l2_data['log_iv'], bins=250, color='grey')
    ax[0,1].set_xlabel('log(IV)')
    ax[0,1].set_ylabel('Frequency')
    ax[0,1].set_title('Distribution of log(IV)')
    ax[0,1].grid()

    # options with nan implied volatility
    # calls only
    nan_iv = l2_data[(l2_data['cp_flag'] == 'C') & (l2_data['impl_volatility'].isna())]
    ax[1,0].scatter(x=nan_iv['date'], y=nan_iv['mnyns'], color='blue', alpha=0.1, s=10, label='Calls')

    # puts only
    nan_iv = l2_data[(l2_data['cp_flag'] == 'P') & (l2_data['impl_volatility'].isna())]
    ax[1,0].scatter(x=nan_iv['date'], y=nan_iv['mnyns'], color='red', alpha=0.1, s=10, label='Puts')

    ax[1,0].set_xlabel('Trade Date')
    ax[1,0].set_ylabel('Moneyness')
    ax[1,0].set_title('Moneyness of Calls with NaN IV')
    ax[1,0].grid()
    ax[1,0].legend()
    ax[1,0].grid()


    # percentage of NaN IV
    nan_percentage = l2_data.groupby(['date', 'cp_flag'])['impl_volatility'].apply(lambda x: (x.isna().sum() / len(x))*100)

    # calls only
    nan_percentage_calls = nan_percentage[nan_percentage.index.get_level_values(1)=='C']
    ax[1,1].scatter(x=nan_percentage_calls.index.get_level_values(0), y=nan_percentage_calls.values, color='blue', alpha = 0.1, s=10, label='Calls')

    # puts only
    nan_percentage_puts = nan_percentage[nan_percentage.index.get_level_values(1)=='P']
    ax[1,1].scatter(x=nan_percentage_puts.index.get_level_values(0), y=nan_percentage_puts.values, color='red', alpha = 0.1, s=10, label='Puts')

    ax[1,1].set_xlabel('Trade Date')
    ax[1,1].set_ylabel('Percentage of NaN IV')
    ax[1,1].set_title('Percentage of NaN IV by Trade Date')
    ax[1,1].legend()
    ax[1,1].grid()

    # Hide ax[1,2]
    #ax[1,2].axis('off')

    plt.suptitle(f'Level 2 Filtered Data: {date_range.replace("_", " to ")}')
    plt.tight_layout()
    plt.show()
    
    fig.savefig(OUTPUT_DIR / f'L3_fig1_post_L2filter_{date_range}.svg')


def nan_iv_in_l2_data(l2_data, date_range):
    nan_iv_calls = l2_data[(l2_data['cp_flag'] == 'C') & (l2_data['impl_volatility'].isna())]
    nan_iv_puts = l2_data[(l2_data['cp_flag'] == 'P') & (l2_data['impl_volatility'].isna())]
    nan_iv_summary = pd.DataFrame(index=['Calls', 'Puts'], columns = ['NaN IV Records', 'Total Records', '% NaN IV'])
    nan_iv_summary.loc['Calls'] = [len(nan_iv_calls), len(l2_data[l2_data['cp_flag'] == 'C']), len(nan_iv_calls)/len(l2_data[l2_data['cp_flag'] == 'C'])*100]
    nan_iv_summary.loc['Puts'] = [len(nan_iv_puts), len(l2_data[l2_data['cp_flag'] == 'P']), len(nan_iv_puts)/len(l2_data[l2_data['cp_flag'] == 'P'])*100]
    nan_iv_summary.style.format({'NaN IV Records': '{:,.0f}', 'Total Records': '{:,.0f}',
                                         '% NaN IV': '{:.2f}%'}).set_caption(f'Summary of NaN IV Records in Level 2 Filtered Data: {date_range.replace("_", " to ")}')
    
    nan_iv_summary.to_latex(OUTPUT_DIR / f'nan_iv_summary_{date_range}.tex')
    return nan_iv_summary


def apply_quadratic_iv_fit(l2_data):
    # Apply the quadratic curve fitting function to the data
    l2_data = l2_data.groupby(['date', 'exdate', 'cp_flag']).apply(fit_and_store_curve)
    return l2_data


def build_l2_fitted_iv_chart(l2_data, date_range):
    fig, ax = plt.subplots(2,3, figsize=(12,8))

    ax[0,0].hist(l2_data['impl_volatility'], bins=250, color='darkblue')
    ax[0,0].set_xlabel('IV')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].set_title('Distribution of IV')
    ax[0,0].grid()

    ax[0,1].hist(l2_data['log_iv'], bins=250, color='grey')
    ax[0,1].set_xlabel('log(IV)')
    ax[0,1].set_ylabel('Frequency')
    ax[0,1].set_title('Distribution of log(IV)')
    ax[0,1].grid()

    # Scatter plot of IV vs fitted IV
    ax[0,2].scatter(x=l2_data['log_iv'], y=l2_data['fitted_iv'], color='darkblue', alpha=0.1)
    ax[0,2].set_xlabel('log(IV)')
    ax[0,2].set_ylabel('Fitted log(IV)')
    ax[0,2].set_title('Fitted log(IV) vs  Observed log(IV)')
    # Add 45-deg line
    ax[0,2].plot([min(l2_data['log_iv']), max(l2_data['log_iv'])], [min(l2_data['log_iv']), max(l2_data['log_iv'])], color='red', linestyle='--')
    ax[0,2].grid()


    ax[1,0].scatter(x=l2_data.xs('C', level='cp_flag')['mnyns'], y=np.exp(l2_data.xs('C', level='cp_flag')['log_iv']), color='blue', alpha=0.1, label='IV')
    ax[1,0].set_xlabel('Moneyness')
    ax[1,0].set_ylabel('IV')
    ax[1,0].set_title('IV vs Moneyness (Calls)')

    ax[1,1].scatter(x=l2_data.xs('P', level='cp_flag')['mnyns'], y=np.exp(l2_data.xs('P', level='cp_flag')['log_iv']), color='red', alpha=0.1, label='IV')
    ax[1,1].set_xlabel('Moneyness')
    ax[1,1].set_ylabel('IV')
    ax[1,1].set_title('IV vs Moneyness (Puts)')

    # Hide ax[1,2]
    ax[1,2].axis('off')

    plt.suptitle(f'Level 2 Filtered Data with Fitted IVs: {date_range.replace("_", " to ")}')
    plt.tight_layout()
    plt.show()
    
    fig.savefig(OUTPUT_DIR / f'L3_fig2_L2fitted_iv_{date_range}.svg')
    
    


def run_filter(date_range='1996-01_2012-01'):
    """
    Runs the level 3 filter.

    Returns:
        result (pd.DataFrame): The final output of the level 3 filter corresponding to Table B1. 
    """
    
    l2_input_file = f"intermediate/data_{date_range}_L2filter.parquet"
    l3_iv_only_output_file = f"intermediate/data_{date_range}_L3filterIVonly.parquet"
    l3_output_file = f"intermediate/data_{date_range}_L3filter.parquet"
    
    
    # read in L2 filtered data
    l2_data = pd.read_parquet(DATA_DIR / l2_input_file, columns=['secid', 'date', 'exdate', 'cp_flag', 'mnyns', 'impl_volatility', 'tb_m3', 'best_bid', 'best_offer', 'strike_price', 'contract_size', 'sec_price'])
    # calc log IV 
    l2_data['log_iv'] = np.log(l2_data['impl_volatility'])
    
    build_l2_data_chart(l2_data, date_range)
    nan_iv_summary = nan_iv_in_l2_data(l2_data, date_range)
    l2_data = apply_quadratic_iv_fit(l2_data)
    build_l2_fitted_iv_chart(l2_data, date_range)
    
    # TODO: START HERE!!!
    
    
    
    
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