import filter_option_data_02 as f2
import matplotlib.pyplot as plt
import pandas as pd
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01
START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02

def build_l2_days_to_mat_plot(optm_l1_df, date_range):
    """Build plot for days to maturity filter
       Save plot to file L2_date_fig1_L2filter.png
    """
    # calculate time to maturity in years for level 1 data
    optm_l1_df['time_to_maturity_yrs'] = f2.calc_time_to_maturity_yrs(optm_l1_df)

    # # create data frame with initial level 2 filter for time to maturity applied
    # optm_l2_df = f2.filter_time_to_maturity(optm_l1_df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plotting histogram for optm_l1_df
    axes[0].hist(optm_l1_df['time_to_maturity_yrs'], bins=10, edgecolor='black')
    axes[0].set_xlabel('Time to Maturity (Years)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram of Time to Maturity (Pre-Filter)')

    # Plotting histogram for optm_l2_df
    axes[1].hist(optm_l2_df['time_to_maturity_yrs'], bins=10, edgecolor='black', color='darkred')
    axes[1].set_xlabel('Time to Maturity (Years)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram of Time to Maturity (Post-Filter)')

    # Adjusting the spacing between subplots
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'L2_{date_range}_fig1.png')


def build_l2_iv_tmm_plot(optm_l1_df, optm_l2_df, date_range):
    plt.clf()
    plt.scatter(optm_l1_df['time_to_maturity_yrs'], optm_l1_df['impl_volatility'], label='Pre-Filter')
    plt.scatter(optm_l2_df['time_to_maturity_yrs'], optm_l2_df['impl_volatility'], label='Post-Filter', color='darkred')
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility vs Time to Maturity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'L2_{date_range}_fig2.png')

def build_l2_iv_dist_plot(optm_l2_df, optm_l2_iv, date_range):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plotting histogram for optm_l2_df
    axes[0].hist(optm_l2_df['impl_volatility'], bins=10, edgecolor='black')
    axes[0].set_xlabel('Implied Volatility')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram of Implied Volatility (Pre-Filter)')

    # Plotting histogram for optm_l1_iv
    axes[1].hist(optm_l2_iv['impl_volatility'], bins=10, edgecolor='black', color='darkred')
    axes[1].set_xlabel('Implied Volatility')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram of Implied Volatility (Post-Filter)')

    # Adjusting the spacing between subplots
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'L2_{date_range}_fig3.png')

def build_l2_mny_plot(optm_l2_iv, optm_l2_mny, date_range):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plotting optm_l2_iv
    axes[0].scatter(optm_l2_iv['mnyns'], optm_l2_iv['volume'])
    axes[0].set_xlabel('Moneyness')
    axes[0].set_ylabel('Volume')
    axes[0].set_title('Moneyness vs Volume (Pre-Filter)')

    # Plotting optm_l2_mny
    axes[1].scatter(optm_l2_mny['mnyns'], optm_l2_mny['volume'], color='darkred')
    axes[1].set_xlabel('Moneyness')
    axes[1].set_ylabel('Volume')
    axes[1].set_title('Moneyness vs Volume (Post-Filter)')

    # Add dotted line representing the range 0.8 to 1.2 on x-axis
    axes[0].axvline(0.8, color='black', linestyle='dotted')
    axes[0].axvline(1.2, color='black', linestyle='dotted')
    axes[1].axvline(0.8, color='black', linestyle='dotted')
    axes[1].axvline(1.2, color='black', linestyle='dotted')

    # Adjusting the spacing between subplots
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'L2_{date_range}_fig4.png')

def build_l2_nocomp_iv_plot(optm_l2_int, optm_l2_univ, date_range):
    nan_percentage = optm_l2_int.loc[optm_l2_int['impl_volatility'].isna()].groupby(['time_to_maturity']).size()/optm_l2_int.groupby(['time_to_maturity']).size()*100
    
    plt.clf()
    plt.scatter(nan_percentage.index, nan_percentage, alpha=0.5, s=10, label='Pre-Filter')
    plt.scatter(optm_l2_univ['time_to_maturity'], optm_l2_univ['impl_volatility'], color='darkred', alpha=0.1, s=10, label='Post-Filter')
    plt.xlabel('Time to Maturity')
    plt.ylabel('Percentage of NaN by Implied Volatility')
    plt.title('Percentage of NaN Implied Volatility by Time to Maturity')
    plt.legend()

    plt.savefig(OUTPUT_DIR / f'L2_{date_range}_fig5.png')


if __name__ == "__main__":
    date_ranges = [f'{START_DATE_01[:7]}_{END_DATE_01[:7]}',
                    f'{START_DATE_02[:7]}_{END_DATE_02[:7]}']
    
    print("Creating Level 2 plots...")
    for date_range in date_ranges:
        # load data with level 1 filters applied
        optm_l1_df = pd.read_parquet(DATA_DIR / "intermediate" / f"data_{date_range}_L1filter.parquet")
        
        optm_l2_df = f2.filter_time_to_maturity(optm_l1_df)

        print("Building Level 2 - Figure 1...")
        # build plot for days to maturity filter
        build_l2_days_to_mat_plot(optm_l1_df, date_range)

        print("Building Level 2 - Figure 2...")
        # build plot for implied volatility vs time to maturity
        build_l2_iv_tmm_plot(optm_l1_df, optm_l2_df, date_range)

        optm_l2_iv = f2.filter_iv(optm_l2_df)

        print("Building Level 2 - Figure 3...")
        # build plot for implied volatility distribution
        build_l2_iv_dist_plot(optm_l2_df, optm_l2_iv, date_range)

        optm_l2_mny = f2.filter_moneyness(optm_l2_iv)

        print("Building Level 2 - Figure 4...")
        # build plot for moneyness vs volume
        build_l2_mny_plot(optm_l2_iv, optm_l2_mny, date_range)

        optm_l2_int = f2.filter_implied_interest_rate(optm_l2_mny)
        optm_l2_univ = f2.filter_unable_compute_iv(optm_l2_int)

        print("Building Level 2 - Figure 5...")
        # build plot for implied volatility unable to compute
        build_l2_nocomp_iv_plot(optm_l2_int, optm_l2_univ, date_range)
