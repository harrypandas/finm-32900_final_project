import pandas as pd
import numpy as np

import config
from pathlib import Path 
import time 

import load_option_data_01 as l1
import filter_option_data_01 as f1
import filter_option_data_02 as f2
import filter_option_data_03 as f3

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02




def getLengths(df): 
	test1 = df['cp_flag'].value_counts().to_dict()
	test1C = test1['C']
	test1P = test1['P'] 
	test1L = len(df)
	return np.array([test1L, test1C, test1P])




def getB1info(func, df, TB, level, name, Lprev = 0):
	df2 = func(df)
	L1 = len(df2)
	TB[name] = [level, (Lprev-L1),float('nan')]
	return df2, TB, L1


def executeLevel(lvl, steps, filters, df, dfTB, save_path): 
	L = len(df)
	for step, filt in zip(steps, filters):
		df, dfTB, L = getB1info(filt, df, dfTB, lvl, step, L)
	dfTB['All' + lvl[6]] = [lvl, float('nan'), L]
	df.to_parquet(save_path)
	return df, dfTB

def appendixBfilter(start=START_DATE_01, end=END_DATE_01): 
	df = l1.load_all_optm_data(data_dir=DATA_DIR,
											wrds_username=WRDS_USERNAME, 
											startDate=start,
											endDate=end)
	df = f1.getSecPrice(df)
	df = f1.calc_moneyness(df)


	L0 = getLengths(df)
	rows_B = ['Step', 'Deleted', 'Remaining']
	dfTB = pd.DataFrame(index = rows_B)
	dfTB['Calls0'] = ['Starting', float('nan'), L0[1]]
	dfTB['Puts0'] = ['Starting', float('nan'), L0[2]]
	dfTB['All0'] = ['Starting', float('nan'), L0[0]]


	lvlOne = 'Level 1 filters'
	save_path_L1 = DATA_DIR.joinpath( f"intermediate/data_{start[:7]}_{end[:7]}_L1filter.parquet")
	lv1OneSteps = ['Identical', 'Identical except price', 
		'Bid = 0', 'Volume = 0'] 
	lvlOneFilters = [f1.delete_identical_filter, f1.delete_identical_but_price_filter, 
		f1.delete_zero_bid_filter, f1.delete_zero_volume_filter]

	lvlTwo = 'Level 2 filters'
	save_path_L2 = DATA_DIR.joinpath( f"intermediate/data_{start[:7]}_{end[:7]}_L2filter.parquet")
	lv1TwoSteps = ['Days to expiration <7 or >180', 'IV <5\\% or >100\\%', 
		'K/S <0.8 or >1.2', 'Implied interest rate < 0',
		'Unable to compute IV'] 
	lvlTwoFilters = [f2.filter_time_to_maturity, f2.filter_iv, 
		f2.filter_moneyness, f2.filter_implied_interest_rate, 
		f2.filter_unable_compute_iv]


	lvlThree = 'Level 3 filters'
	save_path_L3 = DATA_DIR.joinpath( f"intermediate/data_{start[:7]}_{end[:7]}_L3filter.parquet")
	lvlThreeSteps = ['IV filter', 'Put-call parity filter']
	lvlThreeFilters =[f3.IV_filter, f3.put_call_filter]
	



	time0 = time.time() 
	df, dfTB = executeLevel(lvlOne, lv1OneSteps, lvlOneFilters, df, dfTB, save_path_L1)
	print(f"\nData {start}-{end}, Filtered up to Level B1 saved to: {save_path_L1}")
	df, dfTB = executeLevel(lvlTwo, lv1TwoSteps, lvlTwoFilters, df, dfTB, save_path_L2)
	print(f"\nData {start}-{end}, Filtered up to Level B2 saved to: {save_path_L2}")
	df, dfTB = executeLevel(lvlThree, lvlThreeSteps, lvlThreeFilters, df, dfTB, save_path_L3)
	print(f"\nData {start}-{end}, Filtered up to Level B3 saved to: {save_path_L3}")

	b1path = OUTPUT_DIR.joinpath(f"tableB1_{start[:7]}_{end[:7]}.parquet")
	dfTB = dfTB.T
	dfTB.to_parquet(b1path)

	return df, dfTB
if __name__ == "__main__": 

	df01, dfTB01 = appendixBfilter(start=START_DATE_01, end=END_DATE_01)

	df02, dfTB02 = appendixBfilter(start=START_DATE_02, end=END_DATE_02)

