import pandas as pd
import numpy as np
import wrds
import config
from pathlib import Path 

import bsm_pricer
from scipy.optimize import minimize

import load_option_data_01 

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)

def getLengths(df): 
	test1 = df['cp_flag'].value_counts().to_dict()
	test1C = test1['C']
	test1P = test1['P'] 
	test1L = len(df)
	return np.array([test1L, test1C, test1P])


def fixStrike(df): 
	df['strike_price'] = df['strike_price']/1000
	return df 

def getSecPrice(df): 
	df['sec_price'] =  df['close']
	df['mnyns'] = df['strike_price']/df['sec_price']
	return df 

# def implied_volatility(row):
#     if row['cp_flag'] == 'C':
#         objective_function = lambda sigma:  (bsm_pricer.european_call_price(row['sec_price'], row['strike_price'],
#         	row['tb_m3']/(100*365),(row['exdate'] - row['date']).total_seconds()/(24*60*60), 
#         	sigma) - row['best_bid'])**2
#     elif row['cp_flag'] == 'P':
#         objective_function = lambda sigma: (bsm_pricer.european_put_price(row['sec_price'], row['strike_price'],
#         	row['tb_m3']/(100*365),(row['exdate'] - row['date']).total_seconds()/(24*60*60), 
#         	sigma) - row['best_bid'])**2
#     else:
#         raise ValueError("Invalid option type. Use 'C' or 'P'.")

#     result = minimize(objective_function, 0.2, bounds=[(0, None)])
#     return result.x[0]

# def bsm_volatility(df): 
# 	df['BSM_sig'] = df.apply(implied_volatility, axis=1)
# 	# df['BSM_sig'] = df.apply(bsm_pricer.european_sigma(df['best_bid'], 
# 	# 	df['sec_price'], df['strike_price'], df['tb_m3']/(100*365), (df['exdate'] - df['date']) ,type = df[] ), 
# 	# 		axis = 1)
# 	return df 


def delete_identical_filter(df):
	#remove identical options (type, strike, experiation date, price)
	#price is defined on the buy side - so use best_offer: 2/19/24 discussion with Viren
	columns_to_check = ['secid', 'cp_flag', 'strike_price','date', 'exdate', 'best_offer']
	df = df.drop_duplicates(subset=columns_to_check, keep='first')
	return df	

def delete_identical_but_price_filter(df): 

	#some are identical (type, strike, maturity, date) but different prices. 
	#KEEP closest to TBill based implied volatility of moneyness neighbors 
	#delete others 


	#Get Bools of duplicated row: 
	bool_Dup = df.duplicated(subset = ['secid', 'date', 'cp_flag', 'strike_price', 'exdate'], keep = False)

	#remove duplicated 
	df_noDup = df[~bool_Dup]

	#grab duplicates 
	df_Dup = df[bool_Dup]
	df_Dup = df_Dup.sort_values(by=[ 'cp_flag', 'date']).reset_index(drop = True)

	##find moneyness neighbors
	huntlist = ['secid', 'cp_flag', 'date', 'exdate']
	mask = df.set_index(huntlist).index.isin(df_Dup.set_index(huntlist).index)

	#all of the moneyness neighbors: 
	df_Neigh = df[mask]
	df_Neigh = df_Neigh.reset_index(drop =True)

	#in the money neighbors: 
	m1 = df_Neigh.groupby(by = huntlist).apply(lambda x: ((x['mnyns']-1)**2).idxmin())
	dMon = df_Neigh.loc[m1]
	dMon['mon_vola'] = dMon['impl_volatility']
	dMonSub = dMon[huntlist + ['mon_vola']]

	##Join the ITM volatility with the correct option: 
	df_Join = pd.merge(df_Dup, dMonSub, on = huntlist, how = 'inner')
	df_Join['impl_volatility2'] = df_Join['impl_volatility']
	df_Join['impl_volatility2'].fillna(0, inplace=True)
	#findimplied volatility being closest to ITM: 
	idx_keep = df_Join.groupby(by = huntlist).apply(lambda x: ((x['impl_volatility2']-x['mon_vola']).abs()).idxmin())

	#Tidy up the reduced subset of duplicates
	df_reduced= df_Join.loc[idx_keep]
	df_reduced = df_reduced.sort_values(by=[ 'cp_flag', 'date']).reset_index(drop=True)
	df_reduced.drop(['impl_volatility2', 'mon_vola'], axis = 1, inplace = True)

	#Combine the OG dataframe with No Duplicates with the reduced subset of duplicates
	df = pd.concat([df_noDup, df_reduced], ignore_index = True)
	df = df.sort_values(by=[ 'cp_flag', 'date']).reset_index(drop = True)


	return df 

def delete_zero_bid_filter(df): 
	df = df[df['best_bid'] != 0.0]
	return df 

def delete_zero_volume_filter(df): 
	df = df[df['volume'] != 0.0]
	return df 


def appendixBfilter_level1(df): 



	rows = ["Total", "Calls", "Puts"]
	df_sum = pd.DataFrame(index = rows)

	L0 = getLengths(df)
	df_sum['Starting'] = L0


	df = delete_identical_filter(df)
	L1 = getLengths(df)
	df_sum['Identical'] =  L0-L1


	df = delete_identical_but_price_filter(df)
	L2 = getLengths(df)
	df_sum['Identical but Price'] = L1-L2


	df = delete_zero_bid_filter(df)
	L3 = getLengths(df)
	df_sum['Bid = 0'] =  L2-L3

	df = delete_zero_volume_filter(df)
	L4 = getLengths(df)
	df_sum['Volume = 0'] =   L3-L4
	df_sum['Final'] =  L4
	

	rows_B = ['Step', 'Deleted', 'Remaining']
	df_B1 = pd.DataFrame(index = rows_B)
	df_B1['Calls'] = ['Starting', float('nan'), L0[1]]
	df_B1['Puts'] = ['Starting', float('nan'), L0[2]]
	df_B1['All0'] = ['Starting', float('nan'), L0[0]]

	df_B1['Identical'] = ['Level 1 filters', (L0-L1)[0],float('nan')]
	df_B1['Identical except price'] = ['Level 1 filters', (L1-L2)[0],float('nan')]
	df_B1['Bid = 0'] = ['Level 1 filters', (L2-L3)[0],float('nan')]
	df_B1['Volume = 0'] = ['Level 1 filters', (L3-L4)[0],float('nan')]
	df_B1['All1'] = ['Level 1 filters', float('nan'), L4[1]]

	df_B1 = df_B1.T
	df_sum = df_sum.T
	return df, df_sum, df_B1


def appendixBfilter_level2(df): 
	#remove options not within range of (7,180)

	#remove options with implied volatility outside of [0.05,1.00]


	#remove options with moneyness outside of [.8, 1.2] #Done by SQL

	#looks like they calculated implied interest rate??? to get volatility?

	return df 

def appendixBfilter_level3(df): 
	#remove implied volatility outliers??? 

	#Ensure Put-call parity 

	return df 
def group54port(df): 

	#throw into portfolio dictionary that can be accessed as portfolios['C']['30']['0.975']
	cpBools = df['cp_flag'] == 'C'
	dfc = df[cpBools]
	dfp = df[-cpBools]

	center_days = np.array([30.0, 60.0, 90.0])
	bw_days = 10.0
		
	center_money = np.linspace(0.9, 1.1, 9)
	bw_money = 0.0125 

	daykeys =  [f'{day:.0f}' for day in center_days]
	moneykeys = [f'{mon:.3f}' for mon in center_money]


	dfdict = dict(zip(['C','P'] , [dict(
			zip(daykeys, 
				[dict(
					zip(
						moneykeys, [ list() for _ in range(len(moneykeys))]
						) 
					) for _ in range(len(daykeys))]
				)
			) for _ in range(2)] 
	))
	
	return df 




if __name__ == "__main__": 
	
	df = load_option_data_01.load_all_optm_data()
	df = fixStrike(df)
	df = getSecPrice(df)
	#duplicate 
	# df = pd.concat([df, df], axis = 0 )



	dfB1, df_sum, df_tableB1 = appendixBfilter_level1(df)
	

	''' 
	                        Total     Calls     Puts
	Starting              3410580   1704220  1706360
	Identical                  -3        -2       -1
	Identical but Price        -7        -3       -4
	Bid = 0               -272078   -152680  -119398
	Volume = 0           -2093744  -1122939  -970805


	'''
	
	

	save_path = DATA_DIR.joinpath( "data_1996_2012_appendixB.parquet")
	df.to_parquet(save_path)
	# @ihammock try this: df_tableB1 = df_tableB1.reset_index().rename(columns={'index': 'Substep'}).set_index(['Step', 'Substep'])
	df_tableB1.to_parquet(OUTPUT_DIR.joinpath("tableB1_2012.parquet"))


	## Suppress scientific notation and limit to 3 decimal places
	# Sets display, but doesn't affect formatting to LaTeX
	pd.set_option('display.float_format', lambda x: '%.2f' % x)
	# Sets format for printing to LaTeX
	float_format_func = lambda x: '{:.2f}'.format(x)
	tableB_2012 = df_tableB1.to_latex(float_format=float_format_func)



	path = OUTPUT_DIR / f'tableB1_2012.tex'
	with open(path, "w") as text_file:
	    text_file.write(tableB_2012)


