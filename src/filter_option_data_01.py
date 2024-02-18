import pandas as pd
import numpy as np
import wrds
import config
from pathlib import Path 


OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)




def fixStrike(df): 
	df['strike_price'] = df['strike_price']/1000
	return df 

def getSecPrice(df): 
	df['sec_price'] = (df['open'] + df['close'])/2
	return df 

def delete_identical_filter(df):
	columns_to_check = ['cp_flag', 'strike_price','date', 'exdate', 'option_price']
	df = df.drop_duplicates(subset=columns_to_check, keep='first')
	return df	

def appendixBfilter_level1(df): 
	#remove identical options (type, strike, experiation date, price)
	#price is defined on the buy side - so use best_bid?

	# df = df.drop_duplicates(subset = ['secid', 'date', 'cp_flag', 'strike_price', 'exdate', 'best_bid'])

	df = delete_identical_filter(df)

	#some are identical (type, strike, maturity, date) but different prices. 
	#KEEP closest to TBill based implied volatility of moneyness neighbors 
	#delete others 


	#Remove quotes of Bid = 0 

	return df 


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
	save_path = "./../data/sampledata.parquet"
	df = pd.read_parquet(save_path)
	df = fixStrike(df)
	df = getSecPrice(df)
	#duplicate 
	df = pd.concat([df, df], axis = 0 )
	dfB1 = appendixBfilter_level1(df)
	dfB1b = dfB1.drop_duplicates(subset = ['secid', 'date', 'cp_flag', 'strike_price', 'exdate'])

	dg = dfB1[dfB1.duplicated(subset = ['secid', 'date', 'cp_flag', 'strike_price', 'exdate'], keep = False)]
	dg.sort_values(by = 'date')

	