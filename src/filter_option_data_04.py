import pandas as pd
import numpy as np
import config
from pathlib import Path 

import load_option_data_01 
import time 
from datetime import datetime, timedelta

from pandas_market_calendars import get_calendar

from misc_tools import with_lagged_columns

from multiprocessing import Pool



OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02

# Specify the exchange code (e.g., "XNYS" for NYSE)
exchange_code = "XNYS"
calendar = get_calendar(exchange_code)



def removeNotRepeated(df): 

	#Options sharing strike and expiration should be repeated. 
	#If not, then they become missing and then lost in the filtered data.
	option_id = ['secid', 'strike_price' , 'exdate', 'cp_flag']
	bool_Dup = df.duplicated(subset = option_id, keep = False)
	df = df[bool_Dup]
	return df 




def enumerateDays(df): 
	## Prepare trading Days: 
	startt = df['date'].min().date() 
	endd = df['date'].max().date()
	TradingDays =  calendar.valid_days(start_date=startt, end_date=endd , tz = None)	

	## Assign an integer to each day (faster to take differences) 	
	dfNewCal = pd.DataFrame({"date2": TradingDays})
	dfNewCal['da_num'] = list(range(len(dfNewCal)))

	## Take our 'date' and send it to integer 
	df = pd.merge(df, dfNewCal, left_on='date', right_on='date2', how='inner')
	df = df.drop(columns = ["date2"])

	## Prepare new Cal to do the same with expiration date
	dfNewCal['ex_num'] = dfNewCal['da_num']
	dfNewCal = dfNewCal.drop(columns = ['da_num'])

	## Take our 'exdate' and send it to integer 
	df = pd.merge(df, dfNewCal, left_on='exdate', right_on='date2', how='inner')
	df = df.drop(columns = ["date2"])

	## while we're at it, lets get the time to expiration 
	df['expTime'] = df['ex_num']-df['da_num']
	return df 


def daysLost(df): 
	#Group by option ID, then lag by an instance. 
	option_id = ['secid', 'strike_price' , 'exdate', 'cp_flag']
	df = with_lagged_columns( df, columns_to_lag = ['da_num'], id_columns = option_id, 
		lags = -1, date_col = 'date', prefix = 'L')

	#IF there is no day to lag to (nan) set that to the expiration date 
	df['L-1_da_num'] = df['L-1_da_num'].fillna(df['ex_num'])

	#Get days that the option is lost 
	df['days_lost'] = df['L-1_da_num'] - df['da_num']

	return df 







def pushRestExpir(df):
	## Some expire on a friday thats not a trading day? so just pushed it back 
	startt = df['date'].min().date() 
	endd = df['date'].max().date()
	TradingDays =  calendar.valid_days(start_date=startt, end_date=endd , tz = None)

	#daysf = [1,2] 
	#while len(daysf) > 0: 
	L = set(sorted(df['exdate'].unique()))
	L1 = set(TradingDays)

	daysf = sorted(L-L1)
	dfDays = pd.DataFrame({"date2": daysf})
	dfDays['shift'] = dfDays-timedelta(1)

	df = pd.merge(df, dfDays, left_on='exdate', right_on='date2', how='left')


	df['exdate'] = np.where(~df['shift'].isna(), df['shift'], df['exdate'])

	df = df.drop(columns=['date2', 'shift']) 

	return df 

def pushWeekendExpiration(df): 
	#For some reason, some options expire on saturday or sunday??? This will just push it back to previous friday. 

	startt = df['date'].min().date() 
	endd = df['date'].max().date()
	date_range = pd.date_range(start=startt, end=endd, freq='D')

	fridays = date_range[date_range.weekday == 4]

	for day in [5,6]: 
		saturdays = date_range[date_range.weekday == day] 
		dfSatCal = pd.DataFrame({"date2": saturdays})
		dfSatCal['fri'] = fridays


		df = pd.merge(df, dfSatCal, left_on='exdate', right_on='date2', how='left')


		df['exdate'] = np.where(~df['fri'].isna(), df['fri'], df['exdate'])

		df = df.drop(columns=['date2', 'fri']) 

	return df 

def endOfMonth(df): 
	## Prepare trading Days: 
	startt = df['date'].min().date() 
	endd = df['date'].max().date()
	TradingDays =  calendar.valid_days(start_date=startt, end_date=endd , tz = None)	

	TDseries = TradingDays.to_series()

	ends = pd.to_datetime(TDseries.groupby(TDseries.dt.strftime('%Y-%m')).max())

	dfMonthCal = pd.DataFrame({"date2": ends})
	dfMonthCal['m'] = ends 


	df = pd.merge(df, dfMonthCal, left_on='exdate', right_on='date2', how='inner')
	df = df.drop(columns=['date2', 'm']) 

	return df

def table2Logic(df): 

	#if the expiration time equals the days lost, then the option is missing as expired: 
	dfExp = df[ df['days_lost'] == df['expTime']]
	expMissingDays = dfExp['days_lost'].sum()
	expOptions = len(dfExp)

	#if the days_lost == 1, then these options are found: 
	dfFound = df[ df['days_lost'] == 1]
	foundOptions = len(dfFound)

	#if the days_lost are not equal to one nor the expiration time, these options were missing, then found again: 
	dfMissing = df[ ( 1 <  df['days_lost'] ) & (df['days_lost']<  df['expTime']) ]

	#dfMissing = df[ ( 1 <  df['days_lost'] ) & (df['days_lost']<  10) ]
	missingOptions = len(dfMissing)


	return {"found": foundOptions, "miss": missingOptions, "exp": expOptions, }

def table2_info(df):
	rows_2 = ['All', 'Found', 'Missing', 'Expired']
	dT2 = pd.DataFrame(index = rows_2)
	dfc = df[df['cp_flag']=="C"]
	report = table2Logic(dfc)
	dT2['Calls'] = [len(dfc), report['found'], report['miss'], report['exp']]
	dfp = df[df['cp_flag']=="P"]
	report = table2Logic(dfp)
	dT2['Puts'] = [len(dfp), report['found'], report['miss'], report['exp']]

	dT2 = dT2.T

	return dT2

def table2_analysis(start, end): 

	# start = START_DATE_01
	# end = END_DATE_01
	#Insert dataframe: 
	save_path3 = DATA_DIR.joinpath( f"intermediate/data_{start[:7]}_{end[:7]}_L3filter.parquet")
	df= pd.read_parquet(save_path3)
	

	df = pushWeekendExpiration(df)
	df = pushRestExpir(df)
	#Convert date to integer. Going to use relative distance (in trading days) from start date
	df = enumerateDays(df)


	#Determine how many days until the options next occurence. If it doesnt occur, this day is the expiration 
	df = daysLost(df)

	dfM = endOfMonth(df)


	dT = table2_info(df)
	dT.to_parquet(OUTPUT_DIR.joinpath(f"table2_all_{start[:7]}_{end[:7]}.parquet"))
	dTM = table2_info(dfM)
	dTM.to_parquet(OUTPUT_DIR.joinpath(f"table2_month_{start[:7]}_{end[:7]}.parquet"))

	return df, dT, dTM

if __name__ == "__main__": 








	df1, dT1, dTM1 = table2_analysis(START_DATE_01, END_DATE_01)


	#df2, dT2, dTM2 = table2_analysis(START_DATE_02, END_DATE_02)



# dg = df.sort_values('date').head(5000)
# option_id = ['secid', 'strike_price' , 'exdate', 'cp_flag']
# dfsort  = dg.groupby(option_id).apply(lambda group: group.sort_values('date'))
# h = dfsort[['exdate', 'date', 'da_num', 'L-1_da_num', 'ex_num', 'days_lost']]
