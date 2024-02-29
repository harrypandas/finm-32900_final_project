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







if __name__ == "__main__": 


	start = START_DATE_01
	end = END_DATE_01

	## Prepare trading Days: 
	nextTradingDays =  calendar.valid_days(start_date=start, end_date=end , tz = None)
	dfTrade = pd.DataFrame({"date": nextTradingDays})
	dfTrade['nextTradingDay'] = dfTrade['date'].shift(-1)

	#Insert dataframe: 
	save_path3 = DATA_DIR.joinpath( f"intermediate/data_{start[:7]}_{end[:7]}_L3filter.parquet")
	df = pd.read_parquet(save_path3)

	#Options sharing strike and expiration should be repeated. 
	#If not, then they become missing and then lost in the filtered data.
	df = removeNotRepeated(df)


	#Associate Trading Days with their next Trading Day. 
	df = pd.merge(df, dfTrade, on='date', how='inner')

	#Add a dummy date column
	df['dateC'] = df['date']

	#Group by 'secid', 'strike_price' , 'exdate', 'cp_flag', then lag the days in time
	option_id = ['secid', 'strike_price' , 'exdate', 'cp_flag']
	df = with_lagged_columns( df, columns_to_lag = ['dateC'], id_columns = option_id, 
		lags = -1, date_col = 'date', prefix = '')

	
	#If the next time the option appears is the following trading day, it is found! 
	df['found'] = (df['nextTradingDay']) == df['-1_dateC']
	boolsFound = df['found']
	df_Found = df[boolsFound].reset_index(drop =True)

	#Else it is missing: 
	df_Missing = df[~boolsFound].reset_index(drop =True)



	#Now we check if the option will ever return: 
	df_Missing = df_Missing.drop(columns = ["-1_dateC"])
	df_Missing = with_lagged_columns( df_Missing, columns_to_lag = ['dateC'], id_columns = option_id, 
		lags = -1, date_col = 'date', prefix = '')

	
	#Now we check to see if the option dissapears before maturity: 
	df_Missing['expired'] = (df_Missing['time_to_maturity'] <= 3)
	boolsExpired = df_Missing['expired']
	df_Missing_Expired = df_Missing[boolsExpired]

	#Else it is still missing: 
	df_Missing = df_Missing[~boolsExpired]



	#PANDA.TSERIES.OFFSETS FOR BUSINESS DAYS; BDAY
	dg = df_Missing
	dfsort  = df.groupby(option_id).apply(lambda group: group.sort_values('date'))
	# h = dfsort[['exdate', 'date', 'nextTradingDay', '-1_dateC', 'found',  'time_to_maturity']].head(20)



	dg = df.apply(lambda x: len(pd.bdate_range(start=x['date'], end=x['exdate']))) 