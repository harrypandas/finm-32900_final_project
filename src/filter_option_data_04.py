import pandas as pd
import numpy as np
import config
from pathlib import Path 

import load_option_data_01 
import time 
from datetime import datetime, timedelta

from pandas_market_calendars import get_calendar


OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02

# Specify the exchange code (e.g., "XNYS" for NYSE)
exchange_code = "XNYS"
exchange_calendar = get_calendar(exchange_code)



def removeNotRepeated(df): 

	#Options sharing strike and expiration should be repeated. 
	#If not, then they are missing from the filtered data.
	option_id = ['secid', 'strike_price' , 'exdate', 'cp_flag']
	bool_Dup = df.duplicated(subset = option_id, keep = False)
	df = df[bool_Dup]
	return df 


if __name__ == "__main__": 

	start = START_DATE_01
	end = END_DATE_01
	save_path3 = DATA_DIR.joinpath( f"intermediate/data_{start[:7]}_{end[:7]}_L3filter.parquet")
	df = pd.read_parquet(save_path3)

	df_c = df[df['cp_flag'] == 'C']
	option_id = ['secid', 'strike_price' , 'exdate']
	L0 = len(df_c)
	df2 = removeNotRepeated(df_c)

	df3 = df2.groupby(by=option_id)

	current_date = datetime(2024,3,1)
	next_day = current_date + timedelta(days=1)

	# Find the next trading day
	while not exchange_calendar.valid_days(start_date=next_day, end_date=next_day):
	    next_day += timedelta(days=1)
	print(f"The next trading day for {current_date} on {exchange_code} is: {next_trading_day}")