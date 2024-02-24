import pandas as pd
import numpy as np
import wrds
import config
from pathlib import Path 
import time 

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME


#https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics/ivy-db-us/options/option-prices/
description_opt_Met = {
	"secid": "Security ID",
	"cusip": "CUSIP Number",
	"date": "date", 
	"symbol": "symbol", 
	"exdate": "Expiration date of the Option", 
	"last_date": "Date of last trade", 
	"cp_flag": "C=call, P=Put", 
	"strike_price": "Strike Price of the Option TIMES 1000", 
	"best_bid": "Highest Closing Bid Across All Exchanges", 
	"best_offer": "Lowest Closing Ask Across All Exchanges",
	"open_interest": "Open Interest for the Option", 
	"impl_volatility": "Implied Volatility of the Option", 
	"exercise_style": "(A)merican, (E)uropean, or ? (exercise_style)",


} 

#https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/optionmetrics/wrds-overview-optionmetrics/






def sql_query(year = 1996, start = '1996-01-01', end = '2012-01-31'): 

	#use PostgreSQL
	#https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/optionmetrics/wrds-overview-optionmetrics/
	#https://wrds-www.wharton.upenn.edu/data-dictionary/optionm_all/opprcd2023/ 
	#https://wrds-www.wharton.upenn.edu/data-dictionary/optionm_all/secprd1996/

	#https://wrds-www.wharton.upenn.edu/data-dictionary/frb_all/rates_daily/
	sql_query = f"""
		SELECT 
			b.secid, b.date,  
			b.open, b.close, 
			a.cp_flag, 
			a.exdate, a.impl_volatility, c.tb_m3, a.volume, a.open_interest,
			a.best_bid, a.best_offer, a.strike_price, a.contract_size
		FROM
			optionm_all.opprcd{year} AS a
		JOIN 
			optionm_all.secprd{year} AS b ON a.date = b.date AND a.secid = b.secid

		JOIN 
			frb_all.rates_daily AS c ON c.date = a.date 

		WHERE
			(a.secid = 108105) 
		AND 
			(a.date >= \'{start}\')
		AND 
			(a.date <= \'{end}\') 		
	""" 

	return sql_query



def pull_Year_Range(wrds_username = WRDS_USERNAME, yearStart = 1996, yearEnd = 2012, start = '1996-01-01', end = '2012-01-31'):

	db = wrds.Connection(wrds_username=wrds_username, verbose = False)
	dlist = []
	for year in range(yearStart, yearEnd + 1): 
		t0 = time.time()
		#print(year)
		sql = sql_query(year = year, start = start, end = end)
		dftemp = db.raw_sql(sql, date_cols = ["date", "exdate"])
		dlist.append(dftemp)
		t1 = round(time.time()-t0,2)
		print(f"{year} took {t1} seconds")

	df = pd.concat(dlist, axis = 0)
	db.close()
	return df


def load_all_optm_data(data_dir=DATA_DIR,
                        wrds_username=WRDS_USERNAME,
                        startDate="1996-01-01",
                        endDate="2012-01-31"):
	
	yearStart = int(startDate[:4])
	yearEnd = int(endDate[:4])	
	startYearMonth = startDate[:7]
	endYearMonth = endDate[:7]
	
	file_name = f"data_{startYearMonth}_{endYearMonth}.parquet"
	file_path = Path(data_dir) / "pulled" / file_name
	t0 = time.time()
	if file_path.exists():
		print(f'Reading from file: {file_path}')
		df = pd.read_parquet(file_path)
	else:
		df = pull_Year_Range(wrds_username=wrds_username, yearStart=yearStart, yearEnd=yearEnd, start=startDate, end=endDate)
		file_dir = file_path.parent
		file_dir.mkdir(parents=True, exist_ok=True)
		df.to_parquet(file_path)
	
	df = clean_optm_data(df)
	t1 = round(time.time()-t0,2)
	print(f'Loading Data took {t1} seconds')
	return df

def clean_optm_data(df):
	df['strike_price'] = df['strike_price']/1000
	df['tb_m3'] = df['tb_m3']/100
	df['date'] = pd.to_datetime(df['date'])
	return df



def run_load_all_optm_data(data_dir=DATA_DIR,
                        wrds_username=WRDS_USERNAME,
                        startDate="1996-01-01",
                        endDate="2012-01-31"):

	load_all_optm_data(data_dir=data_dir,
						wrds_username=wrds_username, 
						startDate=startDate,
						endDate=endDate)

	return f"Loading Data from OptionMetrics between {startDate} and {endDate}"



if __name__ == "__main__": 
	#x = pull_Option_info()
	#y = pull_Security_info()
	#z = pull_Opt_Sec_info()
	#a = pull_FedH15()
	#b = pull_Opt_Sec_info_WRDS()
	## Run with doit 
	# df = pull_Year_Range(yearStart = 1996, yearEnd = 2012)
	# df.reset_index(drop = True)

	data_199601_201201 = load_all_optm_data(data_dir=DATA_DIR,
											wrds_username=WRDS_USERNAME, 
											startDate="1996-01-01",
											endDate="2012-01-31")

	data_201202_202312 = load_all_optm_data(data_dir=DATA_DIR,
										 	wrds_username=WRDS_USERNAME, 
											startDate="2012-02-01",
											endDate="2023-12-31")
										
	# save_path = DATA_DIR.joinpath( "data_1996_2012.parquet")
	# df.to_parquet(save_path)
	