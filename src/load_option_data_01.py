import pandas as pandas
import numpy as np
import wrds
import config
from pathlib import Path 


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

#want stdopd, secprd, 
# db = wrds.Connection(wrds_username=wrds_username)
# x = [ y  for y in db.list_tables(library = 'optionm_all') if y.find('opprcd') != -1]
#a.exercise_style,

#https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/jupyterhub-wrds/
#https://wrds-www.wharton.upenn.edu/documents/1504/IvyDB_US_v5.4_Reference_Manual.pdf

# def pull_Option_info(wrds_username = WRDS_USERNAME, year = 1996): 
# 	sql_query = f"""
# 		SELECT 
# 			a.date, a.secid, a.days,  
# 			a.cp_flag, a.strike_price, a.forward_price
# 		FROM
# 			optionm_all.stdopd{year} AS a
# 		WHERE 
# 			a.strike_price BETWEEN .8*a.forward_price and 1.2*a.forward_price
# 			AND
# 			a.days BETWEEN '0' and '100'
# 		LIMIT 1000
# 	""" 
# 	#LIMIT 1000
# 	db = wrds.Connection(wrds_username=wrds_username)
# 	optm = db.raw_sql(sql_query, date_cols = ["date"])
# 	db.close()
# 	return optm

'''
	WHERE 
			a.strike_price BETWEEN .8*a.forward_price AND 1.2*a.forward_price
	
		AND

'''

def pull_Option_info(wrds_username = WRDS_USERNAME, year = 1996): 
	#use PostgreSQL
	#https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/optionmetrics/wrds-overview-optionmetrics/
	#https://wrds-www.wharton.upenn.edu/data-dictionary/optionm_all/opprcd2023/ 

	sql_query = f"""
		SELECT 
			a.date, a.secid, a.exdate,  
			a.cp_flag, a.strike_price, a.forward_price
		FROM
			optionm_all.opprcd{year} AS a
		WHERE
			a.exdate BETWEEN a.date AND (a.date + INTERVAL '100 days')
		LIMIT 100
	""" 
	#LIMIT 1000
	db = wrds.Connection(wrds_username=wrds_username)
	optm = db.raw_sql(sql_query, date_cols = ["date"])
	db.close()
	return optm

def pull_Security_info(wrds_username = WRDS_USERNAME, year = 1996): 
	#use PostgreSQL
	#https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/optionmetrics/wrds-overview-optionmetrics/
	#https://wrds-www.wharton.upenn.edu/data-dictionary/optionm_all/secprd1996/
	sql_query = f"""
		SELECT 
			b.date, b.secid,  
			b.open, b.close
		FROM
			optionm_all.secprd{year} AS b
		LIMIT 100
	""" 
	#LIMIT 1000
	db = wrds.Connection(wrds_username=wrds_username)
	optm = db.raw_sql(sql_query, date_cols = ["date"])
	db.close()
	return optm



def pull_Opt_Sec_info(wrds_username = WRDS_USERNAME, year = 1996): 
	#use PostgreSQL
	#https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/optionmetrics/wrds-overview-optionmetrics/
	#https://wrds-www.wharton.upenn.edu/data-dictionary/optionm_all/opprcd2023/ 
	#https://wrds-www.wharton.upenn.edu/data-dictionary/optionm_all/secprd1996/

	#a.secid, a.date,
	sql_query = f"""
		SELECT 
			 b.secid, b.date,  
			b.open, b.close, 
			a.cp_flag, 
			a.exdate, a.strike_price, a.impl_volatility 
		FROM
			optionm_all.opprcd{year} AS a
		JOIN 
			optionm_all.secprd{year} AS b ON a.date = b.date AND a.secid = b.secid
		WHERE
			(a.exdate BETWEEN (a.date + INTERVAL '25 days') AND (a.date + INTERVAL '35 days') )
		OR 
			(a.exdate BETWEEN (a.date + INTERVAL '55 days') AND (a.date + INTERVAL '65 days'))
		OR
			(a.exdate BETWEEN (a.date + INTERVAL '85 days') AND (a.date + INTERVAL '95 days'))
		AND
			(a.strike_price BETWEEN (0.875*0.5*(b.open+b.close) ) AND (1.125*0.5*(b.open+b.close)))
		LIMIT 2000; 
		
	""" 
	#LIMIT 1000
	db = wrds.Connection(wrds_username=wrds_username)
	optm = db.raw_sql(sql_query, date_cols = ["date"])
	db.close()
	return optm
if __name__ == "__main__": 
	#x = pull_Option_info()
	#y = pull_Security_info()
	z =pull_Opt_Sec_info()
	# wrds_username = WRDS_USERNAME
	# db = wrds.Connection(wrds_username=wrds_username)