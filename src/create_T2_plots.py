
import matplotlib.pyplot as plt
import pandas as pd
import config
from pathlib import Path

from pandas_market_calendars import get_calendar

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02

"""
This file will create a .tex table that illustrates the amount of options expiring not on a NYSE trading day. 



"""

# Specify the exchange code (e.g., "XNYS" for NYSE)
exchange_code = "XNYS"
calendar = get_calendar(exchange_code)

def countExpireTradingDays(df):
    ## Prepare trading Days: 
    startt = df['date'].min().date() 
    endd = df['date'].max().date()
    TradingDays =  calendar.valid_days(start_date=startt, end_date=endd , tz = None)    

    ## Assign an integer to each day (faster to take differences)   
    dfNewCal = pd.DataFrame({"date2": TradingDays})
    dfNewCal['ex_num'] = list(range(len(dfNewCal)))

    ## Take our 'exdate' and send it to integer 
    df2 = pd.merge(df, dfNewCal, left_on='exdate', right_on='date2', how='inner')
    df2 = df2.drop(columns = ["date2"])

    N = df2.shape[0]
    return N 

def countExpireSatDays(df): 
    startt = df['date'].min().date() 
    endd = df['date'].max().date()
    date_range = pd.date_range(start=startt, end=endd, freq='D')


    saturdays = date_range[date_range.weekday == 5] 
    dfSatCal = pd.DataFrame({"date2": saturdays})
    dfSatCal['ex_num'] = list(range(len(dfSatCal)))


    df2 = pd.merge(df, dfSatCal, left_on='exdate', right_on='date2', how='inner')

    N = df2.shape[0]
    return N 
      
    
def toPer(x): 
    return f"{100*x:,.0f}"+r"\%"

if __name__ == "__main__":
    date_ranges = [f'{START_DATE_01[:7]}_{END_DATE_01[:7]}' ,f'{START_DATE_02[:7]}_{END_DATE_02[:7]}']
    datekey = [f'{START_DATE_01[:7]} to {END_DATE_01[:7]}' ,f'{START_DATE_02[:7]} to {END_DATE_02[:7]}']
    print("Creating Table 2 extra table...")
    table = pd.DataFrame(index = ['Total Options', 'Trading Days', 'Saturdays', 'Other Days'])

    for (dk, date_range) in zip(datekey, date_ranges):
        # load data with level 1 filters applied
        

        df3 = pd.read_parquet(DATA_DIR / "intermediate" / f"data_{date_range}_L3filter.parquet")
        df3 = df3.reset_index()
        Ntot = df3.shape[0]
        Ntd = countExpireTradingDays(df3)
        Nsat = countExpireSatDays(df3)
        Nmiss = Ntot -(Ntd+Nsat)

        table[f'{dk}'] = [Ntot, toPer(Ntd/Ntot), toPer(Nsat/Ntot), toPer(Nmiss/Ntot)]
    float_format_func = lambda x: f"{x:,.0f}"

    latex_table_string = table.to_latex(float_format=float_format_func)
    path = OUTPUT_DIR / f'T2_days.tex'
    with open(path, "w") as text_file:
        text_file.write(latex_table_string)