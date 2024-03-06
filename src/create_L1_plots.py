import filter_option_data_02 as f2
import matplotlib.pyplot as plt
import pandas as pd
import config
from pathlib import Path
import load_option_data_01 as l1


OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02

"""
This file will create a .tex table that compares the amount of options with 
no volume nor open interest, before and after the application of B1 filters. 


"""

def deltaL(df0, df1): 
    return len(df1)-len(df0)


def overlap(df): 
    df = df[(df['volume'] == 0) & (df['open_interest'] == 0)]
    return df

def find_0vol(df):
    df = df[df['volume'] == 0.0]
    return df 

def find_0int(df):
    df = df[df['open_interest'] == 0.0]
    return df 

if __name__ == "__main__":
    date_ranges = [f'{START_DATE_01[:7]}_{END_DATE_01[:7]}' ,f'{START_DATE_02[:7]}_{END_DATE_02[:7]}']
    
    print("Creating Level 1 extra table...")
    for date_range in date_ranges:
        # load data with level 1 filters applied
        df = pd.read_parquet(DATA_DIR / "pulled" / f"data_{date_range}.parquet")

        df3 = pd.read_parquet(DATA_DIR / "intermediate" / f"data_{date_range}_L3filter.parquet")

        table = pd.DataFrame(index = ['Before (N)', r'Before (\%)', 'After B1 (N)', r'After B1 (\%)'])

        dfvB = find_0vol(df)
        dfvA = find_0vol(df3)

        table['Volume = 0'] = [len(dfvB), 100*len(dfvB)/len(df), len(dfvA), 100*len(dfvA)/len(df3) ]

        dfvB = find_0int(df)
        dfvA = find_0int(df3)

        table['Open Interest = 0'] = [len(dfvB), 100*len(dfvB)/len(df), len(dfvA), 100*len(dfvA)/len(df3) ]

        dfvB = overlap(df)
        dfvA = overlap(df3)

        table['Overlap']= [len(dfvB), 100*len(dfvB)/len(df), len(dfvA), 100*len(dfvA)/len(df3) ]
        table = table.T


        pd.set_option('display.float_format', lambda x: '%.0f' % x)
        # Sets format for printing to LaTeX
        float_format_func = lambda x: f"{x:,.0f}"



        latex_table_string = table.to_latex(float_format=float_format_func)
        #print(latex_table_string)

        path = OUTPUT_DIR / f'L1_noVol_noInt_{date_range}.tex'
        with open(path, "w") as text_file:
            text_file.write(latex_table_string)