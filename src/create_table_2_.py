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


START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02



def convertStr(x): 
	return x if type(x) == type('str') else f"{x:,.0f}"

def create2(T1_all, T2_all, T1_month, T2_month): 

	tableString = r"""

		\begin{tabular}{*{2}{l} *{15}{r} }
		       
		        
		         \multicolumn{2}{c}{}  & \multicolumn{7}{c}{Calls}  &  \multicolumn{1}{c}{} & 
		         \multicolumn{7}{c}{Puts} \\
		         
		          
		         \cline{3-9}
		         \cline{11-17}
		       
		         
		          \multicolumn{1}{l}{Observations} &  \multicolumn{1}{l}{} &
		          \multicolumn{3}{c}{""" +f"{START_DATE_01[:7]} to {END_DATE_01[:7]}" + r"""} & 
		          \multicolumn{1}{c}{} &
			\multicolumn{3}{c}{""" +f"{START_DATE_02[:7]} to {END_DATE_02[:7]}" + r"""} & 
			\multicolumn{1}{c}{} &
		          \multicolumn{3}{c}{""" +f"{START_DATE_01[:7]} to {END_DATE_01[:7]}" + r"""} & 
		          \multicolumn{1}{c}{} &
			\multicolumn{3}{c}{""" +f"{START_DATE_02[:7]} to {END_DATE_02[:7]}" + r"""} & 
		        

		       \hline
		       
		       \multicolumn{17}{c}{All trading days} \\ 
		       
		       \hline 

	""" 

	tableMiddle = r'''
        \hline
        
         \multicolumn{17}{c}{Last trading day of the month} \\

	'''



	tableEnd = r'''

	        \hline
	    \end{tabular}
	''' 


	t1 =  T1_all
	t2 = T2_all


	for k in ['Found', 'Missing', 'Expired']:
	#k = 'Found'

		w= "Calls"
		C1 = convertStr(t1[k][w])
		C1p = convertStr(100*t1[k][w]/t1['All'][w])

		C2 = convertStr(t2[k][w])
		C2p = convertStr(100*t2[k][w]/t2['All'][w])

		w = "Puts"
		P1 = convertStr(t1[k][w])
		P1p = convertStr(100*t1[k][w]/t1['All'][w])

		P2 = convertStr(t2[k][w])
		P2p = convertStr(100*t2[k][w]/t2['All'][w])


		oneline = r"""
		""" + f'{k}' + r""" &   & 
		""" + f"{C1}" + r""" &  & """ + f'{C1p}' + r"""\% & 
		 & 
		 """ + f'{C2}' + r""" & """ + f'{C2p}' + r"""\% & 
		 & 
		 """ + f'{P1}' + r""" &  & """ + f'{P1p}' + r"""\% & 
		 & 
		 """ + f'{P2}' + r"""& &""" + f'{P2p}' + r"""\% 
		 \\

		"""

		tableString = tableString + oneline

	tableString = tableString + tableMiddle

	t1 =  T1_month
	t2 = T2_month

	t1['Interpolated'] = t2['Missing'] + t2['Expired']
	t2['Interpolated'] = t2['Missing'] + t2['Expired']

	for k in ['Found', 'Interpolated']:
	#k = 'Found'

		w= "Calls"
		C1 = convertStr(t1[k][w])
		C1p = convertStr(100*t1[k][w]/t1['All'][w])

		C2 = convertStr(t2[k][w])
		C2p = convertStr(100*t2[k][w]/t2['All'][w])

		w = "Puts"
		P1 = convertStr(t1[k][w])
		P1p = convertStr(100*t1[k][w]/t1['All'][w])

		P2 = convertStr(t2[k][w])
		P2p = convertStr(100*t2[k][w]/t2['All'][w])


		oneline = r"""
		""" + f'{k}' + r""" &   & 
		""" + f"{C1}" + r""" &  & """ + f'{C1p}' + r"""\% & 
		 & 
		 """ + f'{C2}' + r""" & """ + f'{C2p}' + r"""\% & 
		 & 
		 """ + f'{P1}' + r""" &  & """ + f'{P1p}' + r"""\% & 
		 & 
		 """ + f'{P2}' + r"""& &""" + f'{P2p}' + r"""\% 
		 \\

		"""

		tableString = tableString + oneline	



	tableString = tableString + tableEnd

	path = OUTPUT_DIR / f'table2.tex'
	with open(path, "w") as text_file:
	    text_file.write(tableString)

if __name__ == "__main__": 

	table2_01_all = pd.read_parquet(OUTPUT_DIR.joinpath(f"table2_all_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet"))
	table2_01_month = pd.read_parquet(OUTPUT_DIR.joinpath(f"table2_month_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet"))

	# tableB1_02 = pd.read_parquet(Path(OUTPUT_DIR)  / f"tableB1_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet")
	# tableB1_02 = tableB1_02.replace({float('nan'): ''})
	table2_02_all = table2_01_all
	table2_02_month = table2_01_month 


	create2(table2_01_all,table2_02_all, table2_01_month,  table2_02_month)


	#createB1(tableB1_01, tableB1_02)