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

def createB1(T1, T2): 
	step_order = ['Starting', 'Level 1 filters', 'Level 2 filters', 'Level 3 filters', 'Final']
	
	
	


	tableString = r"""

    \begin{tabular}{*{4}{l} *{11}{r} }
       
        
         \multicolumn{4}{c}{}  & \multicolumn{3}{c}{OptionMetrics: """ + f'{START_DATE_01[:7]} to {END_DATE_01[:7]}' + r"""}  &  \multicolumn{1}{c}{} & 
         \multicolumn{3}{c}{OptionMetrics:""" + f'{START_DATE_02[:7]} to {END_DATE_02[:7]}' + r"""}&  \multicolumn{1}{c}{}  &
          \multicolumn{3}{c}{Total}  \\
         \cline{5-7}
                  
         \cline{9-11}
         \cline{13-15}
         
          &  & & & 
          Deleted &  & Remaining & &
          Deleted &  & Remaining & &
          Deleted &  & Remaining 
          \\

       \hline

	""" 
	tableEnd = r'''

	        \hline
	    \end{tabular}
	''' 
	for step in step_order: 
		group01 = tableB1_01[tableB1_01['Step']==step]
		group02 = tableB1_02[tableB1_02['Step']==step]
		for row in range(len(group01)):
			if row == 0: 
				stepstr = step 
			else: 
				stepstr = ' '
			rowname =  group01.iloc[row].name
			rowname = 'All' if rowname.find('All') == 0 else rowname
			rowname = 'Calls' if rowname.find('Calls') == 0 else rowname
			rowname = 'Puts' if rowname.find('Puts') == 0 else rowname
			
			g12_del = group01.iloc[row]['Deleted']
			g12_delS = convertStr(g12_del)

			g12_rem = group01.iloc[row]['Remaining']
			g12_remS = convertStr(g12_rem)

			g23_del = group02.iloc[row]['Deleted']
			g23_delS = convertStr(g23_del)

			g23_rem = group02.iloc[row]['Remaining']
			g23_remS = convertStr(g23_rem)

			gT_del = g12_del + g23_del
			gT_delS = convertStr(gT_del)

			gT_rem = g12_rem + g23_rem
			gT_remS = convertStr(gT_rem)

			rowString = f"""
				{stepstr} & & {rowname} & &
				{g12_delS} & & {g12_remS} & &
				{g23_delS} & & {g23_remS} & &
				{gT_delS} & & {gT_remS} {r"\\"}
			""" 
			tableString = tableString + rowString


	tableString = tableString + tableEnd

	path = OUTPUT_DIR / f'tableB1.tex'
	with open(path, "w") as text_file:
	    text_file.write(tableString)

if __name__ == "__main__": 

	tableB1_01 = pd.read_parquet(Path(OUTPUT_DIR)  / f"tableB1_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet")
	tableB1_01 = tableB1_01.replace({float('nan'): ''})

	# tableB1_02 = pd.read_parquet(Path(OUTPUT_DIR)  / f"tableB1_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet")
	# tableB1_02 = tableB1_02.replace({float('nan'): ''})
	tableB1_02 = tableB1_01

	createB1(tableB1_01, tableB1_02)
