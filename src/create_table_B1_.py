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

def convertStr(x): 
	return x if type(x) == type('str') else f"{x:,.0f}"


if __name__ == "__main__": 

	tableB1_2012 = pd.read_parquet(Path(OUTPUT_DIR)  / "tableB1_2012.parquet")

	tableB1_2023 = pd.read_parquet(Path(OUTPUT_DIR)  / "tableB1_2012.parquet")

	step_order = ['Starting', 'Level 1 filters']

	# Convert 'Step' to a categorical type with the specified order
	tableB1_2012['Step'] = pd.Categorical(tableB1_2012['Step'], categories=step_order, ordered=True)

	# Sort the DataFrame by the 'Step' column
	tableB1_2012.sort_values(by='Step', inplace=True)
	tableB1_2012 = tableB1_2012.replace({float('nan'): ''})

	# Convert the DataFrame to LaTeX
	latex_table = tableB1_2012[['Deleted', 'Remaining']].to_latex(index=True)



	tableString = r"""

    \begin{tabular}{*{4}{l} *{11}{r} }
       
        
         \multicolumn{4}{c}{}  & \multicolumn{3}{c}{OptionMetrics: 2012}  &  \multicolumn{1}{c}{} & 
         \multicolumn{3}{c}{OptionMetrics: 2023}&  \multicolumn{1}{c}{}  &
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
		group2012 = tableB1_2012[tableB1_2012['Step']==step]
		group2023 = tableB1_2012[tableB1_2012['Step']==step]
		for row in range(len(group2012)):
			if row == 0: 
				stepstr = step 
			else: 
				stepstr = ' '
			rowname =  group2012.iloc[row].name
			rowname = rowname if rowname.find('All') == -1 else 'All'
			
			g12_del = group2012.iloc[row]['Deleted']
			g12_delS = convertStr(g12_del)

			g12_rem = group2012.iloc[row]['Remaining']
			g12_remS = convertStr(g12_rem)

			g23_del = group2023.iloc[row]['Deleted']
			g23_delS = convertStr(g23_del)

			g23_rem = group2023.iloc[row]['Remaining']
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

	path = OUTPUT_DIR / f'tableB1_2012.tex'
	with open(path, "w") as text_file:
	    text_file.write(tableString)