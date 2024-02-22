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




if __name__ == "__main__": 

	tableB1_2012 = pd.read_parquet(Path(OUTPUT_DIR)  / "tableB1_2012.parquet")

	tableB1_2020 = pd.read_parquet(Path(OUTPUT_DIR)  / "tableB1_2012.parquet")

	step_order = ['Starting', 'Level 1 filters']

	# Convert 'Step' to a categorical type with the specified order
	tableB1_2012['Step'] = pd.Categorical(tableB1_2012['Step'], categories=step_order, ordered=True)

	# Sort the DataFrame by the 'Step' column
	tableB1_2012.sort_values(by='Step', inplace=True)
	tableB1_2012 = tableB1_2012.replace({float('nan'): ''})
	# Convert the DataFrame to LaTeX
	latex_table = tableB1_2012[['Deleted', 'Remaining']].to_latex(index=True)

	# Print or save the LaTeX table
	print(latex_table)

	tableString = 
	path = OUTPUT_DIR / f'tableB1_2012.tex'
	with open(path, "w") as text_file:
	    text_file.write(latex_table)