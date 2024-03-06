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


"""
This script will take the dates from the .env file and import them into latex. 

This is helpful for importing plots and tables, as their names rely upon what time range one chooses. 
"""

def makeVar(s1,e1,s2,e2): 
	string = r' \newcommand{\STARTONE}{' + f'{s1[:7]}' + r'}' + '\n'

	string = string + r' \newcommand{\ENDONE}{' + f'{e1[:7]}' + r'}' + '\n'

	string = string + r' \newcommand{\STARTTWO}{' + f'{s2[:7]}' + r'}' + '\n'
	string = string + r' \newcommand{\ENDTWO}{' + f'{e2[:7]}' + r'}' + '\n'
	path = OUTPUT_DIR / f'latexVar.tex'
	with open(path, "w") as text_file:
	    text_file.write(string)
if __name__ == "__main__": 

	makeVar(START_DATE_01, END_DATE_01, START_DATE_02, END_DATE_02)

