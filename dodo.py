"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based
"""
import sys
sys.path.insert(1, './src/')

import os
import config
from pathlib import Path
from doit.tools import run_once
import shutil



from load_option_data_01 import run_load_all_optm_data 


OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir=\"{output_dir}\" ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir=\"{output_dir}\" ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir  \"{build_dir}\"  "
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
# fmt: on




# Check if .env file exists. If not, create it by copying from .env.example
env_file = ".env"
env_example_file = "env.example"

if not os.path.exists(env_file):
    shutil.copy(env_example_file, env_file)


def task_run_config(): 
    actdict = {
    'actions': [
    "ipython ./src/config.py"
    ], 
    'clean': True,
    }
    return actdict

# def task_load_and_save_data(): 


#     file_dep = [ "./src/load_option_data_01.py"]
#     file_output = [
#         "pulled/data_1996-01_2012-01.parquet",
#         ]
#     targets = [DATA_DIR  / file for file in file_output]
#     actdict = {
#     'actions': [
#     "ipython ./src/load_option_data_01.py"
#     ], 
#     "targets": targets,
#     "file_dep": file_dep,
#     'clean': True,
#     "verbosity": 2, # Print everything immediately. This is important in
#         # case WRDS asks for credentials.
#     }
#     return actdict

def task_load_and_save_data_01(): 


    file_dep = [ "./src/load_option_data_01.py"]
    file_output = [
        f"pulled/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet",
        ]
    targets = [DATA_DIR  / file for file in file_output]
    actdict = {
    'actions': [
    (run_load_all_optm_data, (f"{DATA_DIR}",
											f"{WRDS_USERNAME}", 
											START_DATE_01,
											END_DATE_01))
    ], 
    "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    "verbosity": 2, # Print everything immediately. This is important in
        # case WRDS asks for credentials.
    }
    return actdict


def task_load_and_save_data_02(): 


    file_dep = [ "./src/load_option_data_01.py"]
    file_output = [
        f"pulled/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet",
        ]
    targets = [DATA_DIR  / file for file in file_output]
    actdict = {
    'actions': [
    (run_load_all_optm_data, (f"{DATA_DIR}",
											f"{WRDS_USERNAME}", 
											START_DATE_02,
											END_DATE_02))
    ], 
    "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    "verbosity": 2, # Print everything immediately. This is important in
        # case WRDS asks for credentials.
    }
    return actdict



def task_filter_appendix_B_01(): 


    file_dep = [ "./src/filter_option_data_01.py", DATA_DIR / f"pulled/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet"]
    targets = [
        OUTPUT_DIR / "tableB1.tex", DATA_DIR / "data_1996_2012_appendixB.parquet",
        ]
    actdict = {
    'actions': [
    "ipython ./src/filter_option_data_01.py"
    ], 
   # "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    }
    return actdict

def task_run_placeholderTables(): 

    #add file dep on data being made 
    file_dep = [ "./src/pandas_to_latex_demo.py"]
    file_output = [
        "pandas_to_latex_simple_table1.tex",
        ]
    targets = [OUTPUT_DIR / file for file in file_output]

    actdict = {
    'actions': [
    "ipython ./src/pandas_to_latex_demo.py",
  
    ], 
    "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    }

    return actdict

# def task_compile_latex_docs():
#     """Example plots"""
#     file_dep = [
#         # "./reports/report_example.tex",
#         # "./reports/slides_example.tex",
#        # "./reports/slides_example.tex",
#         "./output/pandas_to_latex_simple_table1.tex",
#     ]
#     file_output = [
#         "./reports/final_report.pdf",
#        # "./reports/slides_example.pdf",
#     ]
#     targets = [file for file in file_output]

#     return {
#         "actions": [

#             "latexmk -xelatex -cd ./reports/final_report.tex",  # Compile
#             "latexmk -xelatex -c -cd ./reports/final_report.tex",  # Clean
#             # "latexmk -xelatex -cd ./reports/report_example.tex",  # Compile
#             # "latexmk -xelatex -c -cd ./reports/report_example.tex",  # Clean
#             # "latexmk -xelatex -cd ./reports/slides_example.tex",  # Compile
#             # "latexmk -xelatex -c -cd ./reports/slides_example.tex",  # Clean
#             # # "latexmk -CA -cd ../reports/",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }




# def task_pull_CRSP_Compustat():
#     """Pull CRSP/Compustat data from WRDS and save to disk
#     """
#     file_dep = [
#         "./src/config.py", 
#         "./src/load_CRSP_stock.py",
#         "./src/load_CRSP_Compustat.py",
#         ]
#     targets = [
#         Path(DATA_DIR) / "pulled" / file for file in 
#         [
#             ## src/load_CRSP_stock.py
#             "CRSP_MSF_INDEX_INPUTS.parquet", 
#             "CRSP_MSIX.parquet", 
#             ## src/load_CRSP_Compustat.py
#             "Compustat.parquet",
#             "CRSP_stock_ciz.parquet",
#             "CRSP_Comp_Link_Table.parquet",
#             "FF_FACTORS.parquet",
#         ]
#     ]

#     return {
#         "actions": [
#             "ipython src/config.py",
#             "ipython src/load_CRSP_stock.py",
#             "ipython src/load_CRSP_Compustat.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#         "verbosity": 2, # Print everything immediately. This is important in
#         # case WRDS asks for credentials.
#     }


# def task_calc_Fama_French_1993_factors():
#     """Calculate Factors for Fama-French 1993 model
#     """
#     file_dep = [
#         "./src/calc_Fama_French_1993_factors.py",
#         "./src/misc_tools.py",
#         ]
#     targets = [
#         *[Path(DATA_DIR) / "pulled" / file for file in 
#         [
#             ## src/calc_Fama_French_1993_factors.py
#             "FF_1993_vwret.parquet",
#             "FF_1993_vwret_n.parquet",
#             "FF_1993_factors.parquet",
#             "FF_1993_nfirms.parquet",
#         ]],
#         OUTPUT_DIR / "FF_1993_Comparison.png",
#     ]

#     return {
#         "actions": [
#             "ipython src/calc_Fama_French_1993_factors.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }


# def task_convert_notebooks_to_scripts():
#     """Preps the notebooks for presentation format.
#     Execute notebooks with summary stats and plots and remove metadata.
#     """
#     build_dir = Path(OUTPUT_DIR)
#     build_dir.mkdir(parents=True, exist_ok=True)

#     notebooks = [
#         "01_wrds_python_package.ipynb",
#         "02_CRSP_market_index.ipynb",
#         "03_Fama_French_1993.ipynb",
#     ]
#     file_dep = [Path("./src") / file for file in notebooks]
#     stems = [notebook.split(".")[0] for notebook in notebooks]
#     targets = [build_dir / f"_{stem}.py" for stem in stems]

#     actions = [
#         # *[jupyter_execute_notebook(notebook) for notebook in notebooks_to_run],
#         # *[jupyter_to_html(notebook) for notebook in notebooks_to_run],
#         *[jupyter_clear_output(notebook) for notebook in stems],
#         *[jupyter_to_python(notebook, build_dir) for notebook in stems],
#     ]
#     return {
#         "actions": actions,
#         "targets": targets,
#         "task_dep": [],
#         "file_dep": file_dep,
#         "clean": True,
#     }


# def task_run_notebooks():
#     """Preps the notebooks for presentation format.
#     Execute notebooks with summary stats and plots and remove metadata.
#     """
#     notebooks_to_run_as_md = [
#         "01_wrds_python_package.ipynb",
#         "02_CRSP_market_index.ipynb",
#         "03_Fama_French_1993.ipynb",
#     ]
#     stems = [notebook.split(".")[0] for notebook in notebooks_to_run_as_md]

#     file_dep = [
#         # 'load_other_data.py',
#         *[Path(OUTPUT_DIR) / f"_{stem}.py" for stem in stems],
#     ]

#     targets = [
#         ## Notebooks converted to HTML
#         *[OUTPUT_DIR / f"{stem}.html" for stem in stems],
#     ]

#     actions = [
#         *[jupyter_execute_notebook(notebook) for notebook in stems],
#         *[jupyter_to_html(notebook) for notebook in stems],
#         *[jupyter_clear_output(notebook) for notebook in stems],
#         # *[jupyter_to_python(notebook, build_dir) for notebook in notebooks_to_run],
#     ]
#     return {
#         "actions": actions,
#         "targets": targets,
#         "task_dep": [],
#         "file_dep": file_dep,
#         "clean": True,
#     }

