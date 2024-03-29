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
import glob

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
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace ./notebooks/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir=\"{output_dir}\" ./notebooks/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir=\"{output_dir}\" ./notebooks/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --to python ./notebooks/{notebook}.ipynb --output _{notebook}.py --output-dir  \"{build_dir}\"  "
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./notebooks/{notebook}.ipynb"
# fmt: on

# Check if .env file exists. If not, create it by copying from .env.example
env_file = ".env"
env_example_file = "env.example"

if not os.path.exists(env_file):
    shutil.copy(env_example_file, env_file)

def task_run_config(): 
    file_output = [
        "latexVar.tex",
        ]
    targets = [OUTPUT_DIR / file for file in file_output]

    actdict = {
    'actions': [
    "ipython ./src/config.py",
    "ipython ./src/create_latex_variables.py"
    ], 
    "targets": targets,
    'clean': True,
    }
    return actdict

# def task_create_latexVar(): 
#     file_dep = [
#     "./src/create_latex_variables.py", 
#     ]
#     file_output = [
#         "latexVar.tex",
#         ]
#     targets = [OUTPUT_DIR / file for file in file_output]

#     actdict = {
#     'actions': [
#     "ipython ./src/create_latex_variables.py"
#     ], 
#     "targets": targets,
#     "file_dep": file_dep,
#     'clean': True,
#     "verbosity": 2,
#     }
#     return actdict

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



def task_filter_appendix_B(): 


    file_dep = [
    "./src/filter_option_data_01.py", 
    "./src/filter_option_data_02.py", 
    "./src/filter_option_data_03.py", 
    "./src/filter_option_data_B.py",
    DATA_DIR / f"pulled/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet", 
    DATA_DIR / f"pulled/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet",
    ]

    
    
    targets = [
        OUTPUT_DIR / f"tableB1_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet",

        DATA_DIR / f"intermediate/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}_L1filter.parquet",

        DATA_DIR / f"intermediate/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}_L2filter.parquet",

        DATA_DIR / f"intermediate/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}_L3filterIVonly.parquet",
        
        DATA_DIR / f"intermediate/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}_L3filter.parquet",

        OUTPUT_DIR / f"tableB1_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet",

        DATA_DIR / f"intermediate/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}_L1filter.parquet",

        DATA_DIR / f"intermediate/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}_L2filter.parquet",

        DATA_DIR / f"intermediate/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}_L3filterIVonly.parquet",
        
        DATA_DIR / f"intermediate/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}_L3filter.parquet",
        
        OUTPUT_DIR / f"L3_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig1_post_L2filter.png",
        OUTPUT_DIR / f"L3_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig2_L2fitted_iv.png",
        OUTPUT_DIR / f"L3_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig3_IV_filter_only.png",
        OUTPUT_DIR / f"L3_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig3_IV_and_PCP.png",
        OUTPUT_DIR / f"L3_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig3_Final_vs_OptionMetrics.parquet",
        OUTPUT_DIR / f"L3_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig3_Final_vs_OptionMetrics.tex",
        OUTPUT_DIR / f"L3_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig3_nan_ivs_in_L2_data.tex",
        
        OUTPUT_DIR / f"L3_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig1_post_L2filter.png",
        OUTPUT_DIR / f"L3_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig2_L2fitted_iv.png",
        OUTPUT_DIR / f"L3_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig3_IV_filter_only.png",
        OUTPUT_DIR / f"L3_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig3_IV_and_PCP.png",
        OUTPUT_DIR / f"L3_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig3_Final_vs_OptionMetrics.parquet",
        OUTPUT_DIR / f"L3_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig3_Final_vs_OptionMetrics.tex",
        OUTPUT_DIR / f"L3_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig3_nan_ivs_in_L2_data.tex",
        ]
    
    
    actdict = {
    'actions': [
    "ipython ./src/filter_option_data_B.py"
    ], 
    "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    "verbosity": 2,
    }
    return actdict

def task_create_L2_plots():
    """Build plots for Level 2 filter steps and save as png files
    """
    file_dep = ["./src/create_l2_plots.py",
                "./src/filter_option_data_02.py",
                DATA_DIR / "intermediate" / f"data_{START_DATE_01[:7]}_{END_DATE_01[:7]}_L1filter.parquet",
                DATA_DIR / "intermediate" / f"data_{START_DATE_02[:7]}_{END_DATE_02[:7]}_L1filter.parquet"]
    
    file_output = [f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig1.png",
                   f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig2.png",
                   f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig3.png",
                   f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig4.png",
                   f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig5.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig1.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig2.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig3.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig4.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig5.png"]
    
    targets = [OUTPUT_DIR / file for file in file_output]

    actdict = {
                    "actions": [
                        "ipython ./src/create_l2_plots.py",
                    ],
                    "targets": targets,
                    "file_dep": file_dep,
                    "clean": True,
                }
    return actdict 

def task_create_L1_plots():
    """Plots for Level 1 filter steps
    """
    file_dep = ["./src/create_l1_plots.py",
                "./src/filter_option_data_01.py",
                DATA_DIR / f"pulled/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet", 
                DATA_DIR / f"pulled/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet",
                DATA_DIR / "intermediate" / f"data_{START_DATE_01[:7]}_{END_DATE_01[:7]}_L3filter.parquet",
                DATA_DIR / "intermediate" / f"data_{START_DATE_02[:7]}_{END_DATE_02[:7]}_L3filter.parquet",
                ]
    
    file_output = [f"L1_noVol_noInt_{START_DATE_01[:7]}_{END_DATE_01[:7]}.tex",
                    f"L1_noVol_noInt_{START_DATE_02[:7]}_{END_DATE_02[:7]}.tex",
               ]
    
    targets = [OUTPUT_DIR / file for file in file_output]

    actdict = {
                    "actions": [
                        "ipython ./src/create_l1_plots.py",
                    ],
                    "targets": targets,
                    "file_dep": file_dep,
                    "clean": True,
                }
    return actdict 

def task_create_TableB1(): 

    #add file dep on data being made 
    file_dep = [ "./src/create_table_B1_.py", 
    OUTPUT_DIR / f"tableB1_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet",
    OUTPUT_DIR / f"tableB1_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet",

    ]
    file_output = [
        "tableB1.tex",
        ]
    targets = [OUTPUT_DIR / file for file in file_output]

    actdict = {
    'actions': [
    "ipython ./src/create_table_B1_.py",
  
    ], 
    "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    }

    return actdict


def task_Table2_Analysis(): 
    file_dep = [
    "./src/filter_option_data_04.py", 
    DATA_DIR / f"intermediate/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}_L3filter.parquet",
    DATA_DIR / f"intermediate/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}_L3filter.parquet",
    ]
    targets = [
        OUTPUT_DIR /f"table2_all_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet",
        OUTPUT_DIR /f"table2_month_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet", 
        OUTPUT_DIR /f"table2_all_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet",
        OUTPUT_DIR /f"table2_month_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet", 
        ]
    actdict = {
    'actions': [
    "ipython ./src/filter_option_data_04.py"
    ], 
    "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    "verbosity": 2,
    }
    return actdict


def task_create_Table2(): 
    file_dep = [
    "./src/create_table_2_.py", 
        OUTPUT_DIR /f"table2_all_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet",
        OUTPUT_DIR /f"table2_month_{START_DATE_01[:7]}_{END_DATE_01[:7]}.parquet", 
        OUTPUT_DIR /f"table2_all_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet",
        OUTPUT_DIR /f"table2_month_{START_DATE_02[:7]}_{END_DATE_02[:7]}.parquet", 
    ]
    file_output = [
        "table2.tex",
        ]
    targets = [OUTPUT_DIR / file for file in file_output]

    actdict = {
    'actions': [
    "ipython ./src/create_table_2_.py"
    ], 
    "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    "verbosity": 2,
    }
    return actdict


def task_create_Table2_days(): 
    file_dep = [
        "./src/create_T2_plots.py", 
        DATA_DIR / f"intermediate/data_{START_DATE_01[:7]}_{END_DATE_01[:7]}_L3filter.parquet",
        DATA_DIR / f"intermediate/data_{START_DATE_02[:7]}_{END_DATE_02[:7]}_L3filter.parquet",
        ]
    file_output = ["table2.tex" ]
    targets = [OUTPUT_DIR / f'T2_days.tex']

    actdict = {
    'actions': [
    "ipython ./src/create_T2_plots.py"
    ], 
    "targets": targets,
    "file_dep": file_dep,
    'clean': True,
    "verbosity": 2,
    }
    return actdict


def task_compile_latex_docs():
    file_dep = [
        "./reports/final_report.tex",
    	"./src/create_table_B1_.py",
    	"./src/create_table_2_.py", 
    	"./output/tableB1.tex",
        "./output/table2.tex",
    ]

    file_dep2 = [f"L1_noVol_noInt_{START_DATE_01[:7]}_{END_DATE_01[:7]}.tex",
                    f"L1_noVol_noInt_{START_DATE_02[:7]}_{END_DATE_02[:7]}.tex",
                    f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig1.png",
                   f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig2.png",
                   f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig3.png",
                   f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig4.png",
                   f"L2_{START_DATE_01[:7]}_{END_DATE_01[:7]}_fig5.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig1.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig2.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig3.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig4.png",
                   f"L2_{START_DATE_02[:7]}_{END_DATE_02[:7]}_fig5.png",
                   'T2_days.tex',
                   ]
    
    dep2 = [OUTPUT_DIR / file for file in file_dep2]

    file_dep = file_dep + dep2
    
    file_output = [
        "./reports/final_report.pdf",
       # "./reports/slides_example.pdf",
    ]
    targets = [file for file in file_output]

    return {
        "actions": [

            # "latexmk -xelatex -cd ./reports/sample_table.tex",  # Compile
            # "latexmk -xelatex -c -cd ./reports/sample_table.tex",  # Clean
            "latexmk -xelatex -c -cd ./reports/final_report.tex",   # Clean
            "latexmk -xelatex -cd ./reports/final_report.tex",  # Compile
            
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_clean_latex_docs():
    file_dep = [
        "./reports/final_report.tex",
        "./reports/final_report.pdf"
    ,
    ]

    return {
        "actions": [

            "latexmk -xelatex -c -cd ./reports/final_report.tex",   # Clean
        ],
        "file_dep": file_dep,
        "clean": True,
    }


def task_convert_notebooks_to_scripts():
    """Preps the notebooks for presentation format.
    Execute notebooks with summary stats and plots and remove metadata.
    """
    build_dir = Path(OUTPUT_DIR)
    build_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [
        "01_Level01_Filters.ipynb",
        "02_Level02_Filters.ipynb",
        "03_Level03_Filters.ipynb",
    ]
    file_dep = [Path("./notebooks") / file for file in notebooks]
    stems = [notebook.split(".")[0] for notebook in notebooks]
    targets = [build_dir / f"_{stem}.py" for stem in stems]

    actions = [
        # *[jupyter_execute_notebook(notebook) for notebook in notebooks_to_run],
        # *[jupyter_to_html(notebook) for notebook in notebooks_to_run],
        *[jupyter_clear_output(notebook) for notebook in stems],
        *[jupyter_to_python(notebook, build_dir) for notebook in stems],
    ]
    return {
        "actions": actions,
        "targets": targets,
        "task_dep": [],
        "file_dep": file_dep,
        "clean": True,
    }


def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks with summary stats and plots and remove metadata.
    """
    notebooks_to_run_as_md = [
        "01_Level01_Filters.ipynb",
        "02_Level02_Filters.ipynb",
        "03_Level03_Filters.ipynb",
    ]
    stems = [notebook.split(".")[0] for notebook in notebooks_to_run_as_md]

    file_dep = [
        # 'load_other_data.py',
        *[Path(OUTPUT_DIR) / f"_{stem}.py" for stem in stems],
    ]

    targets = [
        ## Notebooks converted to HTML
        *[OUTPUT_DIR / f"{stem}.html" for stem in stems],
    ]

    actions = [
        *[jupyter_execute_notebook(notebook) for notebook in stems],
        *[jupyter_to_html(notebook) for notebook in stems],
        *[jupyter_clear_output(notebook) for notebook in stems],
        # *[jupyter_to_python(notebook, build_dir) for notebook in notebooks_to_run],
    ]
    return {
        "actions": actions,
        "targets": targets,
        "task_dep": [],
        "file_dep": file_dep,
        "clean": True,
    }