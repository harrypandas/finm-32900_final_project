# Finm-32900 Final Project: A puzzling replication of the puzzle of index option returns

## Team
* Ian Hammock <ihammock@uchicago.edu>
* Harrison Holt <hholt@uchicago.edu>
* Viren Desai <vdd@uchicago.edu>

## Final Project Details
* Constantinides, George M., Jens Carsten Jackwerth, and Alexi Savov. “The puzzle of index option returns.” Review of Asset Pricing Studies 3, no. 2 (2013): 229-257.

* Replicate Table 2 and Table B1. Table 3 or 4 may be difficult, but may be attempted as a stretch goal.

* Data used: OptionMetrics and Federal Reserve Board Daily Interest Rates

## Getting Started 
* Be sure to have a WRDS account: [Wharton Research Data Services](https://wrds-www.wharton.upenn.edu/)
* Clone this repository using your favorite flavor of Git. Set this as your working directory: 
	* [https://github.com/harrypandas/finm-32900_final_project.git](https://github.com/harrypandas/finm-32900_final_project.git)
	```
	cd User/*/*/finm-32900_final_project
	```
	
* Create a python environment and install packages using pip:
	```
	conda create --name puzzle python==3.12

	conda activate puzzle

	pip install -r requirements.txt 
	```
* Optional Action: Set up .env file 
	* Copy env.example to .env
	* In this file you may set the directories that the data and output files will be saved to. Please note, that the final latex report will be saved under ./reports.
		```
		DATA_DIR="data"
		OUTPUT_DIR="output"
		```
	* In .env you may provide your WRDS user name, if you do not do this, you will be prompted for username and password twice as the data is loaded. 
		```
		WRDS_USERNAME=""
		```
	* In this file you may also set the two date ranges you'd like to see. The defaults are:
		```
		START_DATE_01="1996-01-01"
		END_DATE_01="2012-01-31"
		START_DATE_02="2012-02-01"
		END_DATE_02="2019-12-31"
		```
	  
* In the */finm-32900_final_project working directory run: 
	```
	doit
	```

## Task List
#### Replicate Table B1
* Create functions to apply the level one, two, and three filters used to create Table B1
* Apply the functions on the raw OptionMetrics data to output the table as part of the LaTex document
* Write jupyter notebook(s) to walkthrough an analysis of the filters
* Provide table of our own summary statistcs and charts typeset on LaTex

#### Replicate Table 2
* Using the final dataset after applying the filters used to create Table B1, replicate the summary statistics for the calls and put portfolios (Table 2)
* Create functions to claculate the Found, Missing, and Expired observations as outlined in the paper
* Output the replicated Table 2 as part of the LaTex document
* Provide table of our own summary statistcs and charts typeset on LaTex

#### Unit Tests
* Design purposeful unit tests to ensure the steps to replicate the tables are producing the expected outputs

#### Other Tasks
* Automate project from end-to-end using PyDoit
* Ensure project meets GitHub requirements
