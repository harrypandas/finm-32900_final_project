# Finm-32900 Final Project: A puzzling replication of the puzzle of index option returns

## Team
* Ian Hammock <ihammock@uchicago.edu>
* Harrison Holt <hholt@uchicago.edu>
* Viren Desai <vdd@uchicago.edu>

## Final Project Details
* Constantinides, George M., Jens Carsten Jackwerth, and Alexi Savov. “The puzzle of index option returns.” Review of Asset Pricing Studies 3, no. 2 (2013): 229-257.

* Replicate Table 2 and Table B1. Table 3 or 4 may be difficult, but may be attempted as a stretch goal.

* Data used: OptionMetrics and Federal Reserve Board Daily Interest Rates

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
