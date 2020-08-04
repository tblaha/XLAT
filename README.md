# XLAT

This branch contains my entry to Round 1 of the A/C localization challenge https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition


## Technical Documentation

Either: See submodule XLAT-docs for LATEX files. 
Or: Sharelatex can be viewed @ https://www.overleaf.com/read/fbbzchxgsxwy


## Prerequisites

installed via conda on Windows:
- geopy
- cartopy
- numpy
- scipy
- pandas


## Directory Structure

<pre>
D:.
+---XLAT
|   |   .gitignore
|   |   LICENSE
|   |   main.py
|   |   README.md
|   |
|   \---MLATlib
|       |   __init__.py
|       |  filt.py
|       ...
|
\---Data
    +---competition_evaluation
    |       round1_competition.csv
    |       round1_sample_empty.csv
    |       sensors.csv
    |       
    \---training_dataset
        +---training_1_category_1
        |       OpenSky_TERMS.txt
        |       sensors.csv
        |       training_1_category_1.csv
        |       
        +---training_1_category_1_result
        |       OpenSky_TERMS.txt
        |       sensors.csv
        |       training_1_category_1_result.csv
        |       
        +---training_2_category_1
        |
        ...
        |
        \---training_7_category_1_result
                OpenSky_TERMS.txt
                sensors.csv
                training_7_category_1_result.csv
                
</pre>
