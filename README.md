# XLAT

will contain my entry to https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition


## Installation

installed via conda on Windows:
- geopy 
- cartoy
- (numpy)
- (pandas)
- mayavi and vtk (plus dependencies) https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-conda-forge

keep XLAT folder in same folder as "training_1_round_1" and so on.

## mayavi install 
pip didn't work for some dependency reasons, so i jumped to the conda section of https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-conda-forge
it's slightly modified, so that it installs onto the base system

install:
 conda config --add channels conda-forge
 conda install vtk=8.2.0 # make sure this version is selected in the anaconda base environment
 conda install pyqt=4 # this was already done by something and then failed to resolve deps; this step was essentially skipped
 conda install mayavi

test:
 mayavi2
 python C:\Users\<username>\anaconda3\pkgs\mayavi-4.7.1-py37hf36c280_0\info\test\run_test.py

## Working principle

(TBA)
