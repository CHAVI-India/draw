@echo off
REM Set the path to your Anaconda or Miniconda installation
set CONDA_PATH=D:\conda

REM Set the name of your Conda environment
set ENV_NAME=a9t

REM Activate the Conda environment
call %CONDA_PATH%\Scripts\activate %ENV_NAME%

REM Run the Python script
python D:\totalsegmentator\a9t_v2\kgp.segmentation\main.py start-pipeline
