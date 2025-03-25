# MT-Data-Analysis-GUI

A comprehensive software tool for analyzing data from Magnetic Tweezer (MT) experiments in single-molecule biophysics research.

## Overview

MT-Data-Analysis-GUI provides a user-friendly GUI for processing and analyzing data collected from magnetic tweezer experiments. It allows researchers to perform various analyses on force-extension characteristics, kinetics, and more.

## Features

- **Force-Extension Analysis**: Analyze force-extension relationships with WLC and eWLC models
- **Force Calibration**: Calibrate force measurements for accurate experimental results
- **Kinetics Analysis**: Analyze molecular dynamics using Hidden Markov Models (HMM)
- **Exponential Calculator**: Fit data to exponential equations for parameter extraction

## Installation

1. Clone this repository
2. Install required packages
3. Run the main application

## Usage

1. Launch the application
2. Select the desired analysis function
3. Load your TDMS data file using the "Select Files" button
4. Click "Run" to execute the selected analysis

## Data Format

The software is designed to work with TDMS files containing magnetic tweezer experimental data. The files should include:
- Time series data
- Extension measurements
- Force data
- Magnet position information

## Output

Results can be exported to Excel files (.xlsx) for further analysis or visualization.

## Development

This tool was developed with Python using:
- PySide6 (Qt) for the GUI
- NumPy and SciPy for numerical analysis
- Matplotlib for visualization
- hmmlearn for Hidden Markov Model implementation

## Author

Ye Yang (叶杨)  
School of Medicine, Zhejiang University (浙江大学医学院)
