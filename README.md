# Callahan-et-al_NCC_2021
ENSO-LongRunMIP-Replication

This repository provides processed data and code required to replicate the results in "Robust decrease in ENSO amplitude under long-term warming," Nature Climate Change, by Christopher Callahan, Chen Chen, Maria Rugenstein, Jonah Bloch-Johnson, Shuting Yang, and Liz Moyer.

The repository contains three folders: Data/, where processed data is stored, Figures/, where .pdf outputs from the scripts are stored, and Scripts/. The original LongRunMIP data is not provided in this repo due to size, but all of the processed data is available and each script should run using the data provided. Information about accessing the raw LongRunMIP data (variables required: tas, psl, thetao, pr), is available at longrunmip.org. There is also a script that requires observed sea surface temperature data from HadISST and ERSST. The processed data is provided in the repo so you don't need to download that raw data, but it is available online at https://catalogue.ceda.ac.uk/uuid/facafa2ae494597166217a9121a62d3c and https://www.ncdc.noaa.gov/data-access/marineocean-data/extended-reconstructed-sea-surface-temperature-ersst-v5.

The scripts are provided as Python code in Jupyter notebooks, which is how the analysis was performed (except for Calculate_Depth_Profile, which is a .py script that was run as a batch job on a supercomputing cluster). Conversion to regular Python scripts is straightforward if you have Jupyter installed, or feel free to email me at Christopher.W.Callahan.GR (at) dartmouth (dot) edu and I can convert them for you.

The full size of the repo is a little over 2 gigabytes.

Christopher Callahan

April 2021
