# Reward Mechanisms for Dynamic Virtual Power Plants (DVPPs)

part of Master Thesis: Reward Mechanisms for DVPPs

## Run

1. Setup and activate python virtual environment and installl `reuirements.txt`

2. Load relevant data into the `data/` folder from [this url](https://drive.google.com/drive/folders/1qcyB8x9WR1VBKc3Dy5aBI7Hjsai5Dbuo?usp=sharing)
- `data/data_ffr_fcr_d_price_2024_25.csv`
- `data/data_ffr_fcr_procurement_2024_25.csv`
- `data/meteoblue/forecasted_pv_wind_profile.csv`
- `data/meteoblue/realized_pv_wind_profile.csv`

3. Create output folder in `pics/` by copy-pasting the folder `pics/template` and giving it a new name (e.g. `pics/my_results`)

4. Specify parameters in `dvpp_case_multiprocessing.py`
    - set variable `save_path = 'pics/my_results/'`
    - set devices (DERs) and their specifications
    - set other variables (start_date, K_errors, devices, ...) according to your study

5. Run `python dvpp_case_multiprocessing.py`

6. Output is available in `pics/my_results`
    - `expected_values_DPF.csv` are the forecasted values
    - `values_DPF.csv` are the realized values
    - `dvpp_info_DPF_FFR + FCR-D.csv` includes details on dc gain, bid submissions, ... for each time step
    - `bids_DPF_FFR + FCR-D.csv` includes all bids achieved in the $K$ scnearios

## Results

Plots for all coalitions can be saved locally. Value and reward mechanisms are visualized in `case_visualize.ipynb`
- specify the load path and run to view plots and reward allocations

## Requirements

install packages: see `requirements.txt`