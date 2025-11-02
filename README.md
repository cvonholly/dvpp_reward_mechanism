# Reward Mechanisms for Dynamic Virtual Power Plants (DVPPs)

part of Master Thesis: Reward Mechanisms for DVPPs

## Run

The files `dvpp_t_xxx.py` are the files to create and run a DVPP (see e.g. `python dvpp_t_v1.py` as example). create_io_dict is a set of devices, consisting of name: (Input-Output system, type, rating) where type can be lpf, bpf or hpf and rating is the nominal value in MW. Further options are the *Save path, Service to be followed, minimum capacity provided for service, number of scenarios, stochastic PV and Wind production, Power Factor type (default dynamic, other: static), to save picture, and real time adaptive function*

## Results

Plots for all coalitions can be saved locally. Value and reward mechanisms are visualized in `visualize.ipynb`

## Requirements

install packages: see `requirements.txt`