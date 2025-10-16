"""
DVPP consisting of Solar PV, Wind and Battery 
    with uncertainty in Pv and Wind generation

run scenarios with uncertainties
    1. stage: coalition forming based on forecasted generation
    2. stage: real-time operation based on actual generation

assumptions:
    - bess operates 1/2 in spot market, thus capacity is varying based on avialable energy. thus it likely cannot provide the full FCR service
    - pv and wind generation are varying based on scenarios, forecast based on historical 2-month data for the hour
    - we bid the capacity based on the maximum possible reward in the forecasted generation
    - the time slots are chosen by randomly but with weighting the historical procurement\
    - the rating is set as follows:
        if set_service_rating:
            sum_service_rating = set_service_rating[subset]
        else:
            sum_service_rating = sum([v[2] for v in subset_io_dict.values() if v[1]=='lpf']) if service!='FFR' else sum([v[2] for v in subset_io_dict.values()]) 
"""

import pandas as pd

# import procduction data
from src.get_device_systems import get_pv_sys, get_wind_sys, get_bess_io_sys, get_bess_energy_sys

from run_bcd_dvpp_sim import run_bcd_dvpp_sim


if __name__ == '__main__':
    # run simulation with default parameters
    def get_io_dict():
        return {'PV': (get_pv_sys(), 'lpf', 1),
                'Wind': (get_wind_sys(), 'lpf', 1),
                'BESS': (get_bess_energy_sys(e_max=4), 'hpf', 1),
                }

    # start_date, end_date = pd.to_datetime(['2025-04-06', '2025-04-13'])

    run_bcd_dvpp_sim(get_io_dict,
                        save_path='pics/v_bcg',
                        STATIC_PF=False,
                        Sx=5,
                        # time_slots=(start_date, end_date),
                        include_battery_uncertainty=False,
                        save_dvpp_info=True,
                        hourly_average=False
                        )