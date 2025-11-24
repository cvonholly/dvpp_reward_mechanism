"""
DVPP consisting of Solar PV, Wind and BESS 
    with case study and uncertainty in Pv and Wind generation

Ilmar Hybdrid DVPP:
wind_cap = 216  # MW
solar_cap = 150  # MW
battery_cap, battery_energy = 25, 50  # MW, MWh
- use 10% of the capacity for the services

run scenarios with uncertainties
    1. stage: optimize grand coalition expected reward
    2. stage: real-time operation based on actual generation, compute value function

todo:
- BESS energy is limited -> make dependend on previous energy level

assumptions:
    - pv and wind generation are varying based on scenarios, forecast based on historical 2-month data for the hour
    - we bid the capacity based on the maximum possible reward in the forecasted generation
    - the rating is set as follows:
        if set_service_rating:
            sum_service_rating = set_service_rating[subset]
        else:
            sum_service_rating = sum([v[2] for v in subset_io_dict.values() if v[1]=='lpf']) if service!='FFR' else sum([v[2] for v in subset_io_dict.values()]) 
"""

import pandas as pd

from src.get_required_services import get_fcr_d, get_ffr_fcr
from run_case_dvpp_sim import run_case_dvpp_sim
from src.get_device_systems import get_pv_sys, get_wind_sys, get_bess_io_sys, get_bess_energy_sys


if __name__ == '__main__':
    # run simulation with default parameters
    rel = .1  # 10 percent of the capacity
    wind_cap = 216 * rel  # MW
    solar_cap = 150 * rel  # MW
    battery_cap, battery_energy = 25 * rel, 50 * rel * 3600 # MW, MWh
    def get_io_dict():
        return {'PV': (get_pv_sys(), 'lpf', solar_cap),
                'Wind': (get_wind_sys(), 'lpf', wind_cap),
                'BESS': (get_bess_energy_sys(e_max=4), 'hpf', battery_cap),
                }

    # normal scenario period: 2025-04-06 10:00:00
    # max wind error: 2024-12-19 15:00:00
    start_date, end_date = pd.to_datetime(['2025-04-07 00:00:00', '2025-04-07 23:00:00'])

    run_case_dvpp_sim(get_io_dict,
                        save_path='pics/v_case_0704',
                        services_input={'FFR-FCR': get_ffr_fcr()},
                        STATIC_PF=False,
                        K_errors=20,   # number of scenarios for the uncertainty
                        save_pics=False,
                        time_slots=(start_date, end_date),
                        save_dvpp_info=True,
                        hourly_average=True,
                        )