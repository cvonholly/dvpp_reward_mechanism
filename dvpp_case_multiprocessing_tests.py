import pandas as pd
import multiprocessing
import os
from functools import partial
import matplotlib.pyplot as plt

# Import your custom modules
from src.get_required_services import get_fcr_d, get_ffr_fcr, get_ffr_fcr_d
from run_case_dvpp_sim_multiprocessing import run_case_dvpp_sim
from src.get_device_systems import get_pv_sys, get_wind_sys, get_bess_io_sys, get_bess_energy_sys

# --- Configuration Constants (Must be Global) ---
start_date = pd.to_datetime('2025-04-12 10:00:00')
end_date = pd.to_datetime('2025-04-12 11:59:59')
SAVE_PICS = True   # default: False
NUMB_WORKERS = 2 # 21  # number of parallel processes
N_HOURS_TOTAL = (end_date - start_date).days * 24 + (end_date - start_date).seconds // 3600 + 1
HOUR_CHUNKS = N_HOURS_TOTAL // NUMB_WORKERS   # number of hours per worker
save_path = 'pics/v_TESTING/'
input_service = {'FFR + FCR-D': get_ffr_fcr_d()}   # dict of services to provide
K_errors = 5      # default: 25  # number of scenarios for the uncertainty
HPF_DC_FACTOR = .25
REL = .1           # scaling factor for device ratings; i.e. 10% of real Ilmar power plant
WIND_CAP = 216 * REL
SOLAR_CAP = 150 * REL
# BATTERY_CAP = 25 * REL   # real Ilmar power plant params
BATTERY_CAP = 150 * REL   # scaled up battery for better performance
BATTERY_ENERGY = BATTERY_CAP * HPF_DC_FACTOR * 4
debug_grand_coalition = True   # save pictures of grand coalition failing to meet service

FONT_SIZE = 18
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX to write text
    "font.family": "serif",
    "font.serif": ["Times"], # Use a serif font (LaTeX will substitute the right one)
    
    # 2. MATCH YOUR DOCUMENT PREAMBLE HERE
    # Use the same packages you use in your main document.
    # 'newtxtext,newtxmath' is the modern choice.
    # Use 'mathptmx' if your paper uses that instead.
    "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}", 
    
    # Optional: Match the font size to your document (usually 10pt for IEEE)
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
})


# save these configuration constants into save_path for recrod keeping
config_constants = {
    'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
    'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S'),
    'K_errors': K_errors,
    'REL': REL,
    'WIND_CAP': WIND_CAP,
    'SOLAR_CAP': SOLAR_CAP,
    'BATTERY_CAP': BATTERY_CAP,
    'BATTERY_ENERGY': BATTERY_ENERGY,
    'HPF_DC_FACTOR': HPF_DC_FACTOR,
    'debug_grand_coalition': debug_grand_coalition
}

def get_io_dict():
    """Helper function to define system I/O (Global for pickling)"""
    return {
        'PV': (get_pv_sys(), 'lpf', SOLAR_CAP),
        'Wind': (get_wind_sys(), 'lpf', WIND_CAP),
        'BESS': (get_bess_energy_sys(e_max=BATTERY_ENERGY), 'hpf', BATTERY_CAP),
    }

def run_simulation_chunk(time_chunk, common_kwargs):
    """Worker function for multiprocessing."""
    chunk_start, chunk_end = time_chunk
    print(f"Process started: {chunk_start} -> {chunk_end}")
    
    try:
        # Run simulation with 'save_files_internally=False'
        # We catch the return values
        realized, expected, dvpp_infos, bids = run_case_dvpp_sim(
            get_io_dict,
            time_slots=(chunk_start, chunk_end),
            **common_kwargs
        )
        return (realized, expected, dvpp_infos, bids)
    except Exception as e:
        print(f"!!! Error in chunk {chunk_start}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # save config constants
    with open(os.path.join(save_path, 'config_constants.txt'), 'w') as f:
        for key, value in config_constants.items():
            f.write(f"{key}: {value}\n")

    # --- 1. Generate Time Chunks (12h periods) ---
    chunks = []
    current_start = start_date
    while current_start < end_date:
        current_end = current_start + pd.Timedelta(hours=HOUR_CHUNKS) - pd.Timedelta(seconds=1)
        if current_end > end_date:
            current_end = end_date
        
        chunks.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(seconds=1)

    print(f"Created {len(chunks)} time chunks.")

    # --- 2. Define Static Arguments ---
    common_kwargs = {
        'save_path': save_path,
        'services_input': input_service,
        'STATIC_PF': False,
        'K_errors': K_errors,
        'save_pics': SAVE_PICS,
        'set_special_ratings': {
            'FFR + FCR-D': {('BESS',): HPF_DC_FACTOR}
        },
        'save_dvpp_info': True,
        'hourly_average': True,
        'allow_sub_coalitions': True,
        'HPF_DC_factor': HPF_DC_FACTOR,
        'save_files_internally': False, # Important: Don't write inside workers
        'debug_grand_coalition': debug_grand_coalition
    }

    # --- 3. Run Multiprocessing ---
    worker = partial(run_simulation_chunk, common_kwargs=common_kwargs)
    
    # Use fewer processes than chunks to avoid overloading memory if RAM is tight
    num_processes = min(len(chunks), multiprocessing.cpu_count() - 1)
    
    print(f"Starting Pool with {num_processes} workers...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker, chunks)

    # --- 4. Aggregate Results ---
    print("Aggregating results...")
    
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("No valid results to aggregate.")
    else:
        # Initialize containers for aggregation
        agg_realized = {}
        agg_expected = {}
        agg_dvpp_infos = {} 
        agg_bids = {}

        # Unpack first result to get service keys structure
        first_dvpp_infos = valid_results[0][2]
        service_keys = first_dvpp_infos.keys()
        
        # Initialize list containers for DataFrames
        dvpp_info_lists = {s: [] for s in service_keys}
        bids_lists = {s: [] for s in service_keys}

        for res in valid_results:
            (r_realized, r_expected, r_dvpp_infos, r_bids) = res
            
            # Update Dictionaries
            agg_realized.update(r_realized)
            agg_expected.update(r_expected)
            
            # Collect DataFrames for later concatenation
            for s in service_keys:
                if s in r_dvpp_infos:
                    dvpp_info_lists[s].append(r_dvpp_infos[s])
                if s in r_bids:
                    bids_lists[s].append(r_bids[s])

        # Concatenate DataFrames
        for s in service_keys:
            if dvpp_info_lists[s]:
                agg_dvpp_infos[s] = pd.concat(dvpp_info_lists[s]).sort_index()
            if bids_lists[s]:
                agg_bids[s] = pd.concat(bids_lists[s]).sort_index()

        # --- 5. Save Final Aggregated Files ---
        # Logic copied from original run_case_dvpp_sim save block
        pf_name = 'DPF' # derived from logic in original script (STATIC_PF=False)
        
        print("Saving aggregated files...")
        
        df = pd.DataFrame.from_dict(agg_realized, orient='index')
        df.to_csv(f'{save_path}/values_{pf_name}.csv', float_format='%.5f')

        df_forecast = pd.DataFrame.from_dict(agg_expected, orient='index')
        df_forecast.to_csv(f'{save_path}/expected_values_{pf_name}.csv', float_format='%.5f')
        
        for s, infos in agg_dvpp_infos.items():
            infos.to_csv(f'{save_path}/dvpp_info_{pf_name}_{s}.csv', index=True, float_format='%.5f')
            # save bids df to csv
            if s in agg_bids:
                agg_bids[s].to_csv(f'{save_path}/bids_{pf_name}_{s}.csv', index=True, float_format='%.5f')

    print("Multiprocessing complete.")