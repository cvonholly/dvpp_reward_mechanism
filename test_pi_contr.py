# Test script to verify PI controller saturation behavior

import control as ct
import numpy as np
import matplotlib.pyplot as plt

from src.get_controllers import get_pi_controller

def test_pi_saturation_individual_vs_dvpp(pi_params, IO_dict):
    """
    Compare PI controller behavior between individual and DVPP systems
    """
    
    # Test with a simple step input
    t = np.linspace(0, 10, 1000)
    step_input = np.ones_like(t)
    
    device_names = list(IO_dict.keys())
    
    print("=== TESTING PI CONTROLLER SATURATION ===")
    
    for name in device_names:
        print(f"\nTesting device: {name}")
        
        # Individual PI controller
        PI_individual = get_pi_controller(params=pi_params[name])
        print(f"Individual PI saturation limits: {pi_params[name].get('saturation_limits', 'None')}")
        
        # Test with different error magnitudes
        for error_magnitude in [0.1, 1.0, 2.0, 5.0]:
            error_signal = error_magnitude * step_input
            
            # Simulate PI response
            try:
                tout, yout = ct.forced_response(PI_individual, t, error_signal)
                max_output = np.max(np.abs(yout))
                print(f"  Error {error_magnitude}: Max PI output = {max_output:.2f}")
                
                # Check if saturation is working
                sat_limits = pi_params[name].get('saturation_limits', (-np.inf, np.inf))
                if max_output > abs(sat_limits[1]) + 0.01:  # small tolerance
                    print(f"    WARNING: PI output ({max_output:.2f}) exceeds saturation limit ({sat_limits[1]})!")
                    
            except Exception as e:
                print(f"  Error simulating with magnitude {error_magnitude}: {e}")
    
    # Compare with DVPP behavior
    print("\n=== DVPP PI CONTROLLER COMPARISON ===")
    
    # Create a simple 2-device DVPP for testing
    if len(device_names) >= 2:
        name1, name2 = device_names[:2]
        
        # Check if PI controllers in DVPP have same saturation
        PI1_dvpp = get_pi_controller(params=pi_params[name1])
        PI2_dvpp = get_pi_controller(params=pi_params[name2])
        
        print(f"DVPP PI1 ({name1}) params: {pi_params[name1]}")
        print(f"DVPP PI2 ({name2}) params: {pi_params[name2]}")
        
        # Test if they respond the same as individual controllers
        for error_mag in [1.0, 2.0]:
            error_signal = error_mag * step_input
            
            try:
                # Individual
                PI_ind1 = get_pi_controller(params=pi_params[name1])
                _, y_ind1 = ct.forced_response(PI_ind1, t, error_signal)
                
                # DVPP
                _, y_dvpp1 = ct.forced_response(PI1_dvpp, t, error_signal)
                
                print(f"Error {error_mag} - {name1}:")
                print(f"  Individual max: {np.max(np.abs(y_ind1)):.2f}")
                print(f"  DVPP max: {np.max(np.abs(y_dvpp1)):.2f}")
                print(f"  Difference: {np.max(np.abs(y_ind1 - y_dvpp1)):.6f}")
                
            except Exception as e:
                print(f"Error in comparison: {e}")

# Run this test:
# test_pi_saturation_individual_vs_dvpp(pi_params, IO_dict)