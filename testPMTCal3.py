from pmt import PMTCal
import numpy as np

# Define PMT and digitized responsivity and gain files
PMT = 'R9880U20'
root_dir = '/Users/erehm/src/pals'
rel_pmt_dir = 'pmt'
resp_csv = 'R9880U20_responsivity_digitized.csv'
gain_csv = 'R9880U20_gain_digitized.csv'
resistance_ohms = 50.0

# Create PMTcal instance from files
pmt_cal = PMTCal.from_files(root_dir, rel_pmt_dir, resp_csv, gain_csv, resistance_ohms)

# Test PMTcal
print(f"PMTCal parameters using {PMT}:")
control_voltage_V = 3.0 
supply_multiplier = 250
wavelength = 532
print(control_voltage_V, supply_multiplier, wavelength)

print(pmt_cal.resp_model(wavelength))
print(pmt_cal.gain_model(control_voltage_V * supply_multiplier))

# Convert a measured voltage ch532_V to optical power using PMTCal
ch532_V = -1
voltage = control_voltage_V * supply_multiplier
watts = pmt_cal.volts_to_watts(
    volts=np.array([ch532_V]),                 # numpy vector from Rigol CSV
    wavelength_nm=wavelength,      # for now (we’ll add 355 nm later)
    control_voltage_V=voltage      # send in high voltage only = control_voltage_V * supply_multiplier
)
print(watts)
