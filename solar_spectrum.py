import pandas as pd
from scipy.interpolate import interp1d

def load_solar_spectrum(csv_path):
    data = pd.read_csv(csv_path)
    wavelengths = data['wavelength'].values  # in nm
    intensity = data['intensity'].values    # in W/m^2/nm
    interp_func = interp1d(wavelengths, intensity, kind='linear', bounds_error=False, fill_value=0)
    return interp_func 