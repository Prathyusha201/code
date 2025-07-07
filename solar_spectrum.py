import pandas as pd
from scipy.interpolate import interp1d

def load_solar_spectrum(file_path: str) -> callable:
    """
    Load the solar spectrum from a CSV file and return a function for interpolation.
    Args:
        file_path (str): Path to the solar spectrum CSV file.
    Returns:
        callable: Function that returns spectral irradiance for a given wavelength (nm).
    """
    data = pd.read_csv(file_path)
    wavelengths = data['wavelength'].values  # in nm
    intensity = data['intensity'].values    # in W/m^2/nm
    interp_func = interp1d(wavelengths, intensity, kind='linear', bounds_error=False, fill_value=0)
    return interp_func 