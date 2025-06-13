import pandas as pd
from scipy.interpolate import interp1d

class Material:
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.wavelengths = data['wavelength'].values  # in nm
        self.n = data['n'].values
        self.k = data['k'].values
        self.n_interp = interp1d(self.wavelengths, self.n, kind='linear', bounds_error=False, fill_value='extrapolate')
        self.k_interp = interp1d(self.wavelengths, self.k, kind='linear', bounds_error=False, fill_value='extrapolate')

    def get_nk(self, wavelength_nm):
        n = self.n_interp(wavelength_nm)
        k = self.k_interp(wavelength_nm)
        return n, k 