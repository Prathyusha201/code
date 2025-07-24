import pandas as pd
from scipy.interpolate import interp1d

class Material:
    """
    Class for handling material properties and refractive index data.
    """
    def __init__(self, data_file: str) -> None:
        """
        Initialize the material with data from a CSV file.
        Args:
            data_file (str): Path to the material data CSV file.
        """
        data = pd.read_csv(data_file)
        self.wavelengths = data['wavelength'].values  # in nm
        self.n = data['n'].values
        self.k = data['k'].values
        self.n_interp = interp1d(self.wavelengths, self.n, kind='linear', bounds_error=False, fill_value='extrapolate')
        self.k_interp = interp1d(self.wavelengths, self.k, kind='linear', bounds_error=False, fill_value='extrapolate')

    def get_nk(self, wavelength_nm: float) -> tuple:
        """
        Get the refractive index (n) and extinction coefficient (k) at a given wavelength.
        Args:
            wavelength_nm (float): Wavelength in nanometers.
        Returns:
            tuple: (n, k) values.
        """
        n = float(self.n_interp(wavelength_nm))
        k = float(self.k_interp(wavelength_nm))
        return n, k 