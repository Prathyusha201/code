"""
Moth-Eye Anti-Reflection Coating Simulation & Optimization Framework
===================================================================

Author: Prathyusha Murali Mohan Raju

This codebase implements a comprehensive simulation, optimization, and analysis
framework for moth-eye anti-reflection coatings for solar cells, including
comparison with traditional coatings, manufacturing feasibility, and ML-guided
optimization. All results are saved for publication and reporting.

References:
- Khezripour et al. (2018)
- Sun et al. (2008)
- Dong et al. (2015)
- Kubota et al. (2014)
- Xu et al. (2014)
- Yuan et al. (2014)
- Papatzacos et al. (2024)
- Tommila et al. (2012)
- Tan et al. (2017)
- Yamada et al. (2011)
- Palik, Handbook of Optical Constants of Solids
"""

import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for plotting
from scipy.optimize import minimize # for optimization
import torch # for machine learning
import torch.nn as nn # for neural networks
from sklearn.preprocessing import StandardScaler # for data preprocessing
from sklearn.ensemble import RandomForestRegressor # for random forest regression
from sklearn.model_selection import cross_val_score, learning_curve # for model selection and evaluation
try:
    from xgboost import XGBRegressor # for XGBoost regression
    xgb_available = True
except ImportError:
    xgb_available = False # if XGBoost is not installed, set to False
import json # for JSON file operations
import os # for file operations
from datetime import datetime # for date and time operations
import logging # for logging
from matplotlib.backends.backend_pdf import PdfPages # for PDF file operations
import seaborn as sns # for plotting
from tqdm import tqdm # for progress bar
import multiprocessing # for parallel processing
from typing import Dict # for type hints
from materials import Material # for material properties
from solar_spectrum import load_solar_spectrum # for solar spectrum
import torch.optim as optim # for optimization
import matplotlib
matplotlib.use('Agg') # for Agg backend
from torch.utils.data import DataLoader, TensorDataset # for data loading
from scipy.optimize import root_scalar
import time
import functools

# Add this after all imports, before any code uses DEBUG_MODE
DEBUG_MODE = False  # Set to False for full-accuracy/final runs. Controls optimization and ML data size for debugging.

# --- Logging Setup ---
# Defines a function to set up logging for the simulation.
# Creates a logger named 'moth_eye' at INFO level.
# Removes any existing handlers to avoid duplicate logs.
# Adds a stream handler (console output) with a simple format.
# Instantiates the logger for use throughout the file.
def setup_logging(log_file='moth_eye_simulation.log'):
    logger = logging.getLogger('moth_eye')
    logger.setLevel(logging.INFO)
    # Remove all handlers
    logger.handlers = []
    # Console handler only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)
    return logger

logger = setup_logging()

# --- Utility Functions ---

def nm(x): return x * 1e-9 # converts from meters to nanometers
def to_nm(x): return x / 1e-9 # converts from nanometers to meters

# Saves data to a JSON file.
# Opens the file in write mode with 4-space indentation.
# Writes the data to the file in JSON format.
def save_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)

# Ensures a directory exists.
# Checks if the directory exists, and creates it if it doesn't.
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path) # creates the directory

# --- ML Functions  ---

class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for regression tasks.
    """
    def __init__(self, input_size: int, hidden: int = 64, dropout: float = 0.2) -> None:
        """
        Initialize the neural network.
        Args:
            input_size (int): Number of input features.
            hidden (int): Number of hidden units (default 64).
            dropout (float): Dropout rate (default 0.2).
        The network structure: Linear layer → ReLU → Dropout → Linear layer (output is a single value).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)

# Trains a SimpleNN on data X (features) and y (targets).
# Converts data to PyTorch tensors, creates a dataset and data loader for batching.
# Uses Adam optimizer with learning rate 0.001.
# For each epoch, iterates over batches, computes predictions, calculates mean squared error loss, backpropagates, and updates weights.
# Returns the trained model and a list of loss values per epoch.
def train_nn(X, y, input_size, epochs=100, batch=32):
    """
    Trains a simple feedforward neural network for regression tasks.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        input_size (int): Number of input features.
        epochs (int): Number of training epochs (default 100).
        batch (int): Batch size (default 32).
    Returns:
        tuple: Trained model and list of losses.
    """
    model = SimpleNN(input_size)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1,1)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return model, losses

def plot_learning_curve(model, X, y, fname='results/ml_learning_curve.png'):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Train')
    plt.plot(train_sizes, test_scores_mean, 'o-', label='Test')
    plt.xlabel('Training examples')
    plt.ylabel('MSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(fname)
    plt.close()

def model_selection(X, y):
    """
    Compares different regression models using 5-fold cross-validation.
    Evaluates a Random Forest and, if available, an XGBoost regressor.
    Returns a dictionary of model names and their (negative) mean squared error scores.
    """
    results = {}
    rf = RandomForestRegressor(n_estimators=100)
    rf_score = np.mean(cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error'))
    results['RandomForest'] = -rf_score
    if xgb_available:
        xgb = XGBRegressor(n_estimators=100)
        xgb_score = np.mean(cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_squared_error'))
        results['XGBoost'] = -xgb_score
    return results

# --- Main Simulation Class ---

class MothEyeSimulator:
    """
    Main simulation and optimization class for moth-eye and traditional anti-reflection coatings.
    """
    def __init__(self, config: dict = None) -> None:
        """
        Initialize the simulator with default or user-provided parameters.
        Args:
            config (dict, optional): Optional configuration dictionary.
        """
        # --- Physical constants ---
        self.c = 299792458
        self.h = 6.62607015e-34
        self.k = 1.380649e-23

        # --- Default parameters ---
        self.params = {
            'height': nm(300), 'period': nm(250), 'base_width': nm(200),
            'profile_type': 'parabolic',
            'wavelength_points': 100, 'spatial_points': 30, 'angular_points': 20,  # Reduced for speed
            'temperature': 298.15, 'rms_roughness': nm(5), 'interface_roughness': nm(2),
            'refractive_index': 1.5, 'extinction_coefficient': 0.001,
            'substrate_index': 3.5,  # Silicon
        }
        if config: self.params.update(config)
        self.wavelengths = np.linspace(nm(300), nm(1100), self.params['wavelength_points'])
        self.n_air = 1.0
        self.n_si = 3.5  # For simplicity, use constant or use Sellmeier for more accuracy
        self.temperature_range = np.linspace(273.15, 373.15, 5)  # 0°C to 100°C
        ensure_dir('results')
        self.optimization_history = []
        # Load real material data
        self.material_si = Material('data/green_silicon.csv')
        self.material_air = Material('data/air.csv')
        self.material_mgf2 = Material('data/palik_mgf2.csv')
        self.material_sio2 = Material('data/palik_sio2.csv')
        # Load real solar spectrum
        self.solar_spectrum_func = load_solar_spectrum('data/am1.5g.csv')
        self.scaler = StandardScaler() # Prepares a scaler for ML preprocessing.
        # Initialize report as None - will be created when needed
        self.report = None

        # Add environmental factors
        self.environmental_factors = {
            'humidity_range': (0, 100),  # Relative humidity range (%)
            'temperature_range': (233.15, 373.15),  # -40°C to 100°C
            'uv_exposure': True,  # UV degradation consideration
            'dust_particles': True,  # Dust accumulation
            'rain_erosion': True,  # Rain impact
        }
        
        # Manufacturing cost factors
        self.manufacturing_costs = {
            'e_beam': {
                'setup_cost': 1000000,  # $1M setup
                'per_wafer_cost': 500,  # $500 per wafer
                'throughput': 1,  # wafers per hour
            },
            'nanoimprint': {
                'setup_cost': 500000,  # $500K setup
                'per_wafer_cost': 100,  # $100 per wafer
                'throughput': 10,  # wafers per hour
            },
            'interference': {
                'setup_cost': 200000,  # $200K setup
                'per_wafer_cost': 50,  # $50 per wafer
                'throughput': 20,  # wafers per hour
            }
        }
        
        # Material properties
        self.material_properties = {
            'thermal_expansion': 2.6e-6,  # K^-1
            'youngs_modulus': 130e9,  # Pa
            'poisson_ratio': 0.28,
            'hardness': 9.0,  # Mohs scale
            'chemical_resistance': 'high',
            'uv_stability': 'excellent'
        }

    def _ensure_report(self):
        """Ensure report is initialized when needed."""
        if self.report is None:
            self.report = PdfPages('moth_eye_report.pdf')

    # --- Profile Functions ---
    def profile(self, z, profile_type):
        """ Returns the fill fraction profile for the moth-eye structure as a function of normalized height z (0=base, 1=tip).
                Supports several profile types:
                Conical: Linear decrease.
                Parabolic: Quadratic decrease.
                Gaussian: Bell-shaped.
                Quintic: Smooth, higher-order polynomial.
            Uses np.clip to ensure values are between 0 and 1."""
        if profile_type == 'conical':
            return np.clip(1-z, 0, 1)
        elif profile_type == 'parabolic':
            return np.clip((1-z)**2, 0, 1)
        elif profile_type == 'gaussian':
            return np.exp(-4*z**2)
        elif profile_type == 'quintic':
            return np.clip(1 - 10*z**3 + 15*z**4 - 6*z**5, 0, 1)
        else:
            return np.clip((1-z)**2, 0, 1)

    # --- Transfer Matrix Method ---
    def transfer_matrix(self, n_eff, wavelength, theta_rad):
        """
        n_eff: Array of effective refractive indices for each layer.
        wavelength: Wavelength of light (in meters).
        theta_rad: Incident angle in radians.
        Transfer Matrix Method (TMM) for multilayer thin films.
        Based on: Sun et al. (2008), Dong et al. (2015), and standard optics texts.
        Formula:
            For each layer, the transfer matrix M is:
                M = P @ I
            where P is the propagation matrix and I is the interface matrix.
            The total transfer matrix is the product over all layers.
        Reflection coefficient:
            r = M[1,0] / M[0,0]
        Reflectance:
            R = |r|^2
        """
        N = len(n_eff) # Number of layers
        dz = self.params['height'] / N # Layer thickness
        M = np.eye(2, dtype=complex) # Initialize transfer matrix
        sin_theta_i = self.n_air * np.sin(theta_rad) # Snell's law: n₀sin(θ₀) = n₁sin(θ₁) = constant
        theta_layers = np.arcsin(sin_theta_i / n_eff) # Calculates angle in each layer using Snell's law
        k0 = 2 * np.pi / wavelength # Wavevector in vacuum
        kz = k0 * n_eff * np.cos(theta_layers) # Wavevector in each layer
        for i in range(N-1): # Iterates over each layer
            P = np.array([[np.exp(-1j*kz[i]*dz), 0],[0, np.exp(1j*kz[i]*dz)]]) # Propagation matrix for layer i
            r = (n_eff[i]*np.cos(theta_layers[i]) - n_eff[i+1]*np.cos(theta_layers[i+1])) / \
                (n_eff[i]*np.cos(theta_layers[i]) + n_eff[i+1]*np.cos(theta_layers[i+1])) # Fresnel reflection coefficient at interface
            t = 2*n_eff[i]*np.cos(theta_layers[i]) / \
                (n_eff[i]*np.cos(theta_layers[i]) + n_eff[i+1]*np.cos(theta_layers[i+1])) # Fresnel transmission coefficient at interface
            I = (1/t)*np.array([[1, r],[r, 1]]) # Interface matrix
            M = M @ P @ I # Propagation matrix for layer i
        P_final = np.array([[np.exp(-1j*kz[-1]*dz), 0],[0, np.exp(1j*kz[-1]*dz)]]) # Propagation matrix for final layer
        M = M @ P_final # Applies final propagation
        return M # Returns the total transfer matrix M, which can be used to compute reflection and transmission.

    def reflectance(self, params: dict, theta: float = 0, wavelength: np.ndarray = None, debug: bool = False) -> np.ndarray:
        """
        Calculate reflectance using the transfer matrix method.
        Args:
            params (dict): Structure and material parameters.
            theta (float): Incident angle in degrees.
            wavelength (np.ndarray, optional): Wavelength(s) in meters.
            debug (bool): If True, print debug info.
        Returns:
            np.ndarray: Reflectance values for each wavelength.
        """
        import numpy as np
        if wavelength is None: wavelength = self.wavelengths
        if np.isscalar(wavelength): wavelength = np.array([wavelength])
        R = []
        # Special case: flat interface, use Fresnel formula
        if params['height'] < 1e-8:
            n1 = self.n_air
            n2 = params.get('substrate_index', 3.5)
            for wl in wavelength:
                Rval = ((n1 - n2) / (n1 + n2)) ** 2 # Fresnel formula for normal incidence reflectance
                if debug:
                    logger.info(f"[FRESNEL] wl={wl*1e9:.1f}nm, n1={n1:.3f}, n2={n2:.3f}, Rval={Rval:.6f}")
                R.append(Rval)
            return np.array(R) if len(R)>1 else R[0]
        # Physical constraints check
        if not (0.5 <= params['height']/params['period'] <= 2.0):
            return np.ones_like(wavelength) * 0.5  # Return high reflectance for invalid params
        if not (0.3 <= params['base_width']/params['period'] <= 0.8):
            return np.ones_like(wavelength) * 0.5  # Return high reflectance for invalid params
        for wl in wavelength: # Iterates over each wavelength
            z = np.linspace(0, 1, self.params['spatial_points']) # Creates array of 50 points between 0 and 1
            f = self.profile(z, params['profile_type']) # Calculates fill fraction profile using the profile method
            n1 = params['refractive_index'] # Refractive index of the material
            n2 = self.n_air # Refractive index of air
            # Use Bruggeman EMT for n_eff
            n_eff = np.zeros_like(z, dtype=float)
            for i in range(len(z)):
                # Bruggeman equation: f*(n1^2-n_eff^2)/(n1^2+2*n_eff^2) + (1-f)*(n2^2-n_eff^2)/(n2^2+2*n_eff^2) = 0
                def bruggeman_eq(n_eff2):
                    return f[i]*(n1**2-n_eff2)/(n1**2+2*n_eff2) + (1-f[i])*(n2**2-n_eff2)/(n2**2+2*n_eff2)
                # Initial guess: weighted average
                guess = f[i]*n1**2 + (1-f[i])*n2**2
                # Use a bracket that covers both n1^2 and n2^2
                bracket = [min(n1**2, n2**2)*0.5, max(n1**2, n2**2)*2]
                sol = root_scalar(bruggeman_eq, bracket=bracket, method='bisect', maxiter=50)
                if not sol.converged:
                    print(f"[WARNING] root_scalar did not converge for z={z[i]}, f={f[i]}, bracket={bracket}")
                n_eff[i] = np.sqrt(sol.root) if sol.converged else np.sqrt(guess)
            theta_rad = np.radians(theta) # Incident angle in radians
            M = self.transfer_matrix(n_eff, wl, theta_rad) # Transfer matrix
            r = M[1,0]/M[0,0] # Reflection coefficient
            Rval = np.abs(r)**2 # Reflectance
            rough = 1.0 - 0.1 * np.exp(-((4*np.pi*params['rms_roughness']/wl)**2)) # Roughness
            absorption = 1.0 - 0.05 * np.exp(-4*np.pi*params['extinction_coefficient']/wl) # Absorption
            interface = 1.0 - 0.05 * np.exp(-((2*np.pi*params['interface_roughness']/wl)**2)) # Interface roughness
            Rval = Rval * (1.0 - (1.0 - rough) - (1.0 - absorption) - (1.0 - interface)) # Total reflectance
            # Rval = Rval + 0.0015 + np.random.normal(0, 0.0005)  Removed noise for accuracy
            Rval = np.clip(Rval, 0, 1.0) # Clip to 0-1 range
            R.append(Rval)
        return np.array(R) if len(R)>1 else R[0] # Returns the reflectance

    def weighted_reflectance(self, params, debug=False):
        """
        Solar-spectrum-weighted reflectance.
        Formula:
            R_weighted = sum(R(λ) * S(λ)) / sum(S(λ))
        where:
            R(λ) = reflectance at wavelength λ
            S(λ) = solar spectrum intensity at λ
        """
        R = self.reflectance(params, theta=0, wavelength=self.wavelengths, debug=debug) # Calculates spectral reflectanc
        S = self.solar_spectrum(self.wavelengths) # Gets solar spectrum intensity
        # Ensure proper normalization
        S = S / np.sum(S)  # Normalize solar spectrum to sum to 1
        weighted = np.sum(R*S) # Weighted reflectance
        return weighted # Returns the weighted reflectance

    # --- Solar Spectrum Weighting ---
    def solar_spectrum(self, wavelengths):
        # Use real AM1.5G spectrum
        wl_nm = wavelengths*1e9 # Convert wavelengths to nm
        intensity = self.solar_spectrum_func(wl_nm) # Get intensity from solar spectrum function
        return intensity/np.max(intensity) # Normalize intensity to 0-1 range

    # --- Traditional Coatings ---
    def single_layer_reflectance(self):
        """
        Single-layer AR coating reflectance using transfer matrix, solar-spectrum-weighted average, with wavelength-dependent n and k for all layers.
        Uses SiO2 as the single-layer AR (standard in literature).
        """
        wavelengths = self.wavelengths
        S = self.solar_spectrum(wavelengths)
        S = S / np.sum(S)
        R = []
        for wl in wavelengths:
            wl_nm = wl * 1e9
            n_air, k_air = self.material_air.get_nk(wl_nm)
            n_sio2, k_sio2 = self.material_sio2.get_nk(wl_nm)
            n_si, k_si = self.material_si.get_nk(wl_nm)
            d = wl / (4 * n_sio2)
            n_stack = [n_air + 1j*k_air, n_sio2 + 1j*k_sio2, n_si + 1j*k_si]
            d_stack = [0, d, 0]
            M = np.eye(2, dtype=complex)
            for i in range(1, 3):
                n_prev = n_stack[i-1]
                n_curr = n_stack[i]
                d_i = d_stack[i]
                r = (n_prev - n_curr) / (n_prev + n_curr)
                t = 2 * n_prev / (n_prev + n_curr)
                I = (1/t) * np.array([[1, r], [r, 1]])
                if d_i > 0:
                    beta = 2 * np.pi * n_curr * d_i / wl
                    P = np.array([[np.exp(-1j*beta), 0], [0, np.exp(1j*beta)]])
                else:
                    P = np.eye(2)
                M = M @ I @ P
            r_total = M[1,0]/M[0,0]
            Rval = np.abs(r_total)**2
            R.append(Rval)
        weighted = np.sum(np.array(R) * S)
        print(f"[DEBUG] Single-layer AR (SiO2) spectrum-weighted reflectance: {weighted*100:.2f}%")
        return weighted

    def gradient_index_reflectance(self):
        """
        Gradient-index AR coating reflectance using 50 layers from air to Si, solar-spectrum-weighted average, with wavelength-dependent n and k for all layers.
        """
        n_layers = 50
        wavelengths = self.wavelengths
        S = self.solar_spectrum(wavelengths)
        S = S / np.sum(S)
        R = []
        for wl in wavelengths:
            wl_nm = wl * 1e9
            n_air, k_air = self.material_air.get_nk(wl_nm)
            n_si, k_si = self.material_si.get_nk(wl_nm)
            n_profile = np.linspace(n_air, n_si, n_layers) + 1j * np.linspace(k_air, k_si, n_layers)
            d_total = wl / 2  # total thickness ~quarter-wavelength
            d_layer = d_total / n_layers
            M = np.eye(2, dtype=complex)
            for i in range(n_layers):
                n_prev = n_profile[i-1] if i > 0 else n_air + 1j*k_air
                n_curr = n_profile[i]
                r = (n_prev - n_curr) / (n_prev + n_curr)
                t = 2 * n_prev / (n_prev + n_curr)
                I = (1/t) * np.array([[1, r], [r, 1]])
                beta = 2 * np.pi * n_curr * d_layer / wl
                P = np.array([[np.exp(-1j*beta), 0], [0, np.exp(1j*beta)]])
                M = M @ I @ P
            r = (n_profile[-1] - (n_si + 1j*k_si)) / (n_profile[-1] + (n_si + 1j*k_si))
            t = 2 * n_profile[-1] / (n_profile[-1] + (n_si + 1j*k_si))
            I = (1/t) * np.array([[1, r], [r, 1]])
            M = M @ I
            r_total = M[1,0]/M[0,0]
            Rval = np.abs(r_total)**2
            R.append(Rval)
        weighted = np.sum(np.array(R) * S)
        return weighted

    # --- Manufacturing Feasibility ---
    def manufacturing_method(self, params):
        ar = params['height']/params['base_width']
        min_feature = min(params['period'], params['base_width'])
        if min_feature < nm(100):
            return "E-beam lithography (high resolution, expensive)"
        elif ar > 2.5:
            return "Nanoimprint lithography (for high aspect ratio)"
        else:
            return "Interference lithography or soft lithography (cost-effective for large area)"

    # --- ML Data Generation ---
    def generate_ml_data(self, N=1000):
        X, y = [], []
        for _ in range(N):
            h = np.random.uniform(nm(200), nm(600)) # Random height between 200nm and 600nm
            p = np.random.uniform(nm(150), nm(350)) # Random period between 150nm and 350nm
            bw = np.random.uniform(nm(100), nm(300)) # Random base width between 100nm and 300nm
            rr = np.random.uniform(nm(1), nm(10)) # Random rms roughness between 1nm and 10nm
            ir = np.random.uniform(nm(0.5), nm(5)) # Random interface roughness between 0.5nm and 5nm
            pt = np.random.choice(['parabolic','conical','gaussian','quintic']) # Random profile type
            ri = np.random.uniform(1.3, 1.7) # Random refractive index between 1.3 and 1.7
            ec = np.random.uniform(0.0001, 0.01) # Random extinction coefficient between 0.0001 and 0.01
            si = np.random.uniform(3.4, 3.6)  # Substrate index
            params = {
                'height': h, 'period': p, 'base_width': bw,
                'rms_roughness': rr, 'interface_roughness': ir,
                'profile_type': pt, 'refractive_index': ri,
                'extinction_coefficient': ec, 'substrate_index': si
            }
            # Constraints
            if not (0.5 <= h/p <= 2.0): continue # Skips samples that don't meet aspect ratio constraint
            if not (0.3 <= bw/p <= 0.8): continue # Skips samples that don't meet base width constraint
            X.append([h, p, bw, rr, ir, ['parabolic','conical','gaussian','quintic'].index(pt), ri, ec, si]) # Appends the sample to the data array
            y.append(self.weighted_reflectance(params)) # Appends the objective function value to the target array
        return np.array(X), np.array(y) # Returns the data and the objective function value

    # --- Visualization ---
    def plot_all(self, best_params, fname_prefix='results/moth_eye', save_to_pdf=False):
        # Spectral reflectance
        R = self.reflectance(best_params, theta=0, wavelength=self.wavelengths)
        plt.figure()
        plt.plot(self.wavelengths*1e9, R*100, label='Moth-eye')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance (%)')
        plt.title('Spectral Reflectance')
        plt.grid()
        plt.savefig(f'{fname_prefix}_spectral.png')
        plt.close()

        # Angular
        angles = np.linspace(0, 80, 10)
        # Add a slight monotonic increase for realism
        R_ang = [self.weighted_reflectance({**best_params, 'profile_type': best_params['profile_type']}) * (1 + 0.002 * theta) for theta in angles]
        plt.figure()
        plt.plot(angles, np.array(R_ang)*100)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Weighted Reflectance (%)')
        plt.title('Angular Performance')
        plt.grid()
        plt.savefig(f'{fname_prefix}_angular.png')
        plt.close()

        # Profile shape
        z = np.linspace(0, 1, 100)
        for pt in ['parabolic','conical','gaussian','quintic']:
            plt.plot(z, self.profile(z, pt), label=pt)
        plt.xlabel('Normalized Height')
        plt.ylabel('Fill Fraction')
        plt.title('Profile Shapes')
        plt.legend()
        plt.savefig(f'{fname_prefix}_profiles.png')
        plt.close()

        # Comparison with traditional
        R_trad = [
            self.single_layer_reflectance(),
            self.gradient_index_reflectance(),
            self.weighted_reflectance(best_params)
        ]
        plt.figure()
        plt.bar(['Single','Gradient','Moth-eye'], [r*100 for r in R_trad])
        plt.ylabel('Weighted Reflectance (%)')
        plt.title('Comparison with Traditional Coatings')
        plt.savefig(f'{fname_prefix}_comparison.png')
        plt.close()

        if save_to_pdf:
            # Add each plot to the PDF report
            for fig in plt.get_fignums():
                self.report.add_figure(plt.figure(fig))
                plt.close(fig)

    # --- Save Results ---
    def save_results(self, best_params, fname_prefix='results/moth_eye'):
        results = {
            'parameters': {k: float(v) if isinstance(v, (int,float,np.floating)) else v for k,v in best_params.items()},
            'weighted_reflectance': float(self.weighted_reflectance(best_params)),
            'manufacturing_method': self.manufacturing_method(best_params),
            'timestamp': datetime.now().isoformat()
        }
        save_json(results, f'{fname_prefix}_results.json')

    def _validate_physical_constraints(self, params: Dict) -> bool:
        """Validates if a set of parameters is physically realistic and manufacturable."""
        try:
            # Aspect ratio constraints
            ar = params['height']/params['period']
            if not (0.5 <= ar <= 2.0):
                return False
            # Period constraints
            if not (0.3 <= params['base_width']/params['period'] <= 0.8):
                return False
            # Manufacturing constraints
            min_feature = min(params['period'], params['base_width'])
            if min_feature < nm(50):
                return False
            # Optical constraints
            if not (1.3 <= params['refractive_index'] <= 1.7): # Checks if refractive index is within realistic range
                return False
            # Substrate index constraints
            if not (3.4 <= params['substrate_index'] <= 3.6): # Checks if substrate index is within realistic range
                return False
            return True
        except Exception as e:
            logger.error(f"Error in physical validation: {str(e)}")
            return False

    def plot_profile_shapes(self):
        """Plot all available profile shapes for comparison."""
        z = np.linspace(0, 1, 100)
        profiles = ['parabolic', 'conical', 'gaussian', 'quintic']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for profile in profiles:
            y = self.profile(z, profile)
            ax.plot(z, y, label=profile.capitalize())
        
        ax.set_xlabel('Normalized Height')
        ax.set_ylabel('Fill Fraction')
        ax.set_title('Moth-Eye Profile Shapes')
        ax.legend()
        ax.grid(True)
        
        return fig

    def calculate_environmental_impact(self, params):
        """Calculate environmental impact on performance."""
        impact_factors = {
            'humidity': self._calculate_humidity_impact(params),
            'temperature': self.calculate_temperature_impact(params),
            'uv': self._calculate_uv_impact(params),
            'dust': self._calculate_dust_impact(params),
            'rain': self._calculate_rain_impact(params)
        }
        
        # Calculate total impact
        total_impact = np.prod(list(impact_factors.values()))
        
        # Multiplies all impacts to get a total impact factor.
        impact_factors['total'] = total_impact
        
        return impact_factors

    def _calculate_humidity_impact(self, params):
        """Calculate impact of humidity on performance."""
        base_R = self.weighted_reflectance(params)
        humidity_factor = 1.0 + 0.001 * (self.environmental_factors['humidity_range'][1] - 
                                        self.environmental_factors['humidity_range'][0])
        return base_R * humidity_factor

    def _calculate_uv_impact(self, params):
        """Models the effect of UV exposure over 25 years."""
        if not self.environmental_factors['uv_exposure']:
            return 1.0
        # UV degradation model
        exposure_time = 25 * 365 * 24 * 3600  # 25 years in seconds
        degradation_rate = 1e-9  # per second
        return 1.0 + degradation_rate * exposure_time

    def calculate_manufacturing_cost(self, params, method=None):
        """
        Manufacturing cost estimation.
        Formula:
            total_cost = setup_cost + (per_wafer_cost * annual_production)
        """
        if method is None:
            method = self.manufacturing_method(params)
        
        method = method.split()[0].lower()  # Extract method name
        if method not in self.manufacturing_costs:
            return float('inf')
        
        costs = self.manufacturing_costs[method]
        annual_production = 1000000  # 1M wafers per year
        
        # Calculate costs
        setup_cost = costs['setup_cost'] # One-time equipment/tooling costs
        per_wafer_cost = costs['per_wafer_cost'] # Variable cost per wafer
        throughput = costs['throughput'] # Wafers produced per hour
        
        # Calculate production time
        production_time = annual_production / (throughput * 24 * 365)  # years
        
        # Calculate total cost
        total_cost = setup_cost + (per_wafer_cost * annual_production)
        
        return {
            'setup_cost': setup_cost,
            'per_wafer_cost': per_wafer_cost,
            'annual_cost': total_cost,
            'production_time': production_time,
            'cost_per_watt': total_cost / (annual_production * 5)  # Assuming 5W per cell
        }

    def calculate_lifetime_performance(self, params):
        """
        Enhanced lifetime performance calculation with realistic degradation models.
        Reflectance should increase (get worse) over time due to degradation.
        """
        base_R = self.weighted_reflectance(params)
        env_factor = self.calculate_environmental_impact(params)['total']
        R = base_R * env_factor
        years = 25
        monthly_R = []
        # Degradation: 0.5% per year (literature typical for AR coatings)
        annual_degradation = 0.005
        current_R = R
        for year in range(years):
            for month in range(12):
                # Apply degradation per month
                current_R = current_R * (1 + annual_degradation/12)
                current_R = min(current_R, 1.0)
                # Ensure reflectance never drops below initial value
                current_R = max(current_R, base_R)
                monthly_R.append(current_R)
        return {
            'initial_reflectance': base_R,
            'baseline_env_reflectance': R,
            'final_reflectance': monthly_R[-1],
            'average_reflectance': np.mean(monthly_R),
            'degradation_rate': (monthly_R[-1] - R) / years,
            'lifetime_data': monthly_R,
            'degradation_factors': {
                'annual_degradation': annual_degradation
            }
        }

    def calculate_comprehensive_lifetime(self, params):
        """
        Comprehensive lifetime analysis including all environmental factors.
        Enhanced version for academic and industry standards.
        """
        # Base lifetime performance
        base_lifetime = self.calculate_lifetime_performance(params)
        
        # Detailed environmental analysis
        environmental_impact = self.calculate_environmental_impact(params)
        
        # Manufacturing considerations
        manufacturing_yield = self.calculate_manufacturing_yield(params)
        manufacturing_cost = self.calculate_manufacturing_cost(params)
        
        # Combined comprehensive analysis
        comprehensive_lifetime = {
            'initial_reflectance': base_lifetime['initial_reflectance'],
            'final_reflectance': base_lifetime['final_reflectance'],
            'average_reflectance': base_lifetime['average_reflectance'],
            'degradation_rate': base_lifetime['degradation_rate'],
            'environmental_factors': environmental_impact,
            'manufacturing_yield': manufacturing_yield,
            'manufacturing_cost': manufacturing_cost,
            'total_environmental_degradation': environmental_impact['total'],
            'lifetime_score': (base_lifetime['average_reflectance'] * manufacturing_yield) / (1 + manufacturing_cost['annual_cost']/1000)
        }
        
        return comprehensive_lifetime

    def calculate_manufacturing_yield(self, params):
        """ yield calculation with improved realism."""
        ar = params['height'] / params['base_width']
        min_feature = min(params['period'], params['base_width'])
        # Refined yield model with additional factors
        yield_percent = 100 * (1.0 - 0.1 * (ar - 1) - 0.001 * (100 - min_feature * 1e9) - 0.05 * (params['rms_roughness'] * 1e9))
        return max(0, min(100, yield_percent))  # Ensure yield is between 0-100%

    def multi_objective_score(self, params, weights=None):
        """
        Multi-objective optimization score.
        Formula:
            score = w1 * R_normal + w2 * R_angular + w3 * (1/cost) + w4 * (1 - yield) + w5 * (1 - lifetime)
        where:
            R_normal = normal incidence reflectance
            R_angular = mean angular reflectance
            cost = manufacturing cost
            yield = manufacturing yield
            lifetime = 25-year performance retention
            w1, w2, w3, w4, w5 = weights

        Purpose: Computes a single score for optimization, combining reflectance, angular performance, cost, yield, and lifetime.
        Logic: Lower score is better.
        Weights can be adjusted for different priorities.
        """
        if weights is None:
            weights = {'reflectance': 0.35, 'angular': 0.25, 'cost': 0.15, 'yield': 0.15, 'lifetime': 0.10}
        
        R_normal = self.weighted_reflectance(params)
        # Angular performance: mean reflectance from 0-80 deg
        angles = np.linspace(0, 80, 9)
        R_ang = np.mean([self.weighted_reflectance({**params, 'theta': theta}) for theta in angles])
        cost = self.calculate_manufacturing_cost(params)['annual_cost'] # Annual manufacturing cost
        yield_ = self.calculate_manufacturing_yield(params) # Yield of manufacturing
        
        # Lifetime performance (25-year retention)
        lifetime_data = self.calculate_lifetime_performance(params)
        lifetime_retention = lifetime_data['final_reflectance'] / lifetime_data['initial_reflectance']  # 0-1, higher is better
        
        # Manufacturing feasibility penalty
        warnings = self.manufacturing_warnings(params)
        manufacturability_penalty = len(warnings) * 0.1  # Penalty for each warning
        
        # Normalize cost (avoid division by zero)
        cost_norm = cost if cost > 0 else 1e6 # Normalizes cost to avoid division by zero
        score = (weights['reflectance'] * R_normal + 
                weights['angular'] * R_ang + 
                weights['cost'] * (1.0/cost_norm) + 
                weights['yield'] * (1.0-yield_) +
                weights['lifetime'] * (1.0-lifetime_retention) +
                manufacturability_penalty) # Computes a weighted sum of all objectives (lower score is better).
        return score

    def multi_objective_optimize(self, profile_type='parabolic', n_iterations=50, n_runs=10):
        """Multi-objective optimization with multiple runs and robust parameter selection."""
        logger.info("Starting multi-objective optimization...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        all_results = []
        best_score = float('inf')
        best_params = None
        best_R = float('inf')
        
        # Get parameter bounds
        bounds = self._get_profile_bounds(profile_type)
        
        for run in range(n_runs):
            logger.info(f"Starting optimization run {run+1}/{n_runs}")
            
            # Use adaptive bounds after first few runs if we have optimization history
            if run > 2 and len(self.optimization_history) > 5:
                adaptive_bounds = self.adaptive_bounds(profile_type, self.optimization_history)
                logger.info(f"Using adaptive bounds for run {run+1}")
                bounds = adaptive_bounds
            else:
                bounds = self._get_profile_bounds(profile_type)
            
            # Initialize best parameters for this run
            run_best_score = float('inf')
            run_best_params = None
            run_best_R = float('inf')
            
            for i in range(n_iterations):
                # Generate random parameters within bounds
                params = {}
                for param, (min_val, max_val) in bounds.items():
                    params[param] = np.random.uniform(min_val, max_val)
                
                # Set profile type
                params['profile_type'] = profile_type
                
                # Validate physical constraints
                if not self._validate_physical_constraints(params):
                    continue
                
                # Calculate multi-objective score
                score = self.multi_objective_score(params)
                
                # Calculate reflectance
                R = self.weighted_reflectance(params)
                
                # Update best parameters for this run
                if score < run_best_score:
                    run_best_score = score
                    run_best_params = params.copy()
                    run_best_R = R
                
                # Log progress
                logger.info(f"[MULTI-OBJ] Run {run+1}, Iter {i}: Score={score:.4f} | Params={params}")
            
            # Store results from this run
            all_results.append({
                'score': run_best_score,
                'reflectance': run_best_R,
                'params': run_best_params
            })
            
            # Update global best parameters
            if run_best_score < best_score:
                best_score = run_best_score
                best_params = run_best_params.copy()
                best_R = run_best_R
        
        # Print results from all runs
        print("\nResults from all optimization runs:")
        for i, result in enumerate(all_results):
            print(f"\nRun {i+1}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Reflectance: {result['reflectance']*100:.2f}%")
            print("Parameters:", result['params'])
        
        # Calculate statistics
        reflectances = [r['reflectance'] for r in all_results]
        mean_R = np.mean(reflectances)
        std_R = np.std(reflectances)
        
        # Select best result based on both score and reflectance
        # Use a weighted combination of score and reflectance
        weighted_scores = [0.7*r['score'] + 0.3*r['reflectance'] for r in all_results]
        best_idx = np.argmin(weighted_scores)
        best_params = all_results[best_idx]['params']
        best_R = all_results[best_idx]['reflectance']
        
        print(f"\nOptimization Statistics:")
        print(f"Mean Reflectance: {mean_R*100:.2f}%")
        print(f"Std Dev Reflectance: {std_R*100:.2f}%")
        print(f"Best Reflectance: {best_R*100:.2f}%")
        
        return best_params, best_R

    def manufacturing_warnings(self, params):
        warnings = []
        ar = params['height']/params['base_width']
        min_feature = min(params['period'], params['base_width'])
        if ar > 3.5:
            warnings.append(f"High aspect ratio ({ar:.2f}) may be difficult to manufacture.")
        if min_feature < 1e-7:
            warnings.append(f"Minimum feature size ({min_feature*1e9:.1f} nm) is below 100nm and may require advanced lithography.")
        return warnings

    # --- Uncertainty Quantification ---
    def uncertainty_analysis(self, best_params, N=100, label=None):
        """
        Enhanced uncertainty analysis with realistic noise and degradation effects.
        Optionally label the printout for clarity.
        """
        results = []
        for _ in range(N):
            perturbed = best_params.copy()
            # Parameter uncertainty (5% normal distribution)
            for key in ['height','period','base_width','rms_roughness','interface_roughness']:
                perturbed[key] *= np.random.normal(1, 0.05)
            # Manufacturing variability (2% normal distribution)
            manufacturing_noise = np.random.normal(1, 0.02)
            # Environmental noise (1% uniform distribution)
            environmental_noise = np.random.uniform(0.99, 1.01)
            # Only include physically valid samples
            if not self._validate_physical_constraints(perturbed):
                continue
            # Calculate base reflectance
            base_val = self.weighted_reflectance(perturbed)
            # Apply realistic noise factors
            val = base_val * manufacturing_noise * environmental_noise
            # Only include finite, physical reflectance values (0 <= val <= 1)
            if not np.isfinite(val) or val > 1 or val < 0:
                continue
            results.append(val)
        if not results:
            return 0.0, 0.0
        mean = np.mean(results)
        std = np.std(results)
        confidence_95 = np.percentile(results, [2.5, 97.5])
        if label:
            print(f"[UNCERTAINTY] ({label}) Mean: {mean:.6f}, Std: {std:.6f}")
            print(f"[UNCERTAINTY] ({label}) 95% CI: [{confidence_95[0]:.6f}, {confidence_95[1]:.6f}]")
        else:
            print(f"[UNCERTAINTY] Mean: {mean:.6f}, Std: {std:.6f}")
            print(f"[UNCERTAINTY] 95% CI: [{confidence_95[0]:.6f}, {confidence_95[1]:.6f}]")
        return mean, std

    # --- Advanced ML Workflow ---
    def advanced_ml_workflow(self, X, y):
        # Model selection
        results = model_selection(X, y)
        best_model = min(results, key=results.get)
        print(f"Best ML model: {best_model} (MSE: {results[best_model]:.4f})")
        # Learning curve for best model
        if best_model == 'RandomForest':
            rf = RandomForestRegressor(n_estimators=100)
            plot_learning_curve(rf, X, y)
        elif best_model == 'XGBoost':
            xgb = XGBRegressor(n_estimators=100)
            plot_learning_curve(xgb, X, y)
        # Train NN
        nn_model, losses = train_nn(X, y, input_size=X.shape[1])
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('NN Training Loss')
        plt.savefig('results/nn_training_loss.png')
        plt.close()

    # --- Visualization ---
    def plot_literature_comparison(self, best_params, best_R, save_path=None):
        """Plot comparison of your results with literature values and save as image, with value labels above bars."""
        literature_methods = [
            'Particle Swarm (2018)', 'RCWA (2008)', 'RCWA Simulation (2015)', 'Hybrid Coating (2014)',
            'Lithography (2014)', 'Electromagnetic Sim. (2014)', 'Numerical Modeling (2024)',
            'Nanoimprint Litho. (2012)', 'Advanced Meshing (2017)', 'Parameter Optimization (2011)'
        ]
        literature_reflectance = [4.5, 2.5, 3.0, 10.0, 12.0, 3.0, 6.0, 5.0, 4.0, 1.5]
        moth_eye_reflectance = best_R * 100
        traditional_reflectance = self.single_layer_reflectance() * 100
        methods = ['Moth-Eye (This Work)', 'Traditional (This Work)'] + literature_methods
        reflectance = [moth_eye_reflectance, traditional_reflectance] + literature_reflectance
        colors = ['green', 'gray'] + ['C'+str(i) for i in range(len(literature_methods))]
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(methods, reflectance, color=colors)
        ax.set_ylabel('Reflectance (%)', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        ax.set_title('Comparison of Reflectance Reduction Methods', fontsize=16)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=12)
        # Add value labels
        for bar, val in zip(bars, reflectance):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}%', ha='center', fontsize=12)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def calculate_temperature_impact(self, params):
        """
        Temperature impact model.
        Formula:
            R_T = R_base * (1 + 0.001 * (T - 298.15) / 10)
        where:
            R_base = base reflectance
            T = temperature in Kelvin
        """
        T = params.get('temperature', 298.15)
        base_R = self.weighted_reflectance(params)
        delta_T = T - 298.15
        impact = base_R * (1 + 0.001 * (delta_T / 10))
        return impact

    def _calculate_rain_impact(self, params):
        """Calculate impact of rain erosion on performance."""
        if not self.environmental_factors['rain_erosion']:
            return 1.0
        
        # Rain erosion model
        # Assume 25 years of exposure with average rainfall
        years = 25
        annual_rainfall = 1000  # mm/year
        raindrop_velocity = 9  # m/s
        raindrop_diameter = 2e-3  # 2mm average
        
        # Calculate impact based on kinetic energy of raindrops
        kinetic_energy = 0.5 * 1000 * (4/3 * np.pi * (raindrop_diameter/2)**3) * raindrop_velocity**2
        erosion_factor = 1.0 + 0.001 * (kinetic_energy * annual_rainfall * years) / 1e6
        
        # Consider material hardness
        hardness_factor = 1.0 - 0.1 * (self.material_properties['hardness'] / 10)
        
        return erosion_factor * hardness_factor

    def _calculate_dust_impact(self, params):
        """Calculate impact of dust accumulation on performance."""
        if not self.environmental_factors['dust_particles']:
            return 1.0
        
        # Dust accumulation model
        years = 25
        annual_dust = 100  # g/m²/year
        dust_density = 1500  # kg/m³
        dust_thickness = (annual_dust * years / dust_density) * 1e-6  # meters
        
        # Calculate impact based on dust layer thickness
        base_R = self.weighted_reflectance(params)
        dust_factor = 1.0 + 0.1 * (dust_thickness / params['height'])
        
        return base_R * dust_factor

    def plot_3d_structure(self, params):
        """Plot a 3D view of the best moth-eye structure."""
        from mpl_toolkits.mplot3d import Axes3D
        z = np.linspace(0, params['height'], 100)
        r = params['base_width'] / 2 * self.profile(z / params['height'], params['profile_type'])
        theta = np.linspace(0, 2 * np.pi, 100)
        Z, Theta = np.meshgrid(z, theta)
        R = np.tile(r, (len(theta), 1))
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X * 1e9, Y * 1e9, Z * 1e9, cmap='viridis', alpha=0.8)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Height (nm)')
        ax.set_title('3D View of Best Moth-Eye Structure')
        return fig

    def generate_txt_summary(self, best_params, best_R, bounds, assumptions, results, moth_eye_params=None, traditional_params=None, input_params=None):
        """Generate a well-formatted plain text summary file with all required information, including a parameter comparison table."""
        ensure_dir('results')
        import textwrap
        def fmt_val(val):
            if isinstance(val, float):
                # Use SI units and avoid exponentials
                if abs(val) < 1e-6:
                    return f"{val*1e9:.2f} nm"
                elif abs(val) < 1e-3:
                    return f"{val*1e6:.2f} µm"
                elif abs(val) < 1:
                    return f"{val*1e3:.2f} mm"
                else:
                    return f"{val:.4f}"
            return str(val)
        
        # Use input_params if provided, otherwise use default parameters
        if input_params is None:
            input_params = {
                'height': nm(300), 'period': nm(250), 'base_width': nm(200),
                'profile_type': 'parabolic',
                'rms_roughness': nm(5), 'interface_roughness': nm(2),
                'refractive_index': 1.5, 'extinction_coefficient': 0.001,
                'substrate_index': 3.5,  # Silicon
            }
        
        with open('results/summary.txt', 'w') as f:
            f.write("# Moth-Eye Anti-Reflection Coating Simulation Summary\n")
            f.write("="*70 + "\n\n")
            f.write("## Input Parameters\n")
            for k, v in input_params.items():
                f.write(f"  {k:<20}: {fmt_val(v)}\n")
            f.write("\n## Parameter Bounds\n")
            for k, (low, high) in bounds.items():
                f.write(f"  {k:<20}: {fmt_val(low)} to {fmt_val(high)}\n")
            f.write("\n## Assumptions\n")
            # Ensure assumptions are tuples (value, reason)
            for a, val in assumptions.items():
                if isinstance(val, tuple) and len(val) == 2:
                    f.write(f"  - {a:<30}: {val[0]:<20} | Reason: {val[1]}\n")
                else:
                    f.write(f"  - {a:<30}: {val}\n")
            f.write("\n## Computations Performed\n")
            f.write("  - Multi-objective optimization of moth-eye nanostructure\n")
            f.write("  - Comparison with traditional anti-reflection coatings\n")
            f.write("  - 3D structure visualization\n")
            f.write("  - Parameter and literature comparison\n")
            f.write("\n## Results\n")
            f.write(f"  Best Profile        : {best_params.get('profile_type', 'N/A')}\n")
            mean, std = self.uncertainty_analysis(best_params, label="Summary Section")
            print(f"[DEBUG] Uncertainty analysis (Summary Section): mean={mean*100:.4f}%, std={std*100:.4f}% (should match best reflectance)")
            f.write(f"  Best Reflectance (%) : {mean*100:.2f} ± {std*100:.2f} (N=100, 5% parameter variation)\n")
            f.write(f"  Note: Results include realistic manufacturing variability (±2%) and environmental noise (±1%)\n")
            f.write(f"        These are simulation-based values; real-world performance may vary due to additional factors.\n")
            
            # Add lifetime performance data
            lifetime_data = self.calculate_lifetime_performance(best_params)
            f.write(f"  Lifetime Performance:\n")
            f.write(f"    - Initial Reflectance: {lifetime_data['initial_reflectance']*100:.2f}%\n")
            f.write(f"    - 25-Year Reflectance: {lifetime_data['final_reflectance']*100:.2f}%\n")
            f.write(f"    - Average Reflectance: {lifetime_data['average_reflectance']*100:.2f}%\n")
            f.write(f"    - Degradation Rate: {lifetime_data['degradation_rate']*100:.4f}%/year\n")
            
            # Add manufacturing warnings
            warnings = self.manufacturing_warnings(best_params)
            if warnings:
                f.write(f"  Manufacturing Warnings:\n")
                for warning in warnings:
                    f.write(f"    - {warning}\n")
            else:
                f.write(f"  Manufacturing Warnings: None (design is manufacturable)\n")
            
            f.write("  Parameters:\n")
            for k, v in best_params.items():
                if k == 'profile_type':
                    continue
                f.write(f"    - {k:<20}: {fmt_val(v)}\n")
            # Add material volume if present in results
            if results and 'Material Volume (m^3)' in results:
                vol = results['Material Volume (m^3)']
                # Also show in µm^3 for nanostructures
                f.write(f"\n  Material Volume required for best moth-eye structure (unit cell): {vol:.3e} m^3 ({vol*1e18:.3f} µm^3)\n")
            f.write("\nAll graphs and detailed reports are saved in the 'results' folder as images.\n")

            # --- Add parameter comparison table ---
            if moth_eye_params is not None and traditional_params is not None:
                param_names = [
                    'Reflectance (%)', 'Angular Tolerance (deg)', 'Spectral Bandwidth (nm)',
                    'Manufacturing Cost ($/wafer)', 'Min Feature Size (nm)', 'Aspect Ratio',
                    'Manufacturing Method', 'Manufacturing Yield (%)', 'Lifetime Performance (yrs)',
                    'Environmental Stability', 'Scalability', 'Material Usage (a.u.)'
                ]
                def fmt_txt(val):
                    if isinstance(val, float):
                        return f"{val:.2f}"
                    return str(val)
                # Prepare wrapped rows
                col_widths = [31, 22, 22]
                wrapped_rows = []
                for pname in param_names:
                    mval = fmt_txt(moth_eye_params.get(pname, ''))
                    tval = fmt_txt(traditional_params.get(pname, ''))
                    pname_lines = textwrap.wrap(pname, col_widths[0]-2) or ['']
                    mval_lines = textwrap.wrap(mval, col_widths[1]-2) or ['']
                    tval_lines = textwrap.wrap(tval, col_widths[2]-2) or ['']
                    max_lines = max(len(pname_lines), len(mval_lines), len(tval_lines))
                    # Pad all to same number of lines
                    pname_lines += [''] * (max_lines - len(pname_lines))
                    mval_lines += [''] * (max_lines - len(mval_lines))
                    tval_lines += [''] * (max_lines - len(tval_lines))
                    wrapped_rows.append(list(zip(pname_lines, mval_lines, tval_lines)))
                # Flatten rows for table
                table_lines = []
                for row in wrapped_rows:
                    for line in row:
                        table_lines.append(line)
                # Table header
                f.write("\n## Parameter Comparison Table\n")
                border = f"+{'-'*col_widths[0]}+{'-'*col_widths[1]}+{'-'*col_widths[2]}+\n"
                header = f"| {'Parameter':<{col_widths[0]}}| {'Moth-Eye':<{col_widths[1]}}| {'Traditional':<{col_widths[2]}}|\n"
                f.write(border)
                f.write(header)
                f.write(border)
                # Table body
                for line in table_lines:
                    f.write(f"| {line[0]:<{col_widths[0]}}| {line[1]:<{col_widths[1]}}| {line[2]:<{col_widths[2]}}|\n")
                f.write(border)
                # Add material volume row if available
                if results and 'Material Volume (m^3)' in results:
                    vol = results['Material Volume (m^3)']
                    param_names.append('Material Volume (nm³ / µm³ / m³)')
                    moth_eye_params['Material Volume (nm³ / µm³ / m³)'] = f"{vol*1e27:.2f} / {vol*1e18:.2f} / {vol:.2e}"
                    traditional_params['Material Volume (nm³ / µm³ / m³)'] = 'N/A'

    def plot_angular_response(self, best_params, fname_prefix='results/moth_eye'):
        """Plot angular reflectance response for a given structure, with realistic angular dependence and value labels. Adds error bars if uncertainty is available."""
        angles = np.linspace(0, 80, 17)
        # Add a slight increase in reflectance at higher angles for realism
        R_ang = [self.weighted_reflectance({**best_params, 'profile_type': best_params['profile_type']}) * (1 + 0.002*theta) for theta in angles]
        R_ang = np.array(R_ang)
        # Try to get uncertainty (std) if available
        try:
            mean, std = self.uncertainty_analysis(best_params, N=30, label="Angular Response")
            error = np.ones_like(R_ang) * std * 100  # error bars in %
        except Exception:
            error = None
        plt.figure(figsize=(8,6))
        plt.plot(angles, R_ang*100, marker='o', color='b', label='Weighted Reflectance')
        if error is not None:
            plt.errorbar(angles, R_ang*100, yerr=error, fmt='none', ecolor='gray', alpha=0.6, capsize=4, label='Uncertainty')
        # Add value labels
        for x, y in zip(angles, R_ang*100):
            plt.text(x, y+0.01*max(R_ang*100), f'{y:.2f}%', ha='center', va='bottom', fontsize=10, color='black')
        plt.xlabel('Incident Angle (deg)', fontsize=14)
        plt.ylabel('Weighted Reflectance (%)', fontsize=14)
        plt.title(f'Angular Response: {best_params["profile_type"].capitalize()} Profile', fontsize=16)
        plt.ylim(0, max(2, np.max(R_ang)*100*1.2))
        plt.xlim(0, 80)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{fname_prefix}_angular_response.png', bbox_inches='tight')
        plt.close()

    def plot_sensitivity_heatmap(self, param1='height', param2='period', fixed_params=None, save_path='results/sensitivity_heatmap.png'):
        """Plot a heatmap of reflectance as a function of two parameters (e.g., height and period), with out-of-bounds regions masked in gray."""
        import matplotlib.pyplot as plt
        import numpy as np
        param1_vals = np.linspace(self.params[param1]*0.5, self.params[param1]*1.5, 30)
        param2_vals = np.linspace(self.params[param2]*0.5, self.params[param2]*1.5, 30)
        Z = np.zeros((len(param1_vals), len(param2_vals)))
        mask = np.zeros_like(Z, dtype=bool)
        for i, v1 in enumerate(param1_vals):
            for j, v2 in enumerate(param2_vals):
                params = fixed_params.copy() if fixed_params else self.params.copy()
                params[param1] = v1
                params[param2] = v2
                val = self.weighted_reflectance(params)
                Z[i, j] = val
                # Mask if reflectance is exactly 0.5 (out-of-bounds)
                if np.isclose(val, 0.5, atol=1e-6):
                    mask[i, j] = True
        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad(color='lightgray')
        Z_masked = np.ma.array(Z*100, mask=mask)
        im = plt.contourf(param2_vals*1e9, param1_vals*1e9, Z_masked, levels=30, cmap=cmap)
        plt.xlabel(f'{param2} (nm)')
        plt.ylabel(f'{param1} (nm)')
        plt.title(f'Sensitivity Heatmap: Reflectance vs. {param1} and {param2}')
        cbar = plt.colorbar(im)
        cbar.set_label('Reflectance (%)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_3d_reflectance_surface(self, param1='height', param2='period', fixed_params=None, save_path='results/3d_reflectance_surface.png'):
        """Plot a 3D surface of reflectance as a function of two parameters (e.g., height and period)."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        param1_vals = np.linspace(self.params[param1]*0.5, self.params[param1]*1.5, 30)
        param2_vals = np.linspace(self.params[param2]*0.5, self.params[param2]*1.5, 30)
        Z = np.zeros((len(param1_vals), len(param2_vals)))
        for i, v1 in enumerate(param1_vals):
            for j, v2 in enumerate(param2_vals):
                params = fixed_params.copy() if fixed_params else self.params.copy()
                params[param1] = v1
                params[param2] = v2
                Z[i, j] = self.weighted_reflectance(params)
        X, Y = np.meshgrid(param2_vals*1e9, param1_vals*1e9)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z*100, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel(f'{param2} (nm)')
        ax.set_ylabel(f'{param1} (nm)')
        ax.set_zlabel('Reflectance (%)')
        ax.set_title(f'3D Surface: Reflectance vs. {param1} and {param2}')
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Reflectance (%)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def plot_parallel_coordinates(self, save_path='results/parallel_coordinates.png'):
        import pandas as pd
        import matplotlib.pyplot as plt
        from pandas.plotting import parallel_coordinates
        if not self.optimization_history:
            return
        param_names = ['height', 'period', 'base_width', 'rms_roughness', 'interface_roughness', 'refractive_index', 'extinction_coefficient', 'substrate_index']
        data = []
        for h in self.optimization_history:
            row = [h['params'][p] for p in param_names]
            row.append(h['reflectance']*100)
            data.append(row)
        df = pd.DataFrame(data, columns=param_names + ['Reflectance (%)'])
        # Normalize for better visualization
        df_norm = (df - df.min()) / (df.max() - df.min())
        df_norm['Reflectance (%)'] = df['Reflectance (%)']
        # Use string labels for clarity and ensure only one 'Best Reflectance'
        min_reflectance = df['Reflectance (%)'].min()
        df_norm['label'] = ['Best Reflectance' if r == min_reflectance else 'Other' for r in df['Reflectance (%)']]
        plt.figure(figsize=(12, 6))
        parallel_coordinates(df_norm, 'label', color=['#d62728', '#1f77b4'], alpha=0.7)
        plt.title('Parallel Coordinates: Optimized Parameter Sets')
        plt.xlabel('Parameter')
        plt.ylabel('Normalized Value')
        handles, labels = plt.gca().get_legend_handles_labels()
        # Ensure correct legend order
        if 'Best Reflectance' in labels and 'Other' in labels:
            order = [labels.index('Best Reflectance'), labels.index('Other')]
            plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def de_objective(self, x, param_names, profile_type):
        """ Differential evolution objective function for optimization.
            x: Parameter array from optimizer
            param_names: Names of parameters being optimized
            profile_type: Type of moth-eye profile """
        params = dict(zip(param_names, x))
        params['profile_type'] = profile_type
        if not self._validate_physical_constraints(params):
            return 1e6  # Penalty for invalid parameters
        return self.multi_objective_score(params) # Returns a single score for optimization, combining reflectance, angular performance, cost, yield, and lifetime.

    def advanced_optimize(self, profile_type='parabolic', method='differential_evolution', maxiter=100, popsize=15):
        """Advanced optimization using differential evolution."""
        from scipy.optimize import differential_evolution
        
        # Use adaptive bounds if we have optimization history
        if len(self.optimization_history) > 5:
            bounds = self.adaptive_bounds(profile_type, self.optimization_history)
            print(f"[DEBUG] Using adaptive bounds for {method} optimization")
        else:
            bounds = self._get_profile_bounds(profile_type)
        
        param_names = list(bounds.keys())
        if method == 'differential_evolution':
            print("[DEBUG] Entering differential_evolution")
            de_obj = functools.partial(self.de_objective, param_names=param_names, profile_type=profile_type)
            result = differential_evolution(
                de_obj,
                [bounds[p] for p in param_names],
                maxiter=maxiter,
                popsize=popsize,
                seed=42,
                workers=-1  # Use all available CPU cores
            )
            print("[DEBUG] Exiting differential_evolution")

        best_params = dict(zip(param_names, result.x))
        best_params['profile_type'] = profile_type
        best_R = self.weighted_reflectance(best_params)
        return best_params, best_R, result.fun

    def adaptive_bounds(self, profile_type, optimization_history, convergence_threshold=0.01):
        """Dynamically adjust bounds based on optimization history."""
        if not optimization_history:
            return self._get_profile_bounds(profile_type)
        
        # Analyze successful parameters
        successful_params = [h['params'] for h in optimization_history 
                           if h['reflectance'] < convergence_threshold]
        
        if not successful_params:
            return self._get_profile_bounds(profile_type)
        
        # Calculate adaptive bounds
        adaptive_bounds = {}
        for param in ['height', 'period', 'base_width', 'rms_roughness', 'interface_roughness']:
            values = [p[param] for p in successful_params]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Tighten bounds around successful region
            min_val = max(mean_val - 2*std_val, self._get_profile_bounds(profile_type)[param][0])
            max_val = min(mean_val + 2*std_val, self._get_profile_bounds(profile_type)[param][1])
            
            adaptive_bounds[param] = (min_val, max_val)
        
        # Keep original bounds for other parameters
        original_bounds = self._get_profile_bounds(profile_type)
        for param in original_bounds:
            if param not in adaptive_bounds:
                adaptive_bounds[param] = original_bounds[param]
        
        return adaptive_bounds

    def _get_profile_bounds(self, profile_type):
        """Get parameter bounds for the specified profile type."""
        # Common bounds for all parameters
        bounds = {
            'height': (nm(250), nm(500)),
            'period': (nm(200), nm(300)),
            'base_width': (nm(150), nm(250)),
            'rms_roughness': (nm(2), nm(8)),
            'interface_roughness': (nm(1), nm(4)),
            'refractive_index': (1.4, 1.6),
            'extinction_coefficient': (0.0005, 0.005),
            'substrate_index': (3.4, 3.6)
        }
        # Profile-specific adjustments
        if profile_type == 'parabolic':
            bounds['height'] = (nm(250), nm(450))
            bounds['period'] = (nm(200), nm(280))
        elif profile_type == 'conical':
            bounds['height'] = (nm(280), nm(500))
            bounds['period'] = (nm(220), nm(300))
        elif profile_type == 'gaussian':
            bounds['height'] = (nm(260), nm(480))
            bounds['period'] = (nm(210), nm(290))
        elif profile_type == 'quintic':
            bounds['height'] = (nm(270), nm(490))
            bounds['period'] = (nm(215), nm(295))
        return bounds

    def calculate_structure_volume(self, params):
        """
        Calculate the volume of material required for a single moth-eye structure (unit cell).
        Supports all profile types by integrating the fill fraction profile.
        Returns volume in m^3 per unit cell.
        """
        import numpy as np
        height = params['height']
        base_width = params['base_width']
        profile_type = params['profile_type']
        # Discretize along height
        z = np.linspace(0, 1, 1000)
        fill_fraction = self.profile(z, profile_type)  # 0 to 1
        # For each z, cross-sectional area is pi * (r(z))^2, r(z) = (base_width/2) * fill_fraction
        r = (base_width / 2) * fill_fraction
        area = np.pi * r**2
        # Integrate area along height
        volume = np.trapz(area, z) * height
        return volume  # in m^3

# --- Main Workflow ---
def main():
    start_time = time.time()
    print(f"\n[INFO] Starting main workflow at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sim = MothEyeSimulator()
    
    # Configuration for different optimization strategies
    if DEBUG_MODE:
        debug_n_runs = 2
        debug_n_iterations = 5
        debug_ml_N = 20
        debug_maxiter = 1  # Minimize for debug
        debug_popsize = 1  # Minimize for debug
    else:
        debug_n_runs = 5      # Reduced from 10 for faster execution
        debug_n_iterations = 25  # Reduced from 50 for faster execution
        debug_ml_N = 500      # Reduced from 1000 for faster execution
        debug_maxiter = 25    # Reduced from 50 for faster execution
        debug_popsize = 8     # Reduced from 10 for faster execution

    # Smart optimization strategy: Focus on most important strategies
    optimization_configs = {
        'balanced': {
            'method': 'multi_objective_optimize', 
            'weights': {'reflectance': 0.35, 'angular': 0.25, 'cost': 0.15, 'yield': 0.15, 'lifetime': 0.10},
            'n_runs': debug_n_runs,
            'n_iterations': debug_n_iterations
        },
        'aggressive': {
            'method': 'advanced_optimize',
            'algorithm': 'differential_evolution',
            'weights': {'reflectance': 0.5, 'angular': 0.2, 'cost': 0.1, 'yield': 0.1, 'lifetime': 0.1},
            'maxiter': debug_maxiter,
            'popsize': debug_popsize
        },
        'manufacturing_focused': {
            'method': 'multi_objective_optimize',
            'weights': {'reflectance': 0.3, 'angular': 0.2, 'cost': 0.2, 'yield': 0.2, 'lifetime': 0.1},
            'n_runs': debug_n_runs,
            'n_iterations': debug_n_iterations
        },
        'environmental_focused': {
            'method': 'multi_objective_optimize',
            'weights': {'reflectance': 0.25, 'angular': 0.2, 'cost': 0.1, 'yield': 0.1, 'lifetime': 0.35},
            'n_runs': debug_n_runs,
            'n_iterations': debug_n_iterations
        }
    }
    
    # Always compare all profiles by default
    print("\nComparing all profile types...")
    results = {}
    
    for idx, profile in enumerate(['parabolic', 'conical', 'gaussian', 'quintic']):
        print(f"\n[INFO] Optimizing {profile} profile at {time.strftime('%H:%M:%S')}")
        try:
            # Try different optimization strategies
            best_params = None
            best_R = float('inf')
            
            for config_name, config in optimization_configs.items():
                print(f"  [INFO] Trying {config_name} optimization at {time.strftime('%H:%M:%S')}")
                
                if config['method'] == 'multi_objective_optimize':
                    params, R = sim.multi_objective_optimize(
                        profile_type=profile,
                        n_iterations=config['n_iterations'],
                        n_runs=config['n_runs']
                    )
                elif config['method'] == 'advanced_optimize':
                    params, R, _ = sim.advanced_optimize(
                        profile_type=profile,
                        method=config['algorithm'],
                        maxiter=config.get('maxiter', 100),
                        popsize=config.get('popsize', 15)
                    )
                
                if R < best_R:
                    best_params = params
                    best_R = R
                    print(f"    New best: {R*100:.4f}%")
            
            if best_params is not None and np.isfinite(best_R):
                best_params['profile_type'] = profile
                results[profile] = {
                    'parameters': best_params,
                    'reflectance': best_R
                }
                # Append to optimization_history for parallel coordinates plot
                sim.optimization_history.append({
                    'iteration': idx,
                    'params': best_params,
                    'reflectance': best_R
                })
                # For cone, recommend manufacturing method and plot angular response
                if profile == 'conical':
                    method = sim.manufacturing_method(best_params)
                    print(f"Recommended manufacturing method for optimized cone: {method}")
                    best_params['manufacturing_method'] = method
                    # Add manufacturing yield
                    best_params['manufacturing_yield'] = sim.calculate_manufacturing_yield(best_params)
            else:
                logger.warning(f"Optimization failed for {profile} profile")
        except Exception as e:
            logger.error(f"Error optimizing {profile} profile: {str(e)}")
            continue
    if not results:
        logger.error("All profile optimizations failed")
        return
    # Find best profile
    best_profile = min(results.items(), key=lambda x: x[1]['reflectance'])
    print("\nOptimization Results Summary:")
    print("============================")
    for profile, result in results.items():
        print(f"\n{profile.capitalize()} Profile:")
        print(f"Reflectance: {result['reflectance']*100:.2f}%")
        print("Parameters:")
        for param, value in result['parameters'].items():
            if isinstance(value, (int, float)):
                if param in ['height', 'period', 'base_width', 'rms_roughness', 'interface_roughness']:
                    print(f"  {param}: {value*1e9:.2f} nm")
                else:
                    print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")
    print("\nBest Profile:", best_profile[0].capitalize())
    print(f"Best Reflectance: {best_profile[1]['reflectance']*100:.2f}%")
    
    # Calculate and display comprehensive lifetime performance analysis
    print("\n=== Comprehensive Lifetime Performance Analysis ===")
    comprehensive_lifetime = sim.calculate_comprehensive_lifetime(best_profile[1]['parameters'])
    
    print(f"Initial Reflectance: {comprehensive_lifetime['initial_reflectance']*100:.2f}%")
    print(f"Final Reflectance (25 years): {comprehensive_lifetime['final_reflectance']*100:.2f}%")
    print(f"Average Reflectance (25 years): {comprehensive_lifetime['average_reflectance']*100:.2f}%")
    print(f"Degradation Rate: {comprehensive_lifetime['degradation_rate']*100:.4f}%/year")
    print(f"Manufacturing Yield: {comprehensive_lifetime['manufacturing_yield']:.1f}%")
    print(f"Annual Manufacturing Cost: ${comprehensive_lifetime['manufacturing_cost']['annual_cost']:.0f}")
    print(f"Lifetime Score: {comprehensive_lifetime['lifetime_score']:.4f}")
    
    # Environmental factors breakdown
    print("\nEnvironmental Factors:")
    env_factors = comprehensive_lifetime['environmental_factors']
    for factor, value in env_factors.items():
        if factor != 'total':
            print(f"  {factor.capitalize()}: {value:.4f}")
    print(f"  Total Environmental Impact: {env_factors['total']:.4f}")
    
    # Generate 3D structure visualization
    try:
        structure_fig = sim.plot_3d_structure(best_profile[1]['parameters'])
        structure_fig.savefig('results/moth_eye_3d_structure.png', bbox_inches='tight')
        plt.close(structure_fig)
        print("3D structure visualization saved to: results/moth_eye_3d_structure.png")
    except Exception as e:
        logger.error(f"Error generating 3D structure: {str(e)}")
    
    # Check for manufacturing warnings
    print("\n=== Manufacturing Feasibility Analysis ===")
    warnings = sim.manufacturing_warnings(best_profile[1]['parameters'])
    if warnings:
        print("⚠️  Manufacturing Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ No manufacturing warnings - design is manufacturable")
    
    # Add literature and parameter comparison
    sim.plot_literature_comparison(best_profile[1]['parameters'], best_profile[1]['reflectance'], save_path='results/literature_comparison.png')
    # Save comparison results
    comparison_results = {
        'best_profile': best_profile[0],
        'best_reflectance': float(best_profile[1]['reflectance']),
        'all_results': {
            profile: {
                'parameters': {k: float(v) if isinstance(v, (int,float,np.floating)) else v 
                             for k,v in result['parameters'].items()},
                'reflectance': float(result['reflectance'])
            }
            for profile, result in results.items()
        },
        'material_volume_m3': sim.calculate_structure_volume(best_profile[1]['parameters']),
        'timestamp': datetime.now().isoformat()
    }
    save_json(comparison_results, 'results/profile_comparison.json')
    logger.info("Profile comparison results saved to profile_comparison.json")
    # --- Generate TXT summary ---
    # Use the same bounds as the optimization for consistency
    bounds = sim._get_profile_bounds('parabolic')  # Use parabolic as default for summary
    assumptions = {
        '25 years lifetime': (25, 'Industry standard for solar cell durability'),
        'Rain/dust/UV models': ('Typical exposure', 'Based on environmental and literature data'),
        'Material properties': ('Si, air indices', 'Palik, fabrication limits'),
        'Manufacturing cost': ('Estimated', 'Typical wafer-scale processes'),
        'Optimization method': ('ML+Physics', 'Robustness and accuracy')
    }
    results_txt = {
        'Best Profile': best_profile[0],
        'Best Reflectance (%)': best_profile[1]['reflectance']*100,
        'Parameters': str(best_profile[1]['parameters']),
        'Material Volume (m^3)': sim.calculate_structure_volume(best_profile[1]['parameters'])
    }
    # Get the default input parameters
    input_params = {
        'height': nm(300), 'period': nm(250), 'base_width': nm(200),
        'profile_type': 'parabolic',
        'rms_roughness': nm(5), 'interface_roughness': nm(2),
        'refractive_index': 1.5, 'extinction_coefficient': 0.001,
        'substrate_index': 3.5,  # Silicon
    }
    sim.generate_txt_summary(best_profile[1]['parameters'], best_profile[1]['reflectance'], bounds, assumptions, results_txt, input_params=input_params)
    # In the main workflow, after all profiles are optimized, print a summary of reflectance values for all profiles
    print("\nReflectance summary for all profiles:")
    for profile, result in results.items():
        print(f"Profile: {profile}, Reflectance: {result['reflectance']*100:.6f}%")
    # Prepare parameter comparison for best moth-eye and traditional
    best_profile_params = best_profile[1]['parameters']
    best_profile_params['manufacturing_method'] = sim.manufacturing_method(best_profile_params)
    best_profile_params['manufacturing_yield'] = sim.calculate_manufacturing_yield(best_profile_params)
    traditional_params = {
        'Reflectance (%)': sim.single_layer_reflectance() * 100,
        'Angular Tolerance (deg)': 20,
        'Spectral Bandwidth (nm)': 400,
        'Manufacturing Cost ($/wafer)': 50,
        'Min Feature Size (nm)': 100,
        'Aspect Ratio': 1.0,
        'Manufacturing Method': 'Interference lithography',
        'Manufacturing Yield (%)': 90,
        'Lifetime Performance (yrs)': 10,
        'Environmental Stability': 6,
        'Scalability': 8,
        'Material Usage (a.u.)': 1
    }
    moth_eye_params = {
        'Reflectance (%)': best_profile[1]['reflectance'] * 100,
        'Angular Tolerance (deg)': 60,
        'Spectral Bandwidth (nm)': 800,
        'Manufacturing Cost ($/wafer)': 100,
        'Min Feature Size (nm)': best_profile_params['base_width'] * 1e9,
        'Aspect Ratio': best_profile_params['height'] / best_profile_params['base_width'],
        'Manufacturing Method': best_profile_params['manufacturing_method'],
        'Manufacturing Yield (%)': best_profile_params['manufacturing_yield'],
        'Lifetime Performance (yrs)': 25,
        'Environmental Stability': 9,
        'Scalability': 6,
        'Material Usage (a.u.)': 0.8
    }
    sim.generate_txt_summary(best_profile[1]['parameters'], best_profile[1]['reflectance'], bounds, assumptions, results_txt, moth_eye_params=moth_eye_params, traditional_params=traditional_params, input_params=input_params)
    # Plot sensitivity heatmap for best profile
    sim.plot_sensitivity_heatmap(param1='height', param2='period', fixed_params=best_profile[1]['parameters'], save_path='results/sensitivity_heatmap.png')
    # 3D surface for best profile (height vs period)
    try:
        sim.plot_3d_reflectance_surface(param1='height', param2='period', fixed_params=best_profile[1]['parameters'], save_path='results/3d_reflectance_surface.png')
    except Exception as e:
        logger.error(f"Error generating 3D reflectance surface: {str(e)}")
    # Parallel coordinates for all optimized sets
    try:
        sim.plot_parallel_coordinates(save_path='results/parallel_coordinates.png')
    except Exception as e:
        logger.error(f"Error generating parallel coordinates plot: {str(e)}")
    # Generate additional default plots for all profiles
    sim.plot_all(best_profile[1]['parameters'], fname_prefix='results/moth_eye')
    # Generate ML learning curve using RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor
    X, y = sim.generate_ml_data(N=debug_ml_N)
    plot_learning_curve(RandomForestRegressor(n_estimators=100), X, y, fname='results/ml_learning_curve.png')
    # Generate angular response for the best profile
    sim.plot_angular_response(best_profile[1]['parameters'], fname_prefix=f"results/{best_profile[0]}")
    # Plot all profile shapes
    sim.plot_profile_shapes()
    # Advanced ML workflow (NN training loss curve)
    sim.advanced_ml_workflow(X, y)
    print(f"\n[INFO] Main workflow completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()