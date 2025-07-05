# Moth-Eye Anti-Reflection Coating Simulation & Optimization

This project implements a comprehensive, publication-grade simulation and optimization framework for moth-eye anti-reflection coatings, inspired by natural nanostructures, for advanced solar cell applications. The framework models optical, environmental, and manufacturing properties, and leverages both physics-based and machine learning-guided optimization.

## Features

- Full electromagnetic simulation of moth-eye nanostructure arrays
- Multiple profile geometries: conical, parabolic, gaussian, and quintic
- Solar spectrum integration for real-world performance evaluation
- Advanced optimization algorithms (differential evolution, multi-objective, ML-guided)
- Machine learning models for rapid parameter prediction and optimization
- Comprehensive visualization and analysis tools
- Angular and spectral dependence modeling
- Environmental and lifetime performance analysis
- Manufacturing feasibility, cost, and scalability assessment
- Automated report and publication-quality figure generation

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Main Simulation
Run the main simulation and optimization:
```bash
python moth_eye_project.py
```
This will:
1. Initialize the simulator with default parameters
2. Run multi-profile optimization (parabolic, conical, gaussian, quintic)
3. Generate performance and manufacturing analysis
4. Create publication-ready plots and reports in the `results/` directory

**Note:** All results are generated automatically. No user input is required.

### Testing
Run the test suite to verify core simulation and optimization routines:
```bash
python test_moth_eye.py
```

## Project Structure

- `moth_eye_project.py`: Core simulation, optimization, and analysis framework
- `ml_models.py`: Machine learning models and utilities
- `solar_spectrum.py`: Solar spectrum data handling
- `materials.py`: Material property interpolation and management
- `validation.py`: Literature comparison and validation tools
- `test_moth_eye.py`: Automated tests for simulation and optimization
- `requirements.txt`: Project dependencies
- `results/`: Output directory for all figures, tables, and reports
- `data/`: Material and spectrum data files

## References

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

## Author

Prathyusha Murali Mohan Raju

---

This work was completed as part of the MEng final year project at DCU. For questions or collaboration, please contact the author.