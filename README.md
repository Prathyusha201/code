# Moth-Eye Anti-Reflection Coating Simulation

## Overview
This project simulates and optimizes traditional and moth-eye nanostructured anti-reflection coatings for solar cells. It includes advanced optical modeling, multi-objective optimization, manufacturability assessment, and automated result generation.

## Setup
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Use a virtual environment for reproducibility.

## Usage
Run the main simulation and optimization:
```bash
python moth_eye_project.py
```
All results, plots, and summary files will be saved in the `results/` folder.

## Testing
Run the test script to verify correctness:
```bash
python test_moth_eye.py
```

## Reproducibility
- All dependencies are listed in `requirements.txt`.
- For best results, use the specified versions.

## Project Structure
- `moth_eye_project.py`: Main simulation and optimization code
- `materials.py`, `solar_spectrum.py`: Supporting modules
- `test_moth_eye.py`: Automated tests
- `results/`: Output folder for all results and plots

## Contact
For questions, contact Prathyusha Murali Mohan Raju

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

## Limitations and Future Work
- No experimental fabrication or physical noise injection; results are simulation-based only.
- Environmental effects (humidity, dust, UV) are modeled based on literature, not explicitly simulated.
- Economic analysis is limited to manufacturing cost estimation; no full lifecycle or market analysis.
- No integration with commercial solar cell manufacturing lines.
- Only silicon-based solar cells are considered; other materials are not studied in depth.
- Future work could include experimental validation, more advanced scattering models, and integration with commercial manufacturing workflows. 

---

This work was completed as part of the MEng final year project at DCU. For questions or collaboration, please contact the author.

# Notes on Simulation Realism and Uncertainty

- **Idealized Results:** All results (reflectance, yield, lifetime) are based on simulation and assume ideal manufacturing and environmental conditions. In real-world fabrication, some deviation is expected due to process variation, defects, and environmental degradation.
- **Manufacturing Yield:** The reported 100% manufacturing yield is an idealized value, standard for simulation studies. Actual yield in production may be lower due to defects or process limitations.
- **Lifetime Performance:** Lifetime reflectance is reported as approaching zero in simulation, which assumes no unmodeled degradation. Real-world performance may degrade over time due to contamination, weathering, or unforeseen factors.
- **Uncertainty:** The uncertainty values (e.g., ±0.01%) are calculated using Monte Carlo analysis with parameter perturbation, not hardcoded. This provides a realistic estimate of expected variability in performance.
- **Recommendation:** For academic and simulation purposes, these assumptions are standard and accepted. For real-world deployment, additional safety margins and empirical validation are recommended.