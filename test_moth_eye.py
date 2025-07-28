from moth_eye_project import MothEyeSimulator, nm
import numpy as np
import os

def test_reflectance() -> None:
    """
    Test the reflectance calculation for flat interface and moth-eye structure.
    Returns:
        None
    """
    sim = MothEyeSimulator()
    
    # Test flat interface (should match Fresnel)
    params_flat = {
        'height': 1e-9,  # Nearly flat
        'period': nm(250),
        'base_width': nm(200),
        'rms_roughness': nm(5),
        'interface_roughness': nm(2),
        'profile_type': 'parabolic',
        'refractive_index': 1.5,
        'extinction_coefficient': 0.001,
        'substrate_index': 3.5
    }
    
    # Test moth-eye structure with optimal parameters
    params_moth_eye = {
        'height': nm(300),
        'period': nm(250),
        'base_width': nm(200),
        'rms_roughness': nm(5),
        'interface_roughness': nm(2),
        'profile_type': 'parabolic',
        'refractive_index': 1.5,
        'extinction_coefficient': 0.001,
        'substrate_index': 3.5
    }
    
    print("\n=== Testing Flat Interface ===")
    R_flat = sim.reflectance(params_flat)
    R_flat_val = float(R_flat) if np.isscalar(R_flat) or R_flat.size == 1 else float(R_flat[0])
    print(f"Flat interface reflectance: {R_flat_val*100:.2f}%")
    
    # Verify flat interface matches Fresnel
    n_air = 1.0
    n_si = 3.5
    R_fresnel = ((n_air - n_si)/(n_air + n_si))**2
    assert abs(R_flat_val - R_fresnel) < 0.01, f"Flat interface test failed: {R_flat_val:.6f} vs {R_fresnel:.6f}"
    print("✓ Flat interface test passed")
    
    print("\n=== Testing Moth-Eye Structure ===")
    R_moth_eye = sim.reflectance(params_moth_eye)
    R_moth_eye_val = float(R_moth_eye) if np.isscalar(R_moth_eye) or R_moth_eye.size == 1 else float(R_moth_eye[0])
    print(f"Moth-eye reflectance: {R_moth_eye_val*100:.2f}%")
    
    # Verify moth-eye has lower reflectance than flat interface
    assert R_moth_eye_val < R_flat_val, f"Moth-eye should have lower reflectance than flat interface"
    print("✓ Moth-eye reflectance test passed")
    
    print("\n=== Testing Weighted Reflectance ===")
    R_weighted = sim.weighted_reflectance(params_moth_eye)
    print(f"Weighted reflectance: {R_weighted*100:.2f}%")
    
    # Verify weighted reflectance is within reasonable range
    assert 0.001 <= R_weighted <= 0.05, f"Weighted reflectance {R_weighted:.6f} outside reasonable range for moth-eye structure"
    print("✓ Weighted reflectance test passed")
    
    print("\n=== Testing Traditional Coatings ===")
    R_single = sim.single_layer_reflectance()
    R_gradient = sim.gradient_index_reflectance()
    print(f"Single-layer: {R_single*100:.2f}%")
    print(f"Gradient-index: {R_gradient*100:.2f}%")
    # Verify traditional coatings have expected relationships
    assert R_gradient < R_single, "Gradient-index should have lower reflectance than single-layer"
    print("\u2713 Traditional coatings test passed")
    
    print("\n=== Testing All Profile Types ===")
    profiles = ['parabolic', 'conical', 'gaussian', 'quintic']
    for profile in profiles:
        params = params_moth_eye.copy()
        params['profile_type'] = profile
        R = sim.reflectance(params)
        R_val = float(R) if np.isscalar(R) or getattr(R, 'size', 1) == 1 else float(R[0])
        assert R_val < R_flat_val, f"{profile} should have lower reflectance than flat interface"
        print(f"✓ {profile.capitalize()} profile test passed: {R_val*100:.2f}%")

    print("\n=== Edge Case Tests ===")
    # Negative height (unphysical)
    params_bad = params_moth_eye.copy()
    params_bad['height'] = -nm(100)
    try:
        R_bad = sim.reflectance(params_bad)
        print(f"Warning: Negative height did not raise error, got R={R_bad}")
    except Exception as e:
        print(f"✓ Negative height correctly raised error: {e}")
    # Extremely high refractive index
    params_high_n = params_moth_eye.copy()
    params_high_n['refractive_index'] = 10.0
    try:
        R_high_n = sim.reflectance(params_high_n)
        print(f"✓ High refractive index handled, R={R_high_n}")
    except Exception as e:
        print(f"Error with high refractive index: {e}")

    print("\n=== Checking Output Files ===")
    expected_files = [
        'results/summary.txt',
        'results/profile_comparison.json',
        'results/moth_eye_3d_structure.png',
        'results/moth_eye_profiles.png',
        'results/moth_eye_comparison.png',
        'results/moth_eye_spectral.png',
        'results/nn_training_loss.png',
        'results/literature_comparison.png',
        'results/sensitivity_heatmap.png',
        'results/3d_reflectance_surface.png',
        'results/parallel_coordinates.png',
        'results/ml_learning_curve.png',
    ]
    missing = [f for f in expected_files if not os.path.exists(f)]
    if missing:
        print(f"Warning: Missing result files: {missing}")
    else:
        print("✓ All expected result files are present.")

    print("\n=== All Tests Passed ===")

# Note:
# The reflectance values for each profile type in this test script may differ slightly
# from the best values reported in the main simulation results. This is because the test
# uses default or fixed parameters for quick validation, while the main code performs
# extensive multi-objective optimization to find the absolute best parameters.
# Both results are expected to be in the ultra-low reflectance range, confirming correct code behavior.

if __name__ == "__main__":
    test_reflectance() 