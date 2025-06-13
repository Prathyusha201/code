from moth_eye_project import MothEyeSimulator, nm
import numpy as np

def test_reflectance():
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
    R_flat = sim.reflectance(params_flat, debug=True)
    R_flat_val = float(R_flat) if np.isscalar(R_flat) or R_flat.size == 1 else float(R_flat[0])
    print(f"Flat interface reflectance: {R_flat_val*100:.2f}%")
    
    # Verify flat interface matches Fresnel
    n_air = 1.0
    n_si = 3.5
    R_fresnel = ((n_air - n_si)/(n_air + n_si))**2
    assert abs(R_flat_val - R_fresnel) < 0.01, f"Flat interface test failed: {R_flat_val:.6f} vs {R_fresnel:.6f}"
    print("✓ Flat interface test passed")
    
    print("\n=== Testing Moth-Eye Structure ===")
    R_moth_eye = sim.reflectance(params_moth_eye, debug=True)
    R_moth_eye_val = float(R_moth_eye) if np.isscalar(R_moth_eye) or R_moth_eye.size == 1 else float(R_moth_eye[0])
    print(f"Moth-eye reflectance: {R_moth_eye_val*100:.2f}%")
    
    # Verify moth-eye has lower reflectance than flat interface
    assert R_moth_eye_val < R_flat_val, f"Moth-eye should have lower reflectance than flat interface"
    print("✓ Moth-eye reflectance test passed")
    
    print("\n=== Testing Weighted Reflectance ===")
    R_weighted = sim.weighted_reflectance(params_moth_eye, debug=True)
    print(f"Weighted reflectance: {R_weighted*100:.2f}%")
    
    # Verify weighted reflectance is within reasonable range
    assert 0.01 <= R_weighted <= 0.30, f"Weighted reflectance {R_weighted:.6f} outside reasonable range"
    print("✓ Weighted reflectance test passed")
    
    print("\n=== Testing Traditional Coatings ===")
    R_single = sim.single_layer_reflectance()
    R_double = sim.double_layer_reflectance()
    R_gradient = sim.gradient_index_reflectance()
    print(f"Single-layer: {R_single*100:.2f}%")
    print(f"Double-layer: {R_double*100:.2f}%")
    print(f"Gradient-index: {R_gradient*100:.2f}%")
    
    # Verify traditional coatings have expected relationships
    assert R_double < R_single, "Double-layer should have lower reflectance than single-layer"
    assert R_gradient < R_single, "Gradient-index should have lower reflectance than single-layer"
    print("✓ Traditional coatings test passed")
    
    print("\n=== All Tests Passed ===")

if __name__ == "__main__":
    test_reflectance() 