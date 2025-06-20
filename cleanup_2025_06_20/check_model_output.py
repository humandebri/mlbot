#!/usr/bin/env python3
"""
Check ONNX model output structure
"""

import onnxruntime as ort
import numpy as np

def check_model():
    """Check model output structure."""
    # Load model
    model_path = "models/v3.1_improved/model.onnx"
    session = ort.InferenceSession(model_path)
    
    # Get model info
    print("Model inputs:")
    for input in session.get_inputs():
        print(f"  Name: {input.name}")
        print(f"  Shape: {input.shape}")
        print(f"  Type: {input.type}")
    
    print("\nModel outputs:")
    for output in session.get_outputs():
        print(f"  Name: {output.name}")
        print(f"  Shape: {output.shape}")
        print(f"  Type: {output.type}")
    
    # Test with different inputs
    print("\nTesting with different inputs:")
    
    # Test 1: Random normal
    test_input = np.random.randn(1, 44).astype(np.float32)
    result = session.run(None, {session.get_inputs()[0].name: test_input})
    print(f"\nRandom normal input result: {result}")
    
    # Test 2: All zeros
    test_input = np.zeros((1, 44), dtype=np.float32)
    result = session.run(None, {session.get_inputs()[0].name: test_input})
    print(f"All zeros input result: {result}")
    
    # Test 3: All ones
    test_input = np.ones((1, 44), dtype=np.float32)
    result = session.run(None, {session.get_inputs()[0].name: test_input})
    print(f"All ones input result: {result}")
    
    # Test 4: Large values
    test_input = np.ones((1, 44), dtype=np.float32) * 5
    result = session.run(None, {session.get_inputs()[0].name: test_input})
    print(f"Large values input result: {result}")
    
    # Test 5: Multiple samples
    test_input = np.random.randn(5, 44).astype(np.float32)
    result = session.run(None, {session.get_inputs()[0].name: test_input})
    print(f"\nMultiple samples result: {result}")
    
    # Check if it's a classifier with probability output
    print("\nChecking for probability outputs...")
    
    # Try different patterns that might trigger different outputs
    patterns = [
        np.array([[1]*22 + [-1]*22], dtype=np.float32),  # Half positive, half negative
        np.array([[-1]*22 + [1]*22], dtype=np.float32),  # Opposite pattern
        np.linspace(-5, 5, 44).reshape(1, -1).astype(np.float32),  # Gradient
        np.random.uniform(-1, 1, (1, 44)).astype(np.float32),  # Uniform random
    ]
    
    for i, pattern in enumerate(patterns):
        result = session.run(None, {session.get_inputs()[0].name: pattern})
        print(f"Pattern {i+1} result: {result}")

if __name__ == "__main__":
    check_model()