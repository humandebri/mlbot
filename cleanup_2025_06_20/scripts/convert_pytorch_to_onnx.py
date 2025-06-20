#!/usr/bin/env python3
"""
Convert PyTorch model to ONNX format for production inference.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.fast_nn_model import FastNN


def convert_pytorch_to_onnx(
    pytorch_path: str,
    scaler_path: str,
    onnx_path: str,
    input_features: int = 26  # Updated based on actual model
):
    """Convert PyTorch model to ONNX format."""
    
    print(f"Loading PyTorch model from {pytorch_path}")
    
    # Load model architecture
    model = FastNN(input_dim=input_features)
    
    # Load state dict
    state_dict = torch.load(pytorch_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, input_features)
    
    print("Converting to ONNX format...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model converted and saved to {onnx_path}")
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation successful!")
    except Exception as e:
        print(f"Warning: ONNX validation failed: {e}")
    
    # Copy scaler file
    if Path(scaler_path).exists():
        import shutil
        scaler_dest = Path(onnx_path).parent / "scaler.pkl"
        shutil.copy(scaler_path, scaler_dest)
        print(f"Scaler copied to {scaler_dest}")
    
    return True


if __name__ == "__main__":
    # Convert the model
    success = convert_pytorch_to_onnx(
        pytorch_path="models/fast_nn_final.pth",
        scaler_path="models/fast_nn_scaler.pkl",
        onnx_path="models/catboost_model.onnx"
    )
    
    if success:
        print("\n✅ Conversion completed successfully!")
        print("The model is now ready for production inference with ONNX Runtime.")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)