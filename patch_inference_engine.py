#!/usr/bin/env python3
"""
Patch to fix the inference engine confidence calculation issue.
This replaces the _calculate_confidence_scores method.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Read the current inference_engine.py
inference_path = project_root / "src" / "ml_pipeline" / "inference_engine.py"
with open(inference_path, 'r') as f:
    content = f.read()

# Define the new method
new_method = '''    def _calculate_confidence_scores(self, predictions: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions with proper handling."""
        try:
            # Ensure predictions is numpy array
            predictions = np.atleast_1d(predictions)
            
            # Method 1: Based on prediction magnitude
            # Map predictions to confidence: higher absolute predictions = higher confidence
            # But cap at reasonable values
            abs_predictions = np.abs(predictions)
            max_pred = 0.3  # Expected max prediction magnitude
            magnitude_confidence = np.clip(abs_predictions / max_pred, 0.0, 1.0)
            
            # Method 2: Based on feature quality
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Check for NaN or extreme values in features
            nan_ratio = np.isnan(features).sum(axis=1) / features.shape[1]
            extreme_ratio = ((np.abs(features) > 10).sum(axis=1) / features.shape[1])
            feature_quality = 1.0 - np.clip(nan_ratio + extreme_ratio * 0.5, 0.0, 1.0)
            
            # Method 3: Prediction reasonableness
            # Penalize extreme predictions
            extreme_penalty = np.where(abs_predictions > 0.5, 0.5, 1.0)
            
            # For single predictions, don't use consistency score
            if len(predictions) == 1:
                # Use only magnitude and feature quality
                confidence_scores = (magnitude_confidence * 0.6 + 
                                   feature_quality * 0.3 + 
                                   extreme_penalty * 0.1)
            else:
                # For batch predictions, can use consistency
                median_pred = np.median(predictions)
                std_pred = np.std(predictions) if len(predictions) > 1 else 0.1
                consistency = 1.0 - np.abs(predictions - median_pred) / (std_pred + 0.1)
                consistency = np.clip(consistency, 0.0, 1.0)
                
                confidence_scores = (magnitude_confidence * 0.4 + 
                                   feature_quality * 0.3 + 
                                   consistency * 0.2 +
                                   extreme_penalty * 0.1)
            
            # Apply sigmoid to smooth confidence scores
            # This prevents always getting 0% or 100%
            confidence_scores = 1 / (1 + np.exp(-4 * (confidence_scores - 0.5)))
            
            # Final clipping to reasonable range
            confidence_scores = np.clip(confidence_scores, 0.1, 0.9)
            
            return confidence_scores
            
        except Exception as e:
            logger.warning("Error calculating confidence scores", exception=e)
            return np.ones(len(predictions)) * 0.5  # Default medium confidence'''

# Find the start and end of the current method
start_marker = "    def _calculate_confidence_scores(self, predictions: np.ndarray, features: np.ndarray) -> np.ndarray:"
end_marker = "            return np.ones(len(predictions)) * 0.5  # Default medium confidence"

start_idx = content.find(start_marker)
if start_idx == -1:
    print("ERROR: Could not find _calculate_confidence_scores method")
    sys.exit(1)

# Find the end of the method
end_idx = content.find(end_marker, start_idx)
if end_idx == -1:
    print("ERROR: Could not find end of _calculate_confidence_scores method")
    sys.exit(1)

# Include the full line
end_idx = content.find('\n', end_idx) + 1

# Replace the method
new_content = content[:start_idx] + new_method + content[end_idx:]

# Write back
with open(inference_path, 'w') as f:
    f.write(new_content)

print("✅ Successfully patched inference_engine.py")
print("✅ Fixed confidence calculation that was always returning 100%")