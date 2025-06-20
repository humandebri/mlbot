"""
Fix inference engine confidence calculation that always returns 100%.
"""

import numpy as np

def fixed_calculate_confidence_scores(predictions: np.ndarray, features: np.ndarray) -> np.ndarray:
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
        print(f"Error calculating confidence scores: {e}")
        # Return moderate confidence on error
        return np.ones(len(predictions)) * 0.5


# Test the function
if __name__ == "__main__":
    # Test single prediction
    single_pred = np.array([0.12])
    features = np.random.randn(44)
    conf = fixed_calculate_confidence_scores(single_pred, features)
    print(f"Single prediction: {single_pred[0]:.4f}, confidence: {conf[0]:.2%}")
    
    # Test batch predictions
    batch_pred = np.array([0.12, 0.15, 0.10, 0.13])
    batch_features = np.random.randn(4, 44)
    batch_conf = fixed_calculate_confidence_scores(batch_pred, batch_features)
    print(f"Batch predictions: {batch_pred}, confidences: {batch_conf}")
    
    # Test extreme prediction
    extreme_pred = np.array([0.8])
    extreme_conf = fixed_calculate_confidence_scores(extreme_pred, features)
    print(f"Extreme prediction: {extreme_pred[0]:.4f}, confidence: {extreme_conf[0]:.2%}")