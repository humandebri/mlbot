#!/usr/bin/env python3
"""
Train LightGBM model for liquidation prediction.

Usage:
    python scripts/train_lightgbm.py --data-path data/features.parquet
"""

import asyncio
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.ml_pipeline.lightgbm_trainer import LightGBMTrainer
from src.ml_pipeline.data_preprocessing import DataPreprocessor
from src.ml_pipeline.model_evaluator import ModelEvaluator
from src.storage.duckdb_manager import DuckDBManager

setup_logging()
logger = get_logger(__name__)


async def prepare_training_data(min_liquidations: int = 100):
    """
    Prepare training data from collected market data.
    
    Args:
        min_liquidations: Minimum liquidation events required
        
    Returns:
        Tuple of (features_df, labels_df)
    """
    storage = DuckDBManager()
    
    try:
        # Check data availability
        summary = await storage.get_data_summary()
        
        total_liquidations = sum(
            summary['data_types'].get('liquidation', 0) 
            for _ in summary['symbols']
        )
        
        if total_liquidations < min_liquidations:
            logger.error(
                f"Insufficient liquidations: {total_liquidations} < {min_liquidations}. "
                "Please collect more data."
            )
            return None, None
        
        logger.info(f"Found {total_liquidations} liquidation events")
        
        # Load feature data
        features_df = await storage.load_features_for_training()
        
        if features_df is None or len(features_df) == 0:
            logger.error("No feature data found")
            return None, None
        
        # Create labels (expected PnL)
        labels_df = await storage.create_training_labels(
            lookahead_seconds=60,
            barrier_pct=0.002
        )
        
        logger.info(
            f"Prepared training data: {len(features_df)} samples, "
            f"{len(features_df.columns)} features"
        )
        
        return features_df, labels_df
        
    finally:
        await storage.close()


def train_model(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train LightGBM model with time series split.
    
    Args:
        features_df: Feature DataFrame
        labels_df: Labels DataFrame
        test_size: Test set size
        
    Returns:
        Training results
    """
    # Time series split (don't shuffle!)
    split_idx = int(len(features_df) * (1 - test_size))
    
    X_train = features_df.iloc[:split_idx]
    y_train = labels_df.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    y_test = labels_df.iloc[split_idx:]
    
    logger.info(
        f"Train/Test split: {len(X_train)}/{len(X_test)} samples"
    )
    
    # Data preprocessing
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor({
        "enable_fractional_diff": True,
        "enable_binning": True,
        "enable_meta_labeling": False  # For first stage
    })
    
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train LightGBM
    logger.info("Training LightGBM model...")
    trainer = LightGBMTrainer({
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "num_threads": -1,
        "device": "cpu",  # Change to "gpu" if available
        "force_col_wise": True,
        "enable_hyperopt": True,
        "hyperopt_trials": 50
    })
    
    results = trainer.train(X_train_processed, y_train, X_test_processed, y_test)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator()
    
    train_pred = trainer.predict(X_train_processed)
    test_pred = trainer.predict(X_test_processed)
    
    eval_results = {
        'train_metrics': evaluator.evaluate_predictions(y_train, train_pred),
        'test_metrics': evaluator.evaluate_predictions(y_test, test_pred),
        'feature_importance': results['feature_importance'],
        'training_time': results['training_time'],
        'model_size_mb': results['model_size_mb']
    }
    
    # Trading simulation
    logger.info("Running trading simulation...")
    trading_results = evaluator.simulate_trading(
        predictions=test_pred,
        actual=y_test.values,
        timestamps=X_test.index,
        initial_capital=10000,
        position_size=0.1,
        fee_rate=0.00055
    )
    
    eval_results['trading_simulation'] = trading_results
    
    # Save model
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = project_root / "models" / f"lightgbm_{model_version}"
    model_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(model_path / "model.txt"))
    
    # Save preprocessor
    import joblib
    joblib.dump(preprocessor, model_path / "preprocessor.pkl")
    
    # Save results
    with open(model_path / "results.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    
    logger.info(f"Model saved to {model_path}")
    
    # Convert to ONNX for production
    logger.info("Converting to ONNX format...")
    trainer.convert_to_onnx(
        X_test_processed.iloc[:10],  # Sample for shape
        str(model_path / "model.onnx")
    )
    
    return eval_results


def print_results(results: Dict[str, Any]):
    """Print training results summary."""
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    # Model metrics
    print("\nModel Performance:")
    print(f"  Train RMSE: {results['train_metrics']['rmse']:.6f}")
    print(f"  Test RMSE:  {results['test_metrics']['rmse']:.6f}")
    print(f"  Train MAE:  {results['train_metrics']['mae']:.6f}")
    print(f"  Test MAE:   {results['test_metrics']['mae']:.6f}")
    
    # Training stats
    print(f"\nTraining Time: {results['training_time']:.1f} seconds")
    print(f"Model Size: {results['model_size_mb']:.1f} MB")
    
    # Feature importance (top 10)
    print("\nTop 10 Important Features:")
    importance_df = pd.DataFrame(results['feature_importance'])
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.1f}")
    
    # Trading simulation
    if 'trading_simulation' in results:
        sim = results['trading_simulation']
        print("\nTrading Simulation Results:")
        print(f"  Total Return: {sim['total_return']:.2%}")
        print(f"  Sharpe Ratio: {sim['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {sim['max_drawdown']:.2%}")
        print(f"  Win Rate: {sim['win_rate']:.2%}")
        print(f"  Total Trades: {sim['total_trades']}")


async def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument(
        "--data-path",
        help="Path to prepared features (optional)"
    )
    parser.add_argument(
        "--min-liquidations",
        type=int,
        default=100,
        help="Minimum liquidation events required"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Load or prepare data
    if args.data_path:
        logger.info(f"Loading data from {args.data_path}")
        data = pd.read_parquet(args.data_path)
        # Assume last column is label
        features_df = data.iloc[:, :-1]
        labels_df = data.iloc[:, -1]
    else:
        logger.info("Preparing training data from collected market data...")
        features_df, labels_df = await prepare_training_data(args.min_liquidations)
        
        if features_df is None:
            logger.error("Failed to prepare training data")
            return
    
    # Train model
    results = train_model(features_df, labels_df, args.test_size)
    
    # Print results
    print_results(results)
    
    print("\n✅ Training completed successfully!")
    print("Next steps:")
    print("  1. Review results in models/ directory")
    print("  2. Test model predictions: python scripts/test_model.py")
    print("  3. Deploy to production: python scripts/deploy_model.py")


if __name__ == "__main__":
    asyncio.run(main())