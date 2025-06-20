# MLBot File Cleanup Report

## Executive Summary

The MLBot project has 294 Python files (excluding virtual environments), but only **22 files are actively used** by the main entry points. This represents significant bloat with 272 unused files that can be cleaned up.

## Main Entry Points (Currently Active)

According to CLAUDE.md and the dependency analysis:
1. **`main_dynamic_integration.py`** - Main integrated system (microservice architecture)
2. **`simple_improved_bot_fixed.py`** - Simplified bot running on EC2
3. **`simple_improved_bot_with_trading_fixed.py`** - Simplified bot with trading features

## Active Files (22 files actually in use)

### Root Directory (4 files)
- `improved_feature_generator.py` - Feature generation from historical data
- `main_dynamic_integration.py` - Main entry point
- `simple_improved_bot_fixed.py` - Simplified bot
- `simple_improved_bot_with_trading_fixed.py` - Simplified bot with trading

### src/common/ (11 files)
- `account_monitor.py` - Account balance monitoring
- `bybit_client.py` - Bybit API client
- `config.py` - Configuration management
- `database.py` - Database connections
- `decorators.py` - Common decorators
- `discord_notifier.py` - Discord notifications
- `error_handler.py` - Error handling
- `exceptions.py` - Custom exceptions
- `logging.py` - Logging configuration
- `performance.py` - Performance monitoring
- `types.py` - Type definitions

### src/feature_hub/ (1 file)
- `main.py` - Feature hub service

### src/ingestor/ (1 file)
- `main.py` - Data ingestion service

### src/integration/ (2 files)
- `dynamic_trading_coordinator.py` - Trading coordination
- `simple_service_manager.py` - Service management

### src/ml_pipeline/ (1 file)
- `inference_engine.py` - ML inference

### src/order_router/ (2 files)
- `main.py` - Order routing service
- `risk_manager.py` - Risk management

## Unused Files by Category

### 1. Cleanup Directory (53 files)
Already moved to cleanup/ - these are temporary files and debug scripts

### 2. Old Files Directory (44 files)
Already moved to old_files/ - deprecated implementations

### 3. Scripts Directory (95 files)
Training scripts, backtests, and analysis tools - not used in production

### 4. Tests (65 files)
Test files not imported by main code

### 5. Core Unused Files (71 files)
These are in the main src/ structure but not used:

#### Unused Feature Engineering
- `src/feature_hub/advanced_features.py`
- `src/feature_hub/liquidation_features.py`
- `src/feature_hub/micro_liquidity.py`
- `src/feature_hub/price_features.py` (duplicate of price_features_fixed.py)
- `src/feature_hub/technical_indicators.py`
- `src/feature_hub/time_context.py`
- `src/feature_hub/volatility_momentum.py`

#### Unused ML Pipeline
- `src/ml_pipeline/data_preprocessing.py`
- `src/ml_pipeline/feature_adapter.py`
- `src/ml_pipeline/feature_adapter_26.py`
- `src/ml_pipeline/feature_adapter_44.py`
- `src/ml_pipeline/pytorch_inference_engine.py`
- `src/ml_pipeline/v31_improved_inference_engine.py`
- All training/optimization modules

#### Unused Model Server
- Entire `src/model_server/` directory (microservice architecture abandoned)

#### Unused Integration Files
- `src/integration/api_gateway.py`
- `src/integration/main.py`
- `src/integration/main_unified.py`
- `src/integration/service_manager.py`
- `src/integration/simple_service_manager_fixed.py`
- `src/integration/trading_coordinator.py`

## Duplicate Files

### Main System Files
- `cleanup/temp_files/main_working*.py` (multiple versions)
- `old_files/unused_bots/main_complete_working_patched.py`

### Service Managers
- `src/integration/simple_service_manager.py` (ACTIVE)
- `src/integration/simple_service_manager_fixed.py` (unused)
- `old_files/fix_scripts/simple_service_manager_fixed.py` (unused)

### Price Features
- `src/feature_hub/price_features.py` (unused)
- `src/feature_hub/price_features_fixed.py` (unused)
- `old_files/fix_scripts/price_features_fixed.py` (unused)

## Recommendations

### 1. Immediate Cleanup (Safe to Delete)
- All files in `cleanup/` directory (53 files)
- All files in `old_files/` directory (44 files)
- All unused test scripts in `scripts/` that are for training/backtesting (95 files)
- All `__init__.py` files in unused directories

### 2. Architecture Cleanup
Since Docker/microservices architecture was abandoned:
- Remove `src/model_server/` directory completely
- Remove unused integration files
- Remove Docker-related files (`docker/`, `docker-compose.yml` if exists)
- Remove Kubernetes files (`k8s/`)

### 3. Feature Engineering Cleanup
The active bots use `improved_feature_generator.py` instead of the modular feature system:
- Remove all unused feature modules in `src/feature_hub/`
- Keep only `main.py` if still needed

### 4. ML Pipeline Cleanup
Since simplified bots use direct ONNX inference:
- Remove unused feature adapters
- Remove training/optimization modules
- Keep only `inference_engine.py`

### 5. Documentation Updates
- Update CLAUDE.md to remove references to deleted files
- Update deployment scripts to match current architecture
- Create a simplified README focusing on the two main entry points

## Summary Statistics

- **Total Python files**: 294
- **Active files**: 22 (7.5%)
- **Unused files**: 272 (92.5%)
- **Potential deletion**: ~250 files (after keeping some scripts/tests)

## Next Steps

1. Move additional unused files to appropriate cleanup directories
2. Delete cleanup directories after confirming with user
3. Update documentation to reflect simplified architecture
4. Update deployment scripts
5. Clean up requirements.txt to remove unused dependencies