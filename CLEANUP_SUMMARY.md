# Cleanup Summary - June 20, 2025

## Overview
A safe cleanup was performed on the mlbot project, focusing on preserving the main bot (`simple_improved_bot_with_trading_fixed.py`) and its dependencies while removing unused files.

## Files Moved to cleanup_2025_06_20/

### Directories
- `cleanup/` - Already identified as unused (55 files)
- `old_files/` - Old versions and test scripts (40+ files)
- `docker/` - Abandoned Docker microservices
- `kubernetes/` - Abandoned Kubernetes configs (if it existed)
- `scripts/` - Training and analysis scripts (kept only essential 4 files)
- `monitoring/` - Unused monitoring dashboard
- `tests/` - Test files not needed for production
- `src/storage/` - Unused storage module
- `src/ingestor/`, `src/feature_hub/`, `src/model_server/`, `src/integration/` - Microservices code

### Individual Files
- 17 unused ML pipeline modules (feature adapters, trainers, validators)
- 7 unused order router modules (dynamic configs, enhanced managers)
- 14 unused common modules (decorators, performance, types, etc.)
- Various root-level Python files (old mains, test files)
- ~120 deployment scripts (kept only 2 essential ones)

## Files Preserved

### Main Bot Files
- `simple_improved_bot_with_trading_fixed.py` - Main bot (EC2で稼働中)
- `improved_feature_generator.py` - Feature generation using historical data

### Essential Modules
- `src/common/__init__.py`
- `src/common/discord_notifier.py`
- `src/common/database.py`
- `src/common/bybit_client.py`
- `src/common/account_monitor.py`
- `src/ml_pipeline/__init__.py`
- `src/ml_pipeline/inference_engine.py`
- `src/order_router/__init__.py`
- `src/order_router/risk_manager.py`

### Essential Scripts
- `scripts/check_status.py`
- `scripts/check_redis.py`
- `scripts/start_system.py`
- `scripts/stop_system.py`

### Config & Documentation
- `.env`
- `requirements.txt`
- `README.md`
- `CLAUDE.md`
- `SYSTEM_ARCHITECTURE.md`
- `REPAIR_PLAN.md`
- `pyproject.toml`
- `deployment/manual_deploy.sh`
- `deployment/production_deployment_guide.md`

### Data & Models
- `models/` directory (all models preserved)
- `data/` directory (historical data preserved)

## Current Project Structure
```
mlbot/
├── simple_improved_bot_with_trading_fixed.py  # Main bot
├── improved_feature_generator.py               # Feature generator
├── cleanup_2025_06_20/                        # All moved files
├── src/
│   ├── common/                                # Essential modules only
│   ├── ml_pipeline/                           # Inference engine only
│   └── order_router/                          # Risk manager only
├── scripts/                                   # 4 essential scripts
├── deployment/                                # 2 essential files
├── models/                                    # All models preserved
├── data/                                      # Historical data
└── logs/                                      # Log files
```

## Results
- **Total items moved**: 200+ files and directories
- **Space saved**: Significant reduction in project complexity
- **Project structure**: Now focused on the simple improved bot only
- **Safety**: All dependencies of the main bot preserved

## Note
The cleanup was extremely conservative. If any file was potentially used by the main bot, it was preserved. The moved files are in `cleanup_2025_06_20/` and can be restored if needed.

## Recommended Next Steps
1. Review files in `cleanup_2025_06_20/` and permanently delete if not needed
2. Clean up Python cache: `find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null`
3. Consider removing unused virtual environments (`.venv_test`, `.venv_tf`) if not needed
4. Apply similar cleanup to EC2 instance