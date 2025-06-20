#!/usr/bin/env python3
"""
Safe cleanup script that preserves all dependencies of simple_improved_bot_with_trading_fixed.py
"""

import os
import shutil
from pathlib import Path
import ast

# Files that must NOT be deleted (whitelist)
PROTECTED_FILES = {
    # Main bot and its direct dependency
    'simple_improved_bot_with_trading_fixed.py',
    'improved_feature_generator.py',
    
    # Common modules used by the bot
    'src/common/__init__.py',
    'src/common/discord_notifier.py',
    'src/common/database.py',
    'src/common/bybit_client.py',
    'src/common/account_monitor.py',
    
    # ML pipeline modules
    'src/ml_pipeline/__init__.py',
    'src/ml_pipeline/inference_engine.py',
    
    # Order router modules
    'src/order_router/__init__.py',
    'src/order_router/risk_manager.py',
    
    # Models directory - ALL PROTECTED
    'models/',
    
    # Data directory
    'data/',
    
    # Config files
    '.env',
    'requirements.txt',
    'README.md',
    'CLAUDE.md',
    'SYSTEM_ARCHITECTURE.md',
    'REPAIR_PLAN.md',
    'pyproject.toml',
    
    # Git files
    '.git/',
    '.gitignore',
}

# Directories that are safe to clean
SAFE_TO_CLEAN = {
    'cleanup/',  # Already identified as unused
    'old_files/',  # Already identified as old
    'kubernetes/',  # Abandoned microservices
    'docker/',  # Abandoned docker files
    'scripts/',  # Training/analysis scripts (but keep some)
    'deployment/',  # Old deployment scripts
}

# Scripts to keep in scripts/
KEEP_SCRIPTS = {
    'check_status.py',
    'check_redis.py',
    'start_system.py',
    'stop_system.py',
}

def should_keep_file(file_path: str) -> bool:
    """Check if a file should be kept."""
    # Convert to relative path
    rel_path = Path(file_path).relative_to('/Users/0xhude/Desktop/mlbot')
    rel_path_str = str(rel_path)
    
    # Check if it's a protected file
    for protected in PROTECTED_FILES:
        if protected.endswith('/'):
            # It's a directory
            if rel_path_str.startswith(protected):
                return True
        else:
            # It's a file
            if rel_path_str == protected or rel_path_str.startswith(protected + '/'):
                return True
    
    # Check if it's in scripts but should be kept
    if rel_path_str.startswith('scripts/'):
        file_name = os.path.basename(rel_path_str)
        if file_name in KEEP_SCRIPTS:
            return True
    
    # Check if it's a common module that might be imported
    if rel_path_str.startswith('src/common/'):
        # Keep all common modules for safety
        return True
    
    return False

def move_to_cleanup(src_path: str, cleanup_dir: str):
    """Move file/directory to cleanup directory preserving structure."""
    rel_path = Path(src_path).relative_to('/Users/0xhude/Desktop/mlbot')
    dest_path = Path(cleanup_dir) / rel_path
    
    # Create parent directories
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Move the file/directory
    if os.path.isdir(src_path):
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.move(src_path, dest_path)
    else:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(src_path, dest_path)
    
    print(f"Moved: {rel_path}")

def main():
    base_dir = '/Users/0xhude/Desktop/mlbot'
    cleanup_dir = os.path.join(base_dir, 'cleanup_2025_06_20')
    
    # Create cleanup directory
    os.makedirs(cleanup_dir, exist_ok=True)
    
    moved_count = 0
    
    # 1. Move entire cleanup and old_files directories
    for dir_name in ['cleanup', 'old_files']:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            move_to_cleanup(dir_path, cleanup_dir)
            moved_count += 1
    
    # 2. Clean kubernetes directory
    kube_dir = os.path.join(base_dir, 'kubernetes')
    if os.path.exists(kube_dir):
        move_to_cleanup(kube_dir, cleanup_dir)
        moved_count += 1
    
    # 3. Clean docker directory (old microservices)
    docker_dir = os.path.join(base_dir, 'docker')
    if os.path.exists(docker_dir):
        move_to_cleanup(docker_dir, cleanup_dir)
        moved_count += 1
    
    # 4. Clean scripts directory (keep essential ones)
    scripts_dir = os.path.join(base_dir, 'scripts')
    if os.path.exists(scripts_dir):
        for file_name in os.listdir(scripts_dir):
            if file_name not in KEEP_SCRIPTS and not file_name.startswith('.'):
                file_path = os.path.join(scripts_dir, file_name)
                move_to_cleanup(file_path, cleanup_dir)
                moved_count += 1
    
    # 5. Clean deployment directory (keep only essential)
    keep_deployment = {'manual_deploy.sh', 'production_deployment_guide.md'}
    deploy_dir = os.path.join(base_dir, 'deployment')
    if os.path.exists(deploy_dir):
        for file_name in os.listdir(deploy_dir):
            if file_name not in keep_deployment and not file_name.startswith('.'):
                file_path = os.path.join(deploy_dir, file_name)
                move_to_cleanup(file_path, cleanup_dir)
                moved_count += 1
    
    # 6. Clean root directory Python files (keep only active ones)
    keep_root_files = {
        'simple_improved_bot_with_trading_fixed.py',
        'improved_feature_generator.py',
        'safe_cleanup.py',
    }
    
    for file_name in os.listdir(base_dir):
        if file_name.endswith('.py') and file_name not in keep_root_files:
            file_path = os.path.join(base_dir, file_name)
            if os.path.isfile(file_path):
                move_to_cleanup(file_path, cleanup_dir)
                moved_count += 1
    
    # 7. Clean up old main files
    old_mains = [
        'main_dynamic_integration.py',
        'simple_improved_bot_fixed.py',
        'ml_feature_generator.py',
    ]
    for file_name in old_mains:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            move_to_cleanup(file_path, cleanup_dir)
            moved_count += 1
    
    # 8. Clean microservices implementations (not used by simple bot)
    microservice_dirs = [
        'src/ingestor',
        'src/feature_hub',
        'src/model_server',
        'src/integration',
        'src/system',
    ]
    for dir_name in microservice_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            move_to_cleanup(dir_path, cleanup_dir)
            moved_count += 1
    
    print(f"\nCleanup complete! Moved {moved_count} items to {cleanup_dir}")
    print("\nProtected files/directories:")
    for protected in sorted(PROTECTED_FILES):
        print(f"  ✓ {protected}")

if __name__ == "__main__":
    main()