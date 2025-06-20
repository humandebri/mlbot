#!/usr/bin/env python3
"""Analyze Python file dependencies in the project."""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, deque

def extract_imports(file_path):
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    for alias in node.names:
                        imports.add(f"{node.module}.{alias.name}")
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return imports

def is_project_module(module_name, project_root):
    """Check if a module is part of the project."""
    parts = module_name.split('.')
    if parts[0] in ['src', 'scripts', 'tests', 'deployment', 'monitoring']:
        return True
    
    # Check if it's a direct file in project root
    test_path = project_root / f"{parts[0]}.py"
    return test_path.exists()

def build_dependency_graph(start_files, project_root):
    """Build a dependency graph starting from given files."""
    visited = set()
    dependencies = defaultdict(set)
    queue = deque(start_files)
    
    while queue:
        file_path = queue.popleft()
        if file_path in visited:
            continue
        
        visited.add(file_path)
        imports = extract_imports(file_path)
        
        for imp in imports:
            if is_project_module(imp, project_root):
                # Convert import to file path
                module_path = imp.replace('.', '/')
                possible_paths = [
                    project_root / f"{module_path}.py",
                    project_root / module_path / "__init__.py"
                ]
                
                for path in possible_paths:
                    if path.exists():
                        dependencies[file_path].add(path)
                        if path not in visited:
                            queue.append(path)
                        break
    
    return visited, dependencies

def find_all_python_files(root_dir):
    """Find all Python files in the project."""
    python_files = set()
    exclude_dirs = {'.venv', '.venv_tf', '.venv_test', '__pycache__', '.git', 'venv', 'env'}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip if any excluded directory is in the path
        if any(exc in root for exc in exclude_dirs):
            continue
            
        # Remove excluded directories from search
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.add(Path(root) / file)
    
    return python_files

def main():
    project_root = Path('/Users/0xhude/Desktop/mlbot')
    
    # Main entry points
    entry_points = [
        project_root / 'main_dynamic_integration.py',
        project_root / 'simple_improved_bot_fixed.py',
        project_root / 'simple_improved_bot_with_trading_fixed.py'
    ]
    
    # Filter to only existing entry points
    existing_entry_points = [ep for ep in entry_points if ep.exists()]
    
    print("=== MLBOT DEPENDENCY ANALYSIS ===\n")
    print(f"Entry points found: {len(existing_entry_points)}")
    for ep in existing_entry_points:
        print(f"  - {ep.name}")
    
    # Build dependency graph
    print("\nAnalyzing dependencies...")
    used_files, dependencies = build_dependency_graph(existing_entry_points, project_root)
    
    # Find all Python files
    all_files = find_all_python_files(project_root)
    
    # Categorize files
    unused_files = all_files - used_files
    
    # Filter out test files from unused unless they're imported
    test_files = {f for f in unused_files if 'test' in f.name.lower() or 'test' in str(f).lower()}
    script_files = {f for f in unused_files if f.parent.name == 'scripts'}
    cleanup_files = {f for f in unused_files if 'cleanup' in str(f)}
    old_files = {f for f in unused_files if 'old_files' in str(f)}
    
    # Core unused files (not tests, scripts, cleanup, or old)
    core_unused = unused_files - test_files - script_files - cleanup_files - old_files
    
    print(f"\nTotal Python files: {len(all_files)}")
    print(f"Used files: {len(used_files)}")
    print(f"Unused files: {len(unused_files)}")
    
    print("\n=== ACTIVE FILES (Used by entry points) ===")
    # Group by directory
    used_by_dir = defaultdict(list)
    for f in sorted(used_files):
        rel_path = f.relative_to(project_root)
        dir_name = str(rel_path.parent) if str(rel_path.parent) != '.' else 'root'
        used_by_dir[dir_name].append(rel_path.name)
    
    for dir_name, files in sorted(used_by_dir.items()):
        print(f"\n{dir_name}/:")
        for f in sorted(files):
            print(f"  {f}")
    
    print("\n=== UNUSED CORE FILES (Can potentially be deleted) ===")
    unused_by_dir = defaultdict(list)
    for f in sorted(core_unused):
        rel_path = f.relative_to(project_root)
        dir_name = str(rel_path.parent) if str(rel_path.parent) != '.' else 'root'
        unused_by_dir[dir_name].append(rel_path.name)
    
    for dir_name, files in sorted(unused_by_dir.items()):
        print(f"\n{dir_name}/:")
        for f in sorted(files):
            print(f"  {f}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Script files (unused): {len(script_files)}")
    print(f"Test files (unused): {len(test_files)}")
    print(f"Cleanup files: {len(cleanup_files)}")
    print(f"Old files: {len(old_files)}")
    print(f"Core unused files: {len(core_unused)}")
    
    # Look for duplicate functionality
    print("\n=== POTENTIAL DUPLICATES ===")
    # Group files by similar names
    name_groups = defaultdict(list)
    for f in all_files:
        base_name = f.stem.replace('_fixed', '').replace('_patched', '').replace('_v2', '')
        base_name = base_name.replace('_final', '').replace('_working', '')
        name_groups[base_name].append(f)
    
    for base_name, files in sorted(name_groups.items()):
        if len(files) > 1:
            print(f"\n{base_name}:")
            for f in files:
                status = "ACTIVE" if f in used_files else "unused"
                print(f"  {f.relative_to(project_root)} [{status}]")

if __name__ == "__main__":
    main()