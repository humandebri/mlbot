#!/usr/bin/env python3
"""
Quick fix for unified trading system
"""

# Read the broken file and fix it
with open('/home/ubuntu/mlbot/src/integration/main_unified.py', 'r') as f:
    content = f.read()

# Fix the broken tasks list
content = content.replace(
    'self.tasks = [\n                feature_hub_task,\n                asyncio.create_task(self._trading_loop()),',
    'self.tasks = [\n                feature_hub_task,\n                asyncio.create_task(self._trading_loop()),'
)

# Write the fixed content
with open('/home/ubuntu/mlbot/src/integration/main_unified_fixed.py', 'w') as f:
    f.write(content)

print("Fixed version created as main_unified_fixed.py")