#!/usr/bin/env python3
"""
Fix initialization error in bot
"""

fix_code = '''
# Find and fix the initialization error
import re

with open('simple_improved_bot_with_trading_fixed.py', 'r') as f:
    content = f.read()

# Find the error handling in __init__ or initialize method
# The error is: "can only concatenate str (not "IOException") to str"
# This means somewhere we're trying to concatenate a string with an exception object

# Fix pattern 1: Logger error calls
content = re.sub(
    r'logger\.error\(f"([^"]+): " \+ e\)',
    r'logger.error(f"\\1: {e}")',
    content
)

# Fix pattern 2: String concatenation with exception
content = re.sub(
    r'logger\.error\("([^"]+): " \+ e\)',
    r'logger.error(f"\\1: {e}")',
    content
)

# Fix pattern 3: Print statements
content = re.sub(
    r'print\(f"([^"]+): " \+ e\)',
    r'print(f"\\1: {e}")',
    content
)

# Fix pattern 4: Any remaining string + exception
content = re.sub(
    r'"([^"]+): " \+ (\w+)(?=\s*\))',
    r'f"\\1: {\\2}"',
    content
)

# Also ensure database path exists
if 'create_trading_tables()' in content and 'os.makedirs' not in content:
    # Add directory creation before create_trading_tables
    content = re.sub(
        r'(\s+)(create_trading_tables\(\))',
        r'\\1# Ensure data directory exists\\n\\1os.makedirs("data", exist_ok=True)\\n\\1\\2',
        content
    )

with open('simple_improved_bot_with_trading_fixed.py', 'w') as f:
    f.write(content)

print("✅ Fixed initialization error handling")
'''

print(fix_code)