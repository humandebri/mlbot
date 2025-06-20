#!/usr/bin/env python3
"""
Patch main_complete_working.py to add signal cooldown and confidence filtering.
"""

import re

# Read the file
with open('cleanup/temp_files/main_complete_working.py', 'r') as f:
    content = f.read()

# Add imports for cooldown tracking
imports_addition = """from datetime import datetime, timedelta
from collections import defaultdict
"""

# Find the imports section and add our imports
import_pattern = r'(import asyncio\nimport signal\nimport sys\nfrom pathlib import Path)'
content = re.sub(import_pattern, imports_addition + r'\1', content)

# Add cooldown tracking to __init__
init_addition = """        
        # Signal tracking for cooldown
        self.last_signal_time = defaultdict(lambda: datetime.min)
        self.signal_cooldown = timedelta(minutes=15)  # 15 minute cooldown per symbol
        self.confidence_threshold = 0.75  # Increased from 0.6
        self.min_prediction_change = 0.02  # 2% minimum prediction change
        self.last_predictions = {}
        self.discord_sent_count = 0
        self.max_discord_per_hour = 10  # Rate limit Discord messages"""

# Find the __init__ method and add our tracking variables
init_pattern = r'(self\.inference_engine = InferenceEngine\(inference_config\))'
content = re.sub(init_pattern, r'\1' + init_addition, content)

# Replace the trading loop condition
old_condition = r'if confidence > 0\.6:'
new_condition = '''if confidence > self.confidence_threshold and self._should_generate_signal(symbol, prediction, confidence):'''

content = re.sub(old_condition, new_condition, content)

# Add the _should_generate_signal method before _health_monitor
method_to_add = '''
    def _should_generate_signal(self, symbol: str, prediction: float, confidence: float) -> bool:
        """Determine if a signal should be generated based on filtering criteria."""
        now = datetime.now()
        
        # Check Discord rate limit
        if self.discord_sent_count >= self.max_discord_per_hour:
            return False
        
        # Check confidence threshold (already checked in caller, but double-check)
        if confidence < self.confidence_threshold:
            return False
        
        # Check cooldown
        time_since_last = now - self.last_signal_time[symbol]
        if time_since_last < self.signal_cooldown:
            return False
        
        # Check prediction significance
        if abs(prediction) < self.min_prediction_change:
            return False
        
        # Check if prediction has changed significantly from last signal
        if symbol in self.last_predictions:
            pred_change = abs(prediction - self.last_predictions[symbol])
            if pred_change < self.min_prediction_change:
                return False
        
        # All checks passed
        return True
'''

# Find where to insert the method (before _health_monitor)
health_monitor_pattern = r'(\n    async def _health_monitor\(self\):)'
content = re.sub(health_monitor_pattern, method_to_add + r'\1', content)

# Update signal tracking after Discord notification
tracking_update = '''
                                
                                # Update tracking
                                self.last_signal_time[symbol] = datetime.now()
                                self.last_predictions[symbol] = prediction
                                self.discord_sent_count += 1'''

# Find where to add tracking update (after Discord notification)
notification_pattern = r'(logger\.info\(f"\ud83d\udcf2 Discord notification sent for {symbol}"\))'
content = re.sub(notification_pattern, r'\1' + tracking_update, content)

# Add hourly reset for Discord rate limit
rate_limit_reset = '''
                # Reset Discord rate limit every hour
                if loop_count % 3600 == 0:
                    self.discord_sent_count = 0
                    logger.info("Reset Discord rate limit counter")
                '''

# Find where to add rate limit reset (in trading loop)
stats_pattern = r'(if loop_count % 300 == 0:.*?High confidence signals={high_confidence_signals}".*?\))'
content = re.sub(stats_pattern, r'\1' + rate_limit_reset, content, flags=re.DOTALL)

# Save the patched file
with open('main_complete_working_patched.py', 'w') as f:
    f.write(content)

print("✅ Successfully created patched version: main_complete_working_patched.py")
print("Features added:")
print("- 15 minute cooldown per symbol")
print("- 75% confidence threshold (increased from 60%)")
print("- 2% minimum prediction change requirement")
print("- Discord rate limiting (max 10 messages per hour)")
print("- Signal tracking to prevent spam")