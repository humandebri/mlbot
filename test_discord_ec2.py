#!/usr/bin/env python3
"""Test Discord notification on EC2"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from src.common.discord_notifier import discord_notifier

# Test notification
result = discord_notifier.send_notification(
    title="🧪 Discord通知テスト",
    description="EC2からの通知テスト",
    color="00ff00",
    fields={
        "Status": "Test",
        "Server": "EC2",
        "Result": "Success if you see this"
    }
)

print(f"Notification sent: {result}")
print(f"Webhook URL: {discord_notifier.webhook_url[:50]}..." if discord_notifier.webhook_url else "No webhook")