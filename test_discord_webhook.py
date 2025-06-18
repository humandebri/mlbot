#!/usr/bin/env python3
"""
Test Discord webhook connectivity to ensure notifications are working.
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

webhook_url = os.getenv("DISCORD_WEBHOOK", "").strip()
if not webhook_url:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

if not webhook_url:
    print("❌ DISCORD_WEBHOOK_URL not found in environment variables")
    exit(1)

# Test message
message = {
    "embeds": [{
        "title": "🔧 Discord Webhook Test",
        "description": "This is a test message to verify Discord notifications are working properly.",
        "color": 0x00ff00,  # Green
        "fields": [
            {
                "name": "Status",
                "value": "✅ Webhook connection successful",
                "inline": True
            },
            {
                "name": "Time",
                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "inline": True
            },
            {
                "name": "Info",
                "value": "If you see this message, Discord notifications should work for trading signals.",
                "inline": False
            }
        ],
        "footer": {
            "text": "ML Trading Bot Test"
        }
    }]
}

try:
    response = requests.post(webhook_url, json=message)
    if response.status_code == 204:
        print("✅ Discord test message sent successfully!")
        print("Check your Discord channel for the test message.")
    else:
        print(f"❌ Failed to send Discord message. Status code: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Error sending Discord message: {e}")