#!/usr/bin/env python3
"""Simple Discord test script."""

import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def test_discord():
    """Test Discord webhook directly."""
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    
    if not webhook_url:
        print("ERROR: DISCORD_WEBHOOK not set in environment")
        return
    
    print(f"Discord webhook found: {webhook_url[:50]}...")
    
    # Create message
    message = {
        "embeds": [{
            "title": "🤖 MLBot Test Message",
            "description": "If you see this message, Discord notifications are working!",
            "color": 0x00FF00,
            "fields": [
                {"name": "Status", "value": "✅ Working", "inline": True},
                {"name": "Time", "value": "Now", "inline": True}
            ]
        }]
    }
    
    # Send message
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status == 204:
                    print("✅ Discord message sent successfully!")
                else:
                    text = await response.text()
                    print(f"❌ Discord error: {response.status} - {text}")
    except Exception as e:
        print(f"❌ Error sending Discord message: {e}")

if __name__ == "__main__":
    asyncio.run(test_discord())