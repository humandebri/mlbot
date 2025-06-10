"""
Main entry point for the integrated trading system.

Starts the API Gateway which manages all system components.
"""

import asyncio
import signal
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger

logger = get_logger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Run the integrated trading system."""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Liquidation Trading Bot System")
    
    # Run API Gateway
    uvicorn.run(
        "src.integration.api_gateway:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()