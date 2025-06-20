"""Main entry point for the trading bot application."""

import asyncio
import signal
import sys
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .common.config import settings
from .common.logging import get_logger, setup_logging
from .common.monitoring import start_monitoring
from .common.database import init_databases, close_databases

# Setup logging
setup_logging()
logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="Bybit Liquidation Trading Bot",
    description="ML-driven trading bot using Bybit liquidation feeds",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
health_checker: Optional[object] = None
metrics_collector: Optional[object] = None
services: List[asyncio.Task] = []


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    global health_checker, metrics_collector
    
    logger.info("Starting Bybit Trading Bot", version="0.1.0", environment=settings.environment)
    
    try:
        # Initialize databases
        await init_databases()
        logger.info("Databases initialized")
        
        # Start monitoring
        health_checker, metrics_collector = await start_monitoring()
        logger.info("Monitoring services started")
        
        # Start individual services based on environment
        if settings.environment == "development":
            # In development, start all services in the same process
            await start_all_services()
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error("Failed to start application", exception=e)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    global metrics_collector
    
    logger.info("Shutting down Bybit Trading Bot")
    
    try:
        # Stop services
        for service in services:
            service.cancel()
        
        if services:
            await asyncio.gather(*services, return_exceptions=True)
        
        # Stop monitoring
        if metrics_collector:
            await metrics_collector.stop()
        
        # Close databases
        await close_databases()
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error("Error during shutdown", exception=e)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if health_checker is None:
        return {"status": "starting"}
    
    status = health_checker.get_status()
    return {
        "status": "healthy" if status["healthy"] else "unhealthy",
        "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        "version": "0.1.0",
        "environment": settings.environment,
        "components": status["components"],
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def start_all_services():
    """Start all trading bot services."""
    global services
    
    try:
        # Import service modules
        from .ingestor.main import start_ingestor
        from .feature_hub.main import start_feature_hub
        from .model_server.main import start_model_server
        from .order_router.main import start_order_router
        
        # Start services
        services.extend([
            asyncio.create_task(start_ingestor(), name="ingestor"),
            asyncio.create_task(start_feature_hub(), name="feature_hub"),
            asyncio.create_task(start_model_server(), name="model_server"),
            asyncio.create_task(start_order_router(), name="order_router"),
        ])
        
        logger.info("All services started successfully")
        
    except ImportError as e:
        logger.warning(f"Some services not available yet: {e}")
    except Exception as e:
        logger.error("Failed to start services", exception=e)
        raise


def handle_signal(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


async def main():
    """Main application entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Run the application
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8080,
        log_level=settings.logging.level.lower(),
        access_log=settings.debug,
        reload=settings.debug,
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())