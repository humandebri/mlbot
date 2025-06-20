"""
API Gateway for unified system access.

Provides a single endpoint for:
- System status and monitoring
- Trading operations
- Configuration management
- Performance metrics
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from .trading_coordinator import TradingCoordinator, SystemConfig
from .service_manager import ServiceManager
from ..common.logging import get_logger
from ..common.monitoring import setup_metrics_server

logger = get_logger(__name__)

# Global instances
trading_coordinator: Optional[TradingCoordinator] = None
service_manager: Optional[ServiceManager] = None

# Create FastAPI app
app = FastAPI(
    title="Liquidation Trading Bot API",
    description="""
    Unified API for the liquidation-driven cryptocurrency trading system.
    
    ## Features
    
    * **System Management**: Start/stop services, health monitoring
    * **Trading Operations**: View positions, performance, risk metrics  
    * **Configuration**: Update trading parameters
    * **Monitoring**: Real-time metrics and system status
    """,
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to ensure system is initialized
async def get_trading_coordinator() -> TradingCoordinator:
    """Get trading coordinator instance."""
    global trading_coordinator
    if trading_coordinator is None:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    return trading_coordinator


async def get_service_manager() -> ServiceManager:
    """Get service manager instance."""
    global service_manager
    if service_manager is None:
        raise HTTPException(status_code=503, detail="Service manager not initialized")
    return service_manager


# System Management Endpoints

@app.post("/system/start", summary="Start Trading System")
async def start_system(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Start all system components."""
    global trading_coordinator, service_manager
    
    try:
        # Initialize service manager
        if service_manager is None:
            service_manager = ServiceManager()
        
        # Start all services
        await service_manager.start_all()
        
        # Wait for services to be ready
        await asyncio.sleep(5)
        
        # Initialize trading coordinator
        if trading_coordinator is None:
            trading_coordinator = TradingCoordinator(SystemConfig())
        
        # Start trading coordinator
        await trading_coordinator.start()
        
        return {
            "status": "success",
            "message": "Trading system started successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to start system", exception=e)
        raise HTTPException(status_code=500, detail=f"System startup failed: {str(e)}")


@app.post("/system/stop", summary="Stop Trading System")
async def stop_system() -> Dict[str, str]:
    """Stop all system components."""
    global trading_coordinator, service_manager
    
    try:
        # Stop trading coordinator
        if trading_coordinator:
            await trading_coordinator.stop()
        
        # Stop all services
        if service_manager:
            await service_manager.stop_all()
        
        return {
            "status": "success",
            "message": "Trading system stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to stop system", exception=e)
        raise HTTPException(status_code=500, detail=f"System shutdown failed: {str(e)}")


@app.get("/system/status", summary="Get System Status")
async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "trading_system": "not_initialized",
        "services": {}
    }
    
    # Get service status
    if service_manager:
        service_status = service_manager.get_status()
        status["services"] = service_status["services"]
    
    # Get trading system status
    if trading_coordinator:
        try:
            trading_status = await trading_coordinator.get_system_status()
            status["trading_system"] = trading_status
        except Exception as e:
            status["trading_system"] = {"error": str(e)}
    
    return status


@app.get("/system/health", summary="Health Check")
async def health_check() -> Dict[str, Any]:
    """Check health of all components."""
    
    health = {
        "status": "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check services
    if service_manager:
        service_health = await service_manager.health_check()
        health["components"].update(service_health)
    
    # Check trading system
    if trading_coordinator:
        trading_health = await trading_coordinator._check_services_health()
        health["components"]["trading_coordinator"] = all(trading_health.values())
    
    # Overall health
    if health["components"]:
        health["status"] = "healthy" if all(health["components"].values()) else "degraded"
    
    return health


# Trading Operations Endpoints

@app.get("/trading/positions", summary="Get Active Positions")
async def get_positions(
    coordinator: TradingCoordinator = Depends(get_trading_coordinator)
) -> Dict[str, Any]:
    """Get all active trading positions."""
    
    try:
        router_status = await coordinator.order_router.get_status()
        
        return {
            "positions": router_status["positions"],
            "total_positions": router_status["active_positions"],
            "total_exposure": sum(p["quantity"] * p["current_price"] for p in router_status["positions"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get positions", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trading/performance", summary="Get Performance Metrics")
async def get_performance(
    coordinator: TradingCoordinator = Depends(get_trading_coordinator)
) -> Dict[str, Any]:
    """Get trading performance metrics."""
    
    try:
        router_status = await coordinator.order_router.get_status()
        performance = router_status["performance"]
        
        return {
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get performance", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trading/risk", summary="Get Risk Metrics")
async def get_risk_metrics(
    coordinator: TradingCoordinator = Depends(get_trading_coordinator)
) -> Dict[str, Any]:
    """Get current risk metrics."""
    
    try:
        router_status = await coordinator.order_router.get_status()
        risk = router_status["risk"]
        
        return {
            "risk_metrics": risk,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get risk metrics", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trading/signals/recent", summary="Get Recent Trading Signals")
async def get_recent_signals(
    limit: int = 20,
    coordinator: TradingCoordinator = Depends(get_trading_coordinator)
) -> Dict[str, Any]:
    """Get recent trading signals."""
    
    try:
        status = await coordinator.get_system_status()
        signals = status.get("recent_signals", [])
        
        return {
            "signals": signals[:limit],
            "total_signals": coordinator.signal_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get signals", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


# Configuration Endpoints

@app.get("/config/trading", summary="Get Trading Configuration")
async def get_trading_config(
    coordinator: TradingCoordinator = Depends(get_trading_coordinator)
) -> Dict[str, Any]:
    """Get current trading configuration."""
    
    return {
        "config": {
            "symbols": coordinator.config.symbols,
            "min_prediction_confidence": coordinator.config.min_prediction_confidence,
            "min_expected_pnl": coordinator.config.min_expected_pnl,
            "prediction_interval_seconds": coordinator.config.prediction_interval_seconds,
            "feature_window_seconds": coordinator.config.feature_window_seconds
        },
        "timestamp": datetime.now().isoformat()
    }


@app.put("/config/trading/symbols", summary="Update Trading Symbols")
async def update_trading_symbols(
    symbols: List[str],
    coordinator: TradingCoordinator = Depends(get_trading_coordinator)
) -> Dict[str, Any]:
    """Update list of traded symbols."""
    
    try:
        # Validate symbols
        valid_symbols = [s.upper() for s in symbols if s.endswith("USDT")]
        
        if not valid_symbols:
            raise ValueError("No valid USDT pairs provided")
        
        # Update configuration
        coordinator.config.symbols = valid_symbols
        coordinator.active_symbols = set(valid_symbols)
        
        logger.info("Updated trading symbols", symbols=valid_symbols)
        
        return {
            "status": "success",
            "symbols": valid_symbols,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to update symbols", exception=e)
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/config/trading/thresholds", summary="Update Trading Thresholds")
async def update_trading_thresholds(
    min_confidence: Optional[float] = None,
    min_expected_pnl: Optional[float] = None,
    coordinator: TradingCoordinator = Depends(get_trading_coordinator)
) -> Dict[str, Any]:
    """Update trading thresholds."""
    
    try:
        if min_confidence is not None:
            if not 0 < min_confidence <= 1:
                raise ValueError("min_confidence must be between 0 and 1")
            coordinator.config.min_prediction_confidence = min_confidence
        
        if min_expected_pnl is not None:
            if not 0 < min_expected_pnl < 0.1:  # Max 10%
                raise ValueError("min_expected_pnl must be between 0 and 0.1")
            coordinator.config.min_expected_pnl = min_expected_pnl
        
        return {
            "status": "success",
            "config": {
                "min_prediction_confidence": coordinator.config.min_prediction_confidence,
                "min_expected_pnl": coordinator.config.min_expected_pnl
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to update thresholds", exception=e)
        raise HTTPException(status_code=400, detail=str(e))


# Service Management Endpoints

@app.post("/services/{service_name}/restart", summary="Restart Service")
async def restart_service(
    service_name: str,
    manager: ServiceManager = Depends(get_service_manager)
) -> Dict[str, Any]:
    """Restart a specific service."""
    
    try:
        success = await manager.restart_service(service_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Service {service_name} restarted",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to restart {service_name}")
            
    except Exception as e:
        logger.error(f"Failed to restart service {service_name}", exception=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services/status", summary="Get Service Status")
async def get_services_status(
    manager: ServiceManager = Depends(get_service_manager)
) -> Dict[str, Any]:
    """Get status of all managed services."""
    
    return manager.get_status()


# Root endpoint
@app.get("/", summary="API Information")
async def root() -> Dict[str, Any]:
    """Get API information."""
    
    return {
        "service": "Liquidation Trading Bot API Gateway",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "system": "/system/*",
            "trading": "/trading/*",
            "config": "/config/*",
            "services": "/services/*",
            "docs": "/docs"
        }
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize metrics server on startup."""
    setup_metrics_server(port=9090)
    logger.info("API Gateway started")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global trading_coordinator, service_manager
    
    if trading_coordinator:
        await trading_coordinator.stop()
    
    if service_manager:
        await service_manager.stop_all()
    
    logger.info("API Gateway shutdown complete")