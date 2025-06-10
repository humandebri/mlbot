"""
Model Server FastAPI Application.

High-performance REST API server for liquidation-driven trading ML models.

Features:
- Real-time inference with sub-millisecond latency
- Batch processing with optimal throughput
- Comprehensive health monitoring and metrics
- Model management and hot-reloading
- Production-ready error handling and logging
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import time
from datetime import datetime

from .api.prediction import router as prediction_router, set_prediction_service
from .api.health import router as health_router
from .api.management import router as management_router
from .services.prediction_service import PredictionService
from .schemas import ErrorResponse
from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)

# Global service instance
prediction_service: PredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    
    logger.info("Starting Model Server")
    
    # Initialize prediction service
    global prediction_service
    prediction_service = PredictionService()
    
    # Initialize with default model
    try:
        initialization_success = await prediction_service.initialize(
            model_version="v1.0",
            enable_auto_reload=True
        )
        
        if initialization_success:
            logger.info("Prediction service initialized successfully")
        else:
            logger.warning("Prediction service initialization failed, starting in degraded mode")
    
    except Exception as e:
        logger.error("Error initializing prediction service", exception=e)
        logger.warning("Starting server without model loaded")
    
    # Set global service instance
    set_prediction_service(prediction_service)
    
    # Setup graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(shutdown())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    yield
    
    # Cleanup
    await shutdown()


async def shutdown():
    """Graceful shutdown."""
    logger.info("Shutting down Model Server")
    
    if prediction_service:
        await prediction_service.shutdown()
    
    logger.info("Model Server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Liquidation Trading Model Server",
    description="""
    High-performance REST API server for liquidation-driven trading ML models.
    
    ## Features
    
    * **Real-time Inference**: Sub-millisecond prediction latency
    * **Batch Processing**: Optimal throughput for bulk predictions
    * **Model Management**: Hot-reloading and version control
    * **Health Monitoring**: Comprehensive metrics and health checks
    * **Production Ready**: Error handling, logging, and monitoring
    
    ## Prediction Endpoints
    
    * `/predict/single` - Single real-time prediction
    * `/predict/batch` - Batch predictions
    * `/predict/async-batch` - Asynchronous batch processing
    
    ## Management Endpoints
    
    * `/model/load` - Load or switch model versions
    * `/model/reload` - Reload current model
    * `/health/` - Comprehensive health check
    * `/health/metrics` - Performance metrics
    """,
    version="1.0.0",
    contact={
        "name": "Model Server Support",
        "email": "support@trading-models.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom middleware for request logging and timing
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and measure response times."""
    
    start_time = time.perf_counter()
    request_id = f"{int(time.time() * 1000)}_{hash(request.url.path) % 10000}"
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    logger.info("Request started",
               request_id=request_id,
               method=request.method,
               url=str(request.url),
               client_ip=request.client.host if request.client else "unknown")
    
    try:
        response = await call_next(request)
        
        response_time = (time.perf_counter() - start_time) * 1000  # ms
        
        logger.info("Request completed",
                   request_id=request_id,
                   method=request.method,
                   url=str(request.url),
                   status_code=response.status_code,
                   response_time_ms=response_time)
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
        
        return response
        
    except Exception as e:
        response_time = (time.perf_counter() - start_time) * 1000
        
        logger.error("Request failed",
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    response_time_ms=response_time,
                    exception=e)
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e),
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            },
            headers={
                "X-Request-ID": request_id,
                "X-Response-Time": f"{response_time:.2f}ms"
            }
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning("HTTP exception",
                  request_id=request_id,
                  status_code=exc.status_code,
                  detail=exc.detail,
                  url=str(request.url))
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "message": exc.detail,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        },
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors (typically validation errors)."""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning("Value error",
                  request_id=request_id,
                  error=str(exc),
                  url=str(request.url))
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        },
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error("Unhandled exception",
                request_id=request_id,
                exception=exc,
                url=str(request.url))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        },
        headers={"X-Request-ID": request_id}
    )


# Include routers
app.include_router(prediction_router)
app.include_router(health_router)
app.include_router(management_router)


# Root endpoint
@app.get("/", summary="Root", description="Model Server information")
async def root() -> Dict[str, Any]:
    """Get basic server information."""
    
    server_info = {
        "service": "Liquidation Trading Model Server",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health/",
            "metrics": "/health/metrics",
            "prediction": "/predict/single",
            "batch": "/predict/batch"
        }
    }
    
    # Add service status if available
    if prediction_service:
        server_info.update({
            "model_loaded": prediction_service.is_ready,
            "model_version": prediction_service.model_version,
            "total_predictions": prediction_service.total_predictions
        })
    
    return server_info


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation."""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Liquidation Trading Model Server",
        version="1.0.0",
        description="""
        Advanced ML inference server for liquidation-driven cryptocurrency trading.
        
        This high-performance API provides real-time predictions for trading opportunities
        based on liquidation cascade analysis and market microstructure features.
        """,
        routes=app.routes,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.trading-models.com",
            "description": "Production server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Development server runner
def run_dev_server():
    """Run development server."""
    
    uvicorn.run(
        "src.model_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"],
        log_level="info",
        access_log=True
    )


# Production server runner
def run_prod_server():
    """Run production server."""
    
    uvicorn.run(
        "src.model_server.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for ML models (avoid memory issues)
        log_level="info",
        access_log=True,
        loop="uvloop",  # High-performance event loop
        http="httptools"  # High-performance HTTP parser
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "prod":
        run_prod_server()
    else:
        run_dev_server()