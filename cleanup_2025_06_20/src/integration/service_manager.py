"""
Service manager for starting and monitoring all system components.

Manages the lifecycle of:
- Data Ingestor
- Feature Hub
- Model Server
- Order Router
- Trading Coordinator
"""

import asyncio
import subprocess
import psutil
import signal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import os
import sys
import time

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a managed service."""
    
    name: str
    command: List[str]
    env: Optional[Dict[str, str]] = None
    working_dir: Optional[str] = None
    port: Optional[int] = None
    health_check_url: Optional[str] = None
    startup_timeout: int = 30
    shutdown_timeout: int = 10
    restart_on_failure: bool = True
    max_restarts: int = 3


class ManagedService:
    """Represents a managed service process."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.start_time: Optional[datetime] = None
        self.restart_count: int = 0
        self.status: str = "stopped"
        self.last_error: Optional[str] = None
    
    async def start(self) -> bool:
        """Start the service."""
        if self.is_running():
            logger.warning(f"{self.config.name} is already running")
            return True
        
        try:
            logger.info(f"Starting {self.config.name}")
            
            # Prepare environment
            env = os.environ.copy()
            if self.config.env:
                env.update(self.config.env)
            
            # Start process
            self.process = subprocess.Popen(
                self.config.command,
                env=env,
                cwd=self.config.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if sys.platform != "win32" else None
            )
            
            self.pid = self.process.pid
            self.start_time = datetime.now()
            self.status = "starting"
            
            # Wait for startup
            await self._wait_for_startup()
            
            self.status = "running"
            logger.info(f"{self.config.name} started successfully", pid=self.pid)
            return True
            
        except Exception as e:
            self.status = "failed"
            self.last_error = str(e)
            logger.error(f"Failed to start {self.config.name}", exception=e)
            return False
    
    async def stop(self) -> bool:
        """Stop the service."""
        if not self.is_running():
            logger.warning(f"{self.config.name} is not running")
            return True
        
        try:
            logger.info(f"Stopping {self.config.name}")
            self.status = "stopping"
            
            # Send SIGTERM
            if sys.platform == "win32":
                self.process.terminate()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait for shutdown
            try:
                self.process.wait(timeout=self.config.shutdown_timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"{self.config.name} did not stop gracefully, forcing")
                if sys.platform == "win32":
                    self.process.kill()
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
            
            self.status = "stopped"
            self.process = None
            self.pid = None
            
            logger.info(f"{self.config.name} stopped")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to stop {self.config.name}", exception=e)
            return False
    
    async def restart(self) -> bool:
        """Restart the service."""
        logger.info(f"Restarting {self.config.name}")
        
        if await self.stop():
            await asyncio.sleep(1)  # Brief pause
            return await self.start()
        
        return False
    
    def is_running(self) -> bool:
        """Check if service is running."""
        if self.process and self.process.poll() is None:
            return True
        return False
    
    async def _wait_for_startup(self) -> None:
        """Wait for service to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < self.config.startup_timeout:
            # Check if process is still running
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.config.name} exited during startup")
            
            # Check port if specified
            if self.config.port:
                if self._is_port_open(self.config.port):
                    await asyncio.sleep(1)  # Give it a moment to fully initialize
                    return
            else:
                # For services without ports, just check if process is alive after a brief startup period
                if time.time() - start_time > 3:  # Give it 3 seconds to start
                    # Check if process is still running
                    if self.process.poll() is None:
                        logger.info(f"{self.config.name} appears to be running (no port to check)")
                        return
            
            await asyncio.sleep(0.5)
        
        # Only timeout if service has a port to check
        if self.config.port:
            raise TimeoutError(f"{self.config.name} failed to start within {self.config.startup_timeout}s")
        else:
            # For services without ports, if process is still running, consider it started
            if self.process.poll() is None:
                logger.info(f"{self.config.name} running without port verification")
                return
            else:
                raise RuntimeError(f"{self.config.name} process exited")
    
    def _is_port_open(self, port: int) -> bool:
        """Check if a port is open."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        status = {
            "name": self.config.name,
            "status": self.status,
            "pid": self.pid,
            "start_time": self.start_time,
            "restart_count": self.restart_count,
            "last_error": self.last_error
        }
        
        # Add resource usage if running
        if self.pid and self.is_running():
            try:
                proc = psutil.Process(self.pid)
                status["cpu_percent"] = proc.cpu_percent()
                status["memory_mb"] = proc.memory_info().rss / 1024 / 1024
            except:
                pass
        
        return status


class ServiceManager:
    """Manages all system services."""
    
    def __init__(self):
        """Initialize service manager."""
        self.services: Dict[str, ManagedService] = {}
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Define services
        self._define_services()
        
        logger.info("Service manager initialized")
    
    def _define_services(self) -> None:
        """Define all managed services."""
        
        # Data Ingestor
        self.add_service(ServiceConfig(
            name="ingestor",
            command=[sys.executable, "-m", "src.ingestor.main"],
            port=None,
            env={"SERVICE_NAME": "ingestor"}
        ))
        
        # Feature Hub
        self.add_service(ServiceConfig(
            name="feature_hub",
            command=[sys.executable, "-m", "src.feature_hub.main"],
            port=None,  # Feature Hub doesn't have HTTP server
            health_check_url=None,
            env={"SERVICE_NAME": "feature_hub"}
        ))
        
        # Model Server
        self.add_service(ServiceConfig(
            name="model_server",
            command=[sys.executable, "-m", "src.model_server.main"],
            port=8000,
            health_check_url="http://localhost:8000/health/ready",
            env={"SERVICE_NAME": "model_server"}
        ))
        
        # Order Router (started by coordinator)
        # Trading Coordinator manages Order Router internally
    
    def add_service(self, config: ServiceConfig) -> None:
        """Add a service to be managed."""
        self.services[config.name] = ManagedService(config)
    
    async def start_all(self) -> None:
        """Start all services in order."""
        logger.info("Starting all services")
        self.running = True
        
        # Start services in dependency order
        startup_order = ["ingestor", "feature_hub", "model_server"]
        
        for service_name in startup_order:
            if service_name in self.services:
                service = self.services[service_name]
                
                success = await service.start()
                if not success:
                    logger.error(f"Failed to start {service_name}, aborting startup")
                    await self.stop_all()
                    raise RuntimeError(f"Service startup failed: {service_name}")
                
                # Brief pause between services
                await asyncio.sleep(2)
        
        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitor_services())
        
        logger.info("All services started successfully")
    
    async def stop_all(self) -> None:
        """Stop all services in reverse order."""
        logger.info("Stopping all services")
        self.running = False
        
        # Cancel monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop services in reverse order
        for service_name in reversed(list(self.services.keys())):
            service = self.services[service_name]
            if service.is_running():
                await service.stop()
        
        logger.info("All services stopped")
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        service = self.services[service_name]
        return await service.restart()
    
    async def _monitor_services(self) -> None:
        """Monitor services and restart if needed."""
        while self.running:
            try:
                for service in self.services.values():
                    if service.status == "running" and not service.is_running():
                        logger.error(f"{service.config.name} has crashed")
                        
                        # Attempt restart if configured
                        if (service.config.restart_on_failure and 
                            service.restart_count < service.config.max_restarts):
                            
                            service.restart_count += 1
                            logger.info(f"Attempting to restart {service.config.name}",
                                       attempt=service.restart_count)
                            
                            success = await service.start()
                            if not success:
                                logger.error(f"Failed to restart {service.config.name}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error("Error in service monitor", exception=e)
                await asyncio.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        return {
            "manager_running": self.running,
            "services": {
                name: service.get_status() 
                for name, service in self.services.items()
            },
            "timestamp": datetime.now()
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all services."""
        health_status = {}
        
        for name, service in self.services.items():
            if service.is_running():
                # TODO: Implement actual health checks (HTTP, etc.)
                health_status[name] = True
            else:
                health_status[name] = False
        
        return health_status