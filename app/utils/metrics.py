"""
Sistema de métricas de rendimiento para Bibliometría App.
Proporciona monitoreo y métricas de rendimiento en tiempo real.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from app.utils.logger import get_logger, log_performance_metrics
from app.config import settings


@dataclass
class PerformanceMetric:
    """Métrica de rendimiento individual."""
    operation: str
    duration: float
    timestamp: datetime
    success: bool
    additional_data: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor de rendimiento centralizado."""
    
    def __init__(self):
        self.logger = get_logger("performance_monitor")
        self.metrics: deque = deque(maxlen=10000)  # Mantener últimas 10k métricas
        self.lock = threading.Lock()
        self._start_time = time.time()
    
    def record_metric(
        self,
        operation: str,
        duration: float,
        success: bool,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Registrar una métrica de rendimiento."""
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=datetime.utcnow(),
            success=success,
            additional_data=additional_data or {}
        )
        
        with self.lock:
            self.metrics.append(metric)
        
        # Log de la métrica
        log_performance_metrics(
            operation=operation,
            duration=duration,
            success=success,
            additional_data=additional_data
        )
    
    def get_stats(self, operation: Optional[str] = None, last_minutes: int = 60) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=last_minutes)
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics 
                if m.timestamp >= cutoff_time and (operation is None or m.operation == operation)
            ]
        
        if not recent_metrics:
            return {
                "operation": operation or "all",
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "p95_duration": 0.0,
                "p99_duration": 0.0
            }
        
        durations = [m.duration for m in recent_metrics]
        successful = [m for m in recent_metrics if m.success]
        
        durations.sort()
        n = len(durations)
        
        return {
            "operation": operation or "all",
            "total_requests": len(recent_metrics),
            "success_rate": len(successful) / len(recent_metrics) * 100,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p95_duration": durations[int(n * 0.95)] if n > 0 else 0,
            "p99_duration": durations[int(n * 0.99)] if n > 0 else 0,
            "time_range_minutes": last_minutes
        }
    
    def get_uptime(self) -> float:
        """Obtener tiempo de actividad en segundos."""
        return time.time() - self._start_time
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado de salud del sistema."""
        stats = self.get_stats(last_minutes=5)
        uptime = self.get_uptime()
        
        # Determinar estado de salud basado en métricas
        health_status = "healthy"
        if stats["success_rate"] < 95:
            health_status = "degraded"
        if stats["success_rate"] < 80:
            health_status = "unhealthy"
        
        return {
            "status": health_status,
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "recent_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }


class PerformanceTimer:
    """Context manager para medir tiempo de ejecución."""
    
    def __init__(self, operation: str, monitor: Optional[PerformanceMonitor] = None):
        self.operation = operation
        self.monitor = monitor or performance_monitor
        self.start_time = None
        self.end_time = None
        self.success = True
        self.additional_data = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is not None:
            self.success = False
            self.additional_data["error"] = str(exc_val)
        
        self.monitor.record_metric(
            operation=self.operation,
            duration=duration,
            success=self.success,
            additional_data=self.additional_data
        )
    
    def add_data(self, key: str, value: Any) -> None:
        """Agregar datos adicionales a la métrica."""
        self.additional_data[key] = value


def measure_performance(operation: str):
    """Decorador para medir rendimiento de funciones."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with PerformanceTimer(operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimiter:
    """Limitador de velocidad simple."""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Verificar si la petición está permitida."""
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self.lock:
            # Limpiar peticiones antiguas
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier] 
                if req_time > cutoff
            ]
            
            # Verificar límite
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            
            # Agregar nueva petición
            self.requests[identifier].append(now)
            return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Obtener número de peticiones restantes."""
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self.lock:
            recent_requests = [
                req_time for req_time in self.requests[identifier] 
                if req_time > cutoff
            ]
            return max(0, self.max_requests - len(recent_requests))


# Instancias globales
performance_monitor = PerformanceMonitor()
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_per_minute,
    window_seconds=60
)


def get_performance_stats(operation: Optional[str] = None) -> Dict[str, Any]:
    """Obtener estadísticas de rendimiento."""
    return performance_monitor.get_stats(operation)


def get_health_status() -> Dict[str, Any]:
    """Obtener estado de salud del sistema."""
    return performance_monitor.get_health_status()


def check_rate_limit(identifier: str) -> bool:
    """Verificar límite de velocidad."""
    return rate_limiter.is_allowed(identifier)


def get_remaining_requests(identifier: str) -> int:
    """Obtener peticiones restantes."""
    return rate_limiter.get_remaining_requests(identifier)




