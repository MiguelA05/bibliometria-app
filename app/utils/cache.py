"""
Sistema de caché para Bibliometría App.
Proporciona caché en memoria y Redis para optimizar consultas frecuentes.
"""

import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from app.config import settings
from app.utils.logger import get_logger


class CacheInterface:
    """Interfaz para sistemas de caché."""
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establecer valor en el caché."""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Eliminar valor del caché."""
        raise NotImplementedError
    
    def clear(self) -> bool:
        """Limpiar todo el caché."""
        raise NotImplementedError


class MemoryCache(CacheInterface):
    """Caché en memoria simple."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger("memory_cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Verificar expiración
        if entry["expires_at"] and time.time() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        # Actualizar último acceso
        entry["last_accessed"] = time.time()
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establecer valor en el caché."""
        try:
            # Limpiar caché si está lleno
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            expires_at = None
            if ttl:
                expires_at = time.time() + ttl
            
            self.cache[key] = {
                "value": value,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "expires_at": expires_at
            }
            
            return True
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Eliminar valor del caché."""
        try:
            if key in self.cache:
                del self.cache[key]
            return True
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Limpiar todo el caché."""
        try:
            self.cache.clear()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
    
    def _evict_oldest(self) -> None:
        """Eliminar la entrada más antigua."""
        if not self.cache:
            return
        
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]["last_accessed"]
        )
        del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        now = time.time()
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if entry["expires_at"] and entry["expires_at"] < now
        )
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "max_size": self.max_size,
            "usage_percentage": (total_entries / self.max_size) * 100
        }


class RedisCache(CacheInterface):
    """Caché usando Redis."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self.logger = get_logger("redis_cache")
        self._redis = None
        self._connect()
    
    def _connect(self) -> None:
        """Conectar a Redis."""
        try:
            import redis
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            # Probar conexión
            self._redis.ping()
            self.logger.info("Connected to Redis cache")
        except ImportError:
            self.logger.warning("Redis not available, falling back to memory cache")
            self._redis = None
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self._redis = None
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        if not self._redis:
            return None
        
        try:
            value = self._redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            self.logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establecer valor en el caché."""
        if not self._redis:
            return False
        
        try:
            serialized_value = json.dumps(value, default=str)
            if ttl:
                return self._redis.setex(key, ttl, serialized_value)
            else:
                return self._redis.set(key, serialized_value)
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Eliminar valor del caché."""
        if not self._redis:
            return False
        
        try:
            return bool(self._redis.delete(key))
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Limpiar todo el caché."""
        if not self._redis:
            return False
        
        try:
            self._redis.flushdb()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False


class CacheManager:
    """Gestor de caché con fallback automático."""
    
    def __init__(self):
        self.logger = get_logger("cache_manager")
        self.primary_cache: Optional[CacheInterface] = None
        self.fallback_cache: Optional[CacheInterface] = None
        self._initialize_caches()
    
    def _initialize_caches(self) -> None:
        """Inicializar sistemas de caché."""
        if settings.cache_enabled:
            # Intentar Redis primero
            redis_cache = RedisCache()
            if redis_cache._redis:
                self.primary_cache = redis_cache
                self.logger.info("Using Redis as primary cache")
            else:
                self.primary_cache = MemoryCache()
                self.logger.info("Using memory cache as primary")
        else:
            self.primary_cache = MemoryCache()
            self.logger.info("Cache disabled, using memory cache")
        
        # Fallback siempre en memoria
        self.fallback_cache = MemoryCache()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché con fallback."""
        # Intentar caché primario
        if self.primary_cache:
            value = self.primary_cache.get(key)
            if value is not None:
                return value
        
        # Fallback a caché secundario
        if self.fallback_cache:
            return self.fallback_cache.get(key)
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establecer valor en el caché con fallback."""
        success = False
        
        # Intentar caché primario
        if self.primary_cache:
            success = self.primary_cache.set(key, value, ttl)
        
        # Fallback a caché secundario
        if not success and self.fallback_cache:
            success = self.fallback_cache.set(key, value, ttl)
        
        return success
    
    def delete(self, key: str) -> bool:
        """Eliminar valor del caché."""
        success = False
        
        if self.primary_cache:
            success = self.primary_cache.delete(key)
        
        if self.fallback_cache:
            success = self.fallback_cache.delete(key) or success
        
        return success
    
    def clear(self) -> bool:
        """Limpiar todo el caché."""
        success = True
        
        if self.primary_cache:
            success = self.primary_cache.clear() and success
        
        if self.fallback_cache:
            success = self.fallback_cache.clear() and success
        
        return success


def generate_cache_key(query: str, max_articles: int, filters: Optional[Dict[str, Any]] = None) -> str:
    """Generar clave de caché única para una consulta."""
    # Crear hash de los parámetros
    params = {
        "query": query,
        "max_articles": max_articles,
        "filters": filters or {}
    }
    
    # Ordenar filtros para consistencia
    if params["filters"]:
        params["filters"] = dict(sorted(params["filters"].items()))
    
    # Generar hash
    params_str = json.dumps(params, sort_keys=True)
    hash_obj = hashlib.md5(params_str.encode())
    
    return f"openalex_search:{hash_obj.hexdigest()}"


def cache_openalex_result(
    query: str,
    max_articles: int,
    filters: Optional[Dict[str, Any]],
    result: Dict[str, Any],
    ttl: Optional[int] = None
) -> bool:
    """Cachear resultado de OpenAlex."""
    cache_key = generate_cache_key(query, max_articles, filters)
    ttl = ttl or settings.cache_ttl
    
    return cache_manager.set(cache_key, result, ttl)


def get_cached_openalex_result(
    query: str,
    max_articles: int,
    filters: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Obtener resultado cacheado de OpenAlex."""
    cache_key = generate_cache_key(query, max_articles, filters)
    return cache_manager.get(cache_key)


# Instancia global del gestor de caché
cache_manager = CacheManager()





