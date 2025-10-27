"""
Sistema de manejo de errores robusto para Bibliometría App.
Proporciona manejo consistente de errores y excepciones personalizadas.
"""

import traceback
from typing import Any, Dict, Optional, Union
from fastapi import HTTPException
from requests.exceptions import RequestException, Timeout, ConnectionError
from app.utils.logger import get_logger


class BibliometriaAppError(Exception):
    """Excepción base para errores de la aplicación."""
    
    def __init__(self, message: str, error_code: str = "GENERIC_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class OpenAlexError(BibliometriaAppError):
    """Error específico de OpenAlex."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "OPENALEX_ERROR", details)
        self.status_code = status_code


class ValidationError(BibliometriaAppError):
    """Error de validación de entrada."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field


class RateLimitError(BibliometriaAppError):
    """Error de límite de velocidad."""
    
    def __init__(self, message: str = "Límite de velocidad excedido", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RATE_LIMIT_ERROR", details)


class CSVExportError(BibliometriaAppError):
    """Error en exportación de CSV."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CSV_EXPORT_ERROR", details)
        self.file_path = file_path


class ErrorHandler:
    """Manejador centralizado de errores."""
    
    def __init__(self):
        self.logger = get_logger("error_handler")
    
    def handle_openalex_error(self, error: RequestException, query: str) -> OpenAlexError:
        """Manejar errores de OpenAlex."""
        if isinstance(error, Timeout):
            message = "Timeout al conectar con OpenAlex. Intente nuevamente."
            status_code = 408
        elif isinstance(error, ConnectionError):
            message = "Error de conexión con OpenAlex. Verifique su conexión a internet."
            status_code = 503
        else:
            message = f"Error de OpenAlex: {str(error)}"
            status_code = getattr(error.response, 'status_code', 500) if hasattr(error, 'response') else 500
        
        details = {
            "query": query,
            "original_error": str(error),
            "error_type": type(error).__name__
        }
        
        self.logger.error(
            "OpenAlex error occurred",
            error=message,
            query=query,
            status_code=status_code,
            details=details
        )
        
        return OpenAlexError(message, status_code, details)
    
    def handle_validation_error(self, error: Exception, field: Optional[str] = None) -> ValidationError:
        """Manejar errores de validación."""
        message = str(error)
        details = {
            "field": field,
            "original_error": str(error),
            "error_type": type(error).__name__
        }
        
        self.logger.warning(
            "Validation error occurred",
            error=message,
            field=field,
            details=details
        )
        
        return ValidationError(message, field, details)
    
    def handle_csv_export_error(self, error: Exception, file_path: Optional[str] = None) -> CSVExportError:
        """Manejar errores de exportación CSV."""
        message = f"Error al exportar CSV: {str(error)}"
        details = {
            "file_path": file_path,
            "original_error": str(error),
            "error_type": type(error).__name__
        }
        
        self.logger.error(
            "CSV export error occurred",
            error=message,
            file_path=file_path,
            details=details
        )
        
        return CSVExportError(message, file_path, details)
    
    def handle_unexpected_error(self, error: Exception, context: Optional[str] = None) -> BibliometriaAppError:
        """Manejar errores inesperados."""
        message = f"Error inesperado: {str(error)}"
        details = {
            "context": context,
            "original_error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc()
        }
        
        self.logger.error(
            "Unexpected error occurred",
            error=message,
            context=context,
            details=details
        )
        
        return BibliometriaAppError(message, "UNEXPECTED_ERROR", details)


def create_error_response(error: BibliometriaAppError) -> Dict[str, Any]:
    """Crear respuesta de error estructurada."""
    return {
        "error": True,
        "error_code": error.error_code,
        "message": error.message,
        "details": error.details,
        "timestamp": None  # Se llenará en el endpoint
    }


def convert_to_http_exception(error: BibliometriaAppError) -> HTTPException:
    """Convertir error de aplicación a HTTPException."""
    status_code_map = {
        "VALIDATION_ERROR": 400,
        "RATE_LIMIT_ERROR": 429,
        "OPENALEX_ERROR": 502,
        "CSV_EXPORT_ERROR": 500,
        "UNEXPECTED_ERROR": 500,
        "GENERIC_ERROR": 500
    }
    
    status_code = status_code_map.get(error.error_code, 500)
    
    return HTTPException(
        status_code=status_code,
        detail=create_error_response(error)
    )


# Instancia global del manejador de errores
error_handler = ErrorHandler()


def handle_exceptions(func):
    """Decorador para manejo automático de excepciones."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            raise convert_to_http_exception(e)
        except OpenAlexError as e:
            raise convert_to_http_exception(e)
        except CSVExportError as e:
            raise convert_to_http_exception(e)
        except BibliometriaAppError as e:
            raise convert_to_http_exception(e)
        except Exception as e:
            app_error = error_handler.handle_unexpected_error(e, func.__name__)
            raise convert_to_http_exception(app_error)
    
    return wrapper


def safe_execute(func, *args, **kwargs):
    """
    Ejecutar función de forma segura con manejo de errores.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        **kwargs: Argumentos con nombre
        
    Returns:
        Tupla (resultado, error) donde error es None si fue exitoso
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_handler.logger.error(
            "Safe execute failed",
            function=func.__name__,
            error=str(e),
            error_type=type(e).__name__
        )
        return None, e




