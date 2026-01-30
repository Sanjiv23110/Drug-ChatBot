"""
Structured logging configuration for production.

Features:
- JSON-formatted logs for log aggregation
- Correlation IDs for request tracing
- Different log levels by component
- File and console handlers
"""
import logging
import sys
import json
from datetime import datetime
from typing import Optional
from pathlib import Path
import os


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Output format:
        {"timestamp": "...", "level": "INFO", "message": "...", "module": "..."}
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'drug_name'):
            log_data["drug_name"] = record.drug_name
        if hasattr(record, 'document_hash'):
            log_data["document_hash"] = record.document_hash
        if hasattr(record, 'correlation_id'):
            log_data["correlation_id"] = record.correlation_id
            
        return json.dumps(log_data)


class PrettyFormatter(logging.Formatter):
    """
    Human-readable formatter for development.
    
    Output format:
        2024-01-29 10:30:00 | INFO | module:function:line | message
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        msg = (
            f"{timestamp} | "
            f"{color}{record.levelname:8}{self.RESET} | "
            f"{record.module}:{record.funcName}:{record.lineno} | "
            f"{record.getMessage()}"
        )
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
            
        return msg


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (for production) or pretty format (dev)
        log_file: Optional file path for log output
        
    Returns:
        Configured root logger
        
    Example:
        logger = setup_logging(level="DEBUG", json_format=False)
        logger.info("Application started", extra={"drug_name": "aspirin"})
    """
    # Get or create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = PrettyFormatter()
    
    # Console handler - with UTF-8 encoding for Windows
    console_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return logging.getLogger(name)


# Context class for adding correlation IDs
class LogContext:
    """
    Context manager for adding correlation IDs to logs.
    
    Usage:
        with LogContext(correlation_id="abc123"):
            logger.info("Processing request")  # Includes correlation_id
    """
    
    _current_context = {}
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.previous_context = {}
    
    def __enter__(self):
        self.previous_context = LogContext._current_context.copy()
        LogContext._current_context.update(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        LogContext._current_context = self.previous_context
        return False
    
    @classmethod
    def get(cls, key: str, default=None):
        return cls._current_context.get(key, default)


# Initialize default logging on import
if os.getenv("LOG_FORMAT") == "json":
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        json_format=True,
        log_file=os.getenv("LOG_FILE")
    )
else:
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        json_format=False
    )
