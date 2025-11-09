"""Logging configuration for gym-tracker application."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "gymtracker",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Configure and return application logger.
    
    Args:
        name: Logger name (default: "gymtracker")
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only logs to console.
        console: Whether to log to console (default: True)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(level="DEBUG", log_file="logs/app.log")
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger instance
logger = setup_logger()


def log_function_call(func):
    """Decorator to log function calls with arguments.
    
    Example:
        >>> @log_function_call
        ... def my_function(x, y):
        ...     return x + y
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func_name} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper
