"""Logging configuration for Weaver."""

import logging
import logging.config
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_console: bool = True
) -> None:
    """
    Setup logging configuration for Weaver.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        enable_console: Whether to enable console logging
    """
    
    # Create logs directory if log_file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {},
        "root": {
            "level": log_level,
            "handlers": []
        }
    }
    
    # Add console handler
    if enable_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
        config["root"]["handlers"].append("console")
    
    # Add file handler
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": str(log_file),
            "mode": "a",
            "encoding": "utf-8"
        }
        config["root"]["handlers"].append("file")
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Set specific logger levels
    logging.getLogger("weaver").setLevel(log_level)
    logging.getLogger("sqlalchemy").setLevel("WARNING")
    logging.getLogger("urllib3").setLevel("WARNING") 


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(f"weaver.{name}")
