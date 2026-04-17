from __future__ import annotations

import logging
import os

from pythonjsonlogger.json import JsonFormatter

from app.config import settings

LOG_DIR = settings.LOG_DIR
LOG_FILE = os.path.join(LOG_DIR, settings.LOG_FILE)


class CustomJsonFormatter(JsonFormatter):
    """Extend the JSON formatter to include standard fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = record.created
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno


def setup_logger(name: str = "fraud_api") -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Avoid duplicate handlers on reload
    if logger.handlers:
        return logger

    # Console handler — human-readable in DEBUG, JSON otherwise
    console = logging.StreamHandler()
    if settings.DEBUG:
        console_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        console_fmt = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(logger)s %(message)s"
        )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler — always JSON
    file_handler = logging.FileHandler(LOG_FILE)
    json_fmt = CustomJsonFormatter(
        "%(timestamp)s %(level)s %(logger)s %(message)s"
    )
    file_handler.setFormatter(json_fmt)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
