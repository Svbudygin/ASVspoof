import logging
import logging.config
import json
from pathlib import Path

py_logger = logging.getLogger("")


def _ensure_logger_configured():
    py_logger.handlers.clear()
    if not py_logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        py_logger.addHandler(handler)
