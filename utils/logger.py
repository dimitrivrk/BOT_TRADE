"""
Logger centralisé pour le bot de trading.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
import yaml


def setup_logger(
    name: str,
    config_path: str = "config/config.yaml",
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Configure et retourne un logger avec rotation de fichiers.

    Args:
        name: Nom du module (ex: 'data.collector')
        config_path: Chemin vers le fichier de config
        level: Override du niveau de log

    Returns:
        Logger configuré
    """
    # Charger la config
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        log_config = config.get("logging", {})
        log_level = level or log_config.get("level", "INFO")
        log_file = log_config.get("file", "logs/bot.log")
        max_size_mb = log_config.get("max_size_mb", 100)
        backup_count = log_config.get("backup_count", 10)
    except FileNotFoundError:
        log_level = level or "INFO"
        log_file = "logs/bot.log"
        max_size_mb = 100
        backup_count = 10

    # Créer le dossier logs si nécessaire
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Éviter les handlers dupliqués
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # Handler fichier avec rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Raccourci pour récupérer un logger déjà configuré."""
    return logging.getLogger(name)
