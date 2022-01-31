"""
overwatch.py

Utility class for creating a centralized/standardized python logger, with a sane default format, at the appropriate
logging level.
"""
import logging
import sys
import warnings


# Constants - for Formatting
FORMATTER = logging.Formatter("[*] %(asctime)s - %(name)s - %(levelname)s :: %(message)s", datefmt="%m/%d [%H:%M:%S]")


def get_overwatch(level: int, name: str) -> logging.Logger:
    """
    Initialize logging.Logger with the appropriate name, console, and file handlers.

    :param level: Default logging level --> should usually be INFO (inherited from entry point).
    :param name: Name of the top-level logger --> should reflect entry point.

    :return: Default logger object :: logging.Logger
    """
    # Create Default Logger & add Handlers
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(level)

    # Create Console Handler --> Write to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    logger.addHandler(console_handler)

    # Silence PyTorch Lightning loggers --> only get WARNING+ messages, suppress DataLoader warnings...
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    # Do not propagate by default...
    logger.propagate = False
    return logger
