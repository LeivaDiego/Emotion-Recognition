import logging

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name (str): The name of the logger. Defaults to the module's name.
        level (int): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        logger.setLevel(level)
        # Create a console handler with a specific format
        formatter = logging.Formatter("[%(levelname)s] %(message)s")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False  # Prevent propagation to the root logger

    return logger
