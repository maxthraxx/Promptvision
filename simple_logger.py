import logging
import os
from logging.handlers import RotatingFileHandler

def create_logger(name, log_dir='log', max_size_mb=100, console_output=False):
    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger object
    logger = logging.getLogger(name)

    # Set the level of the logger (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create a handler for rotating file
    file_path = os.path.join(log_dir, f'{name}.log')
    file_handler = RotatingFileHandler(file_path, maxBytes=max_size_mb * 1024 * 1024, backupCount=1)

    # Create a formatter for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line:%(lineno)d - %(message)s")

    # Set the formatter for the handler
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Optionally, add console output handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Return the logger object
    return logger
