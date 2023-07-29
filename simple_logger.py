# Import the logging module
import logging

# Define a function that creates and returns a logger object
def create_logger(name):
    # Create a logger object
    logger = logging.getLogger(name)

    # Set the level of the logger (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create a handler for outputting to the console
    console_handler = logging.StreamHandler()

    # Create a formatter for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line:%(lineno)d - %(message)s")

    # Set the formatter for the handler
    console_handler.setFormatter(formatter)

    # Check if the logger already has a handler
    if not logger.hasHandlers():
        # Add the handler to the logger
        logger.addHandler(console_handler)

    # Return the logger object
    return logger
