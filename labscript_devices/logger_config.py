import os
import logging

# Log file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "devices.log")

# Create logger
logger = logging.getLogger("BS_34")
logger.setLevel(logging.DEBUG)

# Ensure fresh start and no duplicate handlers
if logger.hasHandlers():
    logger.handlers.clear()

# File handler (truncate on each run)
handler = logging.FileHandler(LOG_FILE, mode="w")
handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s"
)
handler.setFormatter(formatter)

# Attach handler
logger.addHandler(handler)

# Initial log entry
logger.info("=================== Logger initialized successfully ===================")
