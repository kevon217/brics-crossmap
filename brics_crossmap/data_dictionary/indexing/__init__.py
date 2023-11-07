import logging
from brics_crossmap.utils.logger.config_logging import setup_log, log, copy_log

# CREATE LOGGER
setup_log()
indexing_logger = logging.getLogger("indexing_logger")
indexing_logger.info("Initiating indexing logger.")
