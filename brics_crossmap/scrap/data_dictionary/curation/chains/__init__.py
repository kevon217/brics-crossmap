import logging
from brics_crossmap.utils.logger.config_logging import setup_log, log, copy_log

# CREATE LOGGER
setup_log()
cur_logger = logging.getLogger("curation_logger")
# logger.propagate = False
cur_logger.info("Initiating brics_tools curation logger.")
