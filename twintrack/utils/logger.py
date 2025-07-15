import logging
import sys

class Logger:
    """
    Simple logger utility for info, warning, error, and file logging.
    """
    def __init__(self, log_file=None):
        self.logger = logging.getLogger('TwinTrackLogger')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg) 