import logging
import config
import os


PROJECT_DIR = config.PROJECT_DIR
log_filename = os.path.join(PROJECT_DIR, 'logs/info.log')
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_filename,
                    filemode='a')

def get_logger(logger_name):
    return logging.getLogger(logger_name)