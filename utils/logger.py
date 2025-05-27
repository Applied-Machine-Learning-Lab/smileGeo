import logging
import time

class Mylogger():
    def __init__(self, name='my_logger', level=logging.DEBUG, out_file='saved/train_main.log'):
        # Create a logger object
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler for outputting logs
        file_handler = logging.FileHandler(out_file, encoding='utf8')
        file_handler.setLevel(logging.DEBUG)  # Only log infos and above to the file

        # Create a console handler for outputting logs to stdout
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Log everything to console

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log(self, msg, level='info'):
        level = level.lower()
        if level == 'debug':
            self.logger.debug(msg)
            return True
        if level == 'info':
            self.logger.info(msg)
            return True
        if level == 'error':
            self.logger.error(msg)
            return True
        else:
            print('logger error!')
            return False
