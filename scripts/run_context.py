from UKF.context import Context
from UKF.logger import Logger
from UKF.data_processor import DataProcessor
from pathlib import Path


launch_log = Path("launch_data/pelicanator_launch_2.csv")
logger = Logger()
data_processor = DataProcessor(launch_log)
context = Context(data_processor, logger=logger)
context.update()
