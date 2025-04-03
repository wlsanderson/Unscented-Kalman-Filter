from UKF.context import Context
from UKF.logger import Logger
from UKF.data_processor import DataProcessor
from pathlib import Path

def run():
    launch_log = Path("launch_data/pelicanator_launch_2.csv")
    logger = Logger()
    data_processor = DataProcessor(launch_log, logger=logger)
    context = Context(data_processor)
    run_data_loop(context)

def run_data_loop(context):
    while True:
        context.update()
        
        if context.shutdown_requested:
            break


if __name__ == "__main__":
    run()