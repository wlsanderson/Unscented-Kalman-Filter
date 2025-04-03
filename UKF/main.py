from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path

def run():
    launch_log = Path("launch_data/pelicanator_launch_2.csv")
    plotter = Plotter()
    data_processor = DataProcessor(launch_log, cutoff = 200)
    context = Context(data_processor, plotter=plotter)
    run_data_loop(context)

def run_data_loop(context):
    while True:
        context.update()
        
        if context.shutdown_requested:
            break


if __name__ == "__main__":
    run()