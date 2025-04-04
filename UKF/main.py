from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path
from UKF.constants import States

def run():
    launch_log = Path("launch_data/pelicanator_launch_2.csv")
    min_r = 5002
    max_r=30000
    plot_state = [States.VELOCITY.value]

    plotter = Plotter(state_index=plot_state, file_path=launch_log, min_r=min_r, max_r=max_r)
    data_processor = DataProcessor(launch_log, min_r=min_r, max_r=max_r)
    context = Context(data_processor, plotter=plotter)
    run_data_loop(context)

def run_data_loop(context: Context):
    while True:
        context.update()
        
        if context.shutdown_requested:
            break


if __name__ == "__main__":
    run()