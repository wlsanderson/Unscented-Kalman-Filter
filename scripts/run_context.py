from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path


launch_log = Path("launch_data/pelicanator_launch_2.csv")
plotter = Plotter()
data_processor = DataProcessor(launch_log)
context = Context(data_processor, plotter=plotter)
for i in range(50):
    context.update()
