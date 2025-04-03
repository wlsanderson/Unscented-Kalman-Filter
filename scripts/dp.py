from UKF.data_processor import DataProcessor

dp = DataProcessor("launch_data/pelicanator_launch_2.csv")
print(dp.get_initial_vals())