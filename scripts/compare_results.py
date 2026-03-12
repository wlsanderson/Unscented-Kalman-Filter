import pandas as pd
import numpy as np

py = pd.read_csv("C/test_output_python.csv")
c = pd.read_csv("C/test_output.csv")

cumdt_py = py["dt"].cumsum()
cumdt_c = c["dt"].cumsum()

print(f"{'t(s)':>6}  {'pos_err(m)':>12}  {'vel_err(m/s)':>14}  {'quat_err':>10}  {'alt_py(m)':>10}  {'alt_c(m)':>10}")
print("-" * 80)

for t in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0]:
    idx_py = (cumdt_py - t).abs().idxmin()
    idx_c = (cumdt_c - t).abs().idxmin()

    dp = np.sqrt(
        (py.loc[idx_py, "pos_x"] - c.loc[idx_c, "pos_x"]) ** 2
        + (py.loc[idx_py, "pos_y"] - c.loc[idx_c, "pos_y"]) ** 2
        + (py.loc[idx_py, "pos_z"] - c.loc[idx_c, "pos_z"]) ** 2
    )
    dv = np.sqrt(
        (py.loc[idx_py, "vel_x"] - c.loc[idx_c, "vel_x"]) ** 2
        + (py.loc[idx_py, "vel_y"] - c.loc[idx_c, "vel_y"]) ** 2
        + (py.loc[idx_py, "vel_z"] - c.loc[idx_c, "vel_z"]) ** 2
    )
    dq = np.sqrt(
        (py.loc[idx_py, "qw"] - c.loc[idx_c, "qw"]) ** 2
        + (py.loc[idx_py, "qx"] - c.loc[idx_c, "qx"]) ** 2
        + (py.loc[idx_py, "qy"] - c.loc[idx_c, "qy"]) ** 2
        + (py.loc[idx_py, "qz"] - c.loc[idx_c, "qz"]) ** 2
    )
    alt_py = py.loc[idx_py, "pos_z"]
    alt_c = c.loc[idx_c, "pos_z"]

    print(f"{t:6.1f}  {dp:12.4f}  {dv:14.4f}  {dq:10.6f}  {alt_py:10.2f}  {alt_c:10.2f}")
