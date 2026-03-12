#!/usr/bin/env python3
"""
ESKF C-port validation script.

1. Pre-processes the same launch data as the Python ESKF
2. Exports test_input.csv  (one row per timestep: dt + calibrated sensor data)
3. Runs the Python ESKF on the same data for reference
4. Compiles & runs the C test harness
5. Compares Python vs C output
"""
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Configuration ─────────────────────────────────────────────────────────────
LAUNCH_FOLDER = Path("launch_data/lil_frank")
MIN_T = 1700
MAX_T = 1760

C_DIR = Path("C")
C_TEST_SRC = C_DIR / "test_eskf.c"
C_SOURCES = [
    C_DIR / "ukf_data_processing" / "matrixhelper.c",
    C_DIR / "ukf_data_processing" / "kalman_filter_config.c",
    C_DIR / "ukf_data_processing" / "state_machine.c",
    C_DIR / "ukf_data_processing" / "ukf_functions.c",
    C_DIR / "ukf_data_processing" / "unscented_kalman_filter.c",
    C_TEST_SRC,
]
C_EXE = C_DIR / "test_eskf.exe"

MEASUREMENT_FIELDS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "pressure",
    "mag_x", "mag_y", "mag_z",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sensor_csv(path: Path) -> pd.DataFrame:
    headers = pd.read_csv(path, nrows=0)
    needed = list((set(MEASUREMENT_FIELDS) | {"timestamp"}) & set(headers.columns))
    return pd.read_csv(path, usecols=needed)


def fix_mag_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    replace_idx = list(range(10, len(df), 11))
    cols = df.columns[1:]
    df.loc[replace_idx, cols] = df.loc[[i - 1 for i in replace_idx], cols].values
    return df


def preprocess_data(launch_folder: Path, min_t: float, max_t: float):
    """Load, merge, calibrate sensor CSVs — mirrors DataProcessor.fetch()."""
    bmp_df = load_sensor_csv(launch_folder / "BMP581_data.csv")
    imu_df = load_sensor_csv(launch_folder / "ICM45686_data.csv")
    mag_df = fix_mag_data(load_sensor_csv(launch_folder / "MMC5983MA_data.csv"))

    merged = pd.concat([bmp_df, imu_df, mag_df], ignore_index=True)
    merged.sort_values("timestamp", inplace=True, ignore_index=True)
    merged = merged[(merged["timestamp"] > min_t) & (merged["timestamp"] < max_t)]

    # Load calibration
    cal_file = launch_folder / "calibration.yaml"
    if cal_file.exists():
        with open(cal_file, "r") as f:
            cal_root = yaml.safe_load(f) or {}
        cal = cal_root.get("calibration", {})
        acc_offset = np.array(cal.get("accel_offset", [0, 0, 0]), dtype=np.float64)
        gyro_offset = np.array(cal.get("gyro_offset", [0, 0, 0]), dtype=np.float64)
        mag_offset = np.array(cal.get("mag_offset", [0, 0, 0]), dtype=np.float64)
        mag_scale = np.array(cal.get("mag_scale", np.eye(3).tolist()), dtype=np.float64)
    else:
        acc_offset = np.zeros(3, dtype=np.float64)
        gyro_offset = np.zeros(3, dtype=np.float64)
        mag_offset = np.zeros(3, dtype=np.float64)
        mag_scale = np.eye(3, dtype=np.float64)

    # Forward-fill to create complete rows (mirrors DataProcessor.fetch())
    last_data = np.full(len(MEASUREMENT_FIELDS), np.nan)
    rows_out = []
    last_ts = None

    for _, row in merged.iterrows():
        ts = row["timestamp"]
        for i, field in enumerate(MEASUREMENT_FIELDS):
            val = row.get(field, np.nan)
            if pd.notna(val):
                last_data[i] = val

        if np.any(np.isnan(last_data)):
            continue

        dt = 0.0 if last_ts is None else float(ts - last_ts)
        last_ts = ts

        # Copy and calibrate
        meas = last_data.copy()
        # accel calibration (indices 0:3)
        meas[0:3] -= acc_offset
        # gyro calibration (indices 3:6)
        meas[3:6] -= gyro_offset
        # mag calibration (hard/soft iron) (indices 7:10)
        mag_raw = meas[7:10]
        mag_cal = (mag_raw - mag_offset) @ mag_scale
        # NOTE: we pass the calibrated (but NOT normalised) mag to the C code,
        # because the C code normalises internally via eskf_set_measurement().
        meas[7:10] = mag_cal

        rows_out.append([dt] + meas.tolist())

    columns = ["dt"] + MEASUREMENT_FIELDS
    df = pd.DataFrame(rows_out, columns=columns)
    return df


def run_python_eskf(input_df: pd.DataFrame):
    """Run the Python ESKF on the same pre-processed data for reference."""
    # We import the ESKF modules here so import errors don't kill the whole script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from UKF.eskf import ESKF
    from UKF.eskf_functions import (
        nominal_predict, nominal_predict_init,
        error_state_jacobian, error_state_jacobian_init,
        process_noise_matrix,
        measurement_function, measurement_jacobian,
        _imu_to_vehicle, R_IMU_TO_VEHICLE,
    )
    from UKF.constants import (
        ESKF_NOMINAL_DIM, ESKF_ERROR_DIM, ESKF_MEASUREMENT_DIM,
        ESKF_INITIAL_STATE_ESTIMATE, ESKF_INITIAL_STATE_COV,
        ESKFProcessCovariance, ESKFMeasurementNoise,
        GRAVITY,
    )
    import quaternion as q

    eskf = ESKF(dim_nom=ESKF_NOMINAL_DIM, dim_err=ESKF_ERROR_DIM, dim_z=ESKF_MEASUREMENT_DIM)
    eskf.x_nom = np.copy(ESKF_INITIAL_STATE_ESTIMATE).astype(np.float64)
    eskf.P = np.copy(ESKF_INITIAL_STATE_COV).astype(np.float64)
    eskf.measurement_func = measurement_function
    eskf.measurement_jacobian_func = measurement_jacobian

    INIT_DURATION = 0.5

    # State tracking
    in_init = True
    elapsed = 0.0
    accel_accum = np.zeros(3, dtype=np.float64)
    gyro_accum = np.zeros(3, dtype=np.float64)
    n_samples = 0
    accel_bias = np.zeros(3, dtype=np.float64)
    gyro_bias = np.zeros(3, dtype=np.float64)
    initial_pressure = None
    initial_mag = None
    initial_quat = None

    # Python ESKF helper: calculate_initial_orientation equivalent
    def calc_init_orientation(acc_sensor, mag_sensor):
        """Mirrors the C calculate_initial_orientation logic."""
        norm_acc = np.linalg.norm(acc_sensor)
        norm_mag = np.linalg.norm(mag_sensor)
        # sensor -> vehicle rotation for accel
        s2 = 1.0 / np.sqrt(2)
        acc_vehicle = np.array([
            (acc_sensor[0] * s2 - acc_sensor[1] * s2) / norm_acc,
            (acc_sensor[0] * s2 + acc_sensor[1] * s2) / norm_acc,
            acc_sensor[2] / norm_acc,
        ])
        # mag in vehicle frame (z-flip from sensor)
        mag_v = np.array([
            mag_sensor[0] / norm_mag,
            mag_sensor[1] / norm_mag,
            -mag_sensor[2] / norm_mag,
        ])

        roll = np.arctan2(acc_vehicle[1], acc_vehicle[2])
        pitch = np.arctan2(-acc_vehicle[0],
                           np.sqrt(acc_vehicle[1]**2 + acc_vehicle[2]**2))
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        mx2 = mag_v[0] * cp + mag_v[2] * sp
        my2 = mag_v[0] * sr * sp + mag_v[1] * cr - mag_v[2] * sr * cp
        yaw = np.arctan2(-my2, mx2)

        cr2, sr2 = np.cos(roll/2), np.sin(roll/2)
        cp2, sp2 = np.cos(pitch/2), np.sin(pitch/2)
        cy2, sy2 = np.cos(yaw/2), np.sin(yaw/2)

        qw = cr2*cp2*cy2 + sr2*sp2*sy2
        qx = sr2*cp2*cy2 - cr2*sp2*sy2
        qy = cr2*sp2*cy2 + sr2*cp2*sy2
        qz = cr2*cp2*sy2 - sr2*sp2*cy2

        init_q = q.quaternion(qw, qx, qy, qz)
        mag_v_q = q.quaternion(0, mag_v[0], mag_v[1], mag_v[2])
        conj_q = q.quaternion(qw, -qx, -qy, -qz)
        mag_world_q = init_q * mag_v_q * conj_q
        mag_world = np.array([mag_world_q.x, mag_world_q.y, mag_world_q.z])
        return init_q, mag_world

    results = []
    for idx, row in input_df.iterrows():
        dt = float(row["dt"])
        accel_raw = np.array([row["accel_x"], row["accel_y"], row["accel_z"]], dtype=np.float64)
        gyro_raw = np.array([row["gyro_x"], row["gyro_y"], row["gyro_z"]], dtype=np.float64)
        pressure = float(row["pressure"])
        mag_raw = np.array([row["mag_x"], row["mag_y"], row["mag_z"]], dtype=np.float64)

        # First iteration: set initial pressure and orientation
        if initial_pressure is None:
            initial_pressure = pressure
        if initial_quat is None:
            initial_quat, initial_mag = calc_init_orientation(accel_raw, mag_raw)
            eskf.x_nom[6:10] = q.as_float_array(initial_quat)

        # Accumulate during init
        if in_init:
            accel_accum += accel_raw
            gyro_accum += gyro_raw
            n_samples += 1

        # Subtract biases
        accel_cal = accel_raw - accel_bias
        gyro_cal = gyro_raw - gyro_bias
        u = np.array([accel_cal[0], accel_cal[1], accel_cal[2],
                       gyro_cal[0], gyro_cal[1], gyro_cal[2]], dtype=np.float64)

        # Set function handles for init vs running
        if in_init:
            eskf.nominal_predict_func = nominal_predict_init
            eskf.error_jacobian_func = error_state_jacobian_init
            eskf.process_noise_func = lambda x, u_arg, dt_arg: process_noise_matrix(
                x, u_arg, dt_arg, ESKFProcessCovariance.INIT.array)
            eskf.R = np.diag(ESKFMeasurementNoise.INIT.matrix).astype(np.float64)
        else:
            eskf.nominal_predict_func = nominal_predict
            eskf.error_jacobian_func = error_state_jacobian
            eskf.process_noise_func = lambda x, u_arg, dt_arg: process_noise_matrix(
                x, u_arg, dt_arg, ESKFProcessCovariance.RUNNING.array)
            eskf.R = np.diag(ESKFMeasurementNoise.RUNNING.matrix).astype(np.float64)

        # Predict
        if dt > 1e-12:
            eskf.predict(dt, u)

        # Build measurement (normalise mag)
        mag_norm = np.linalg.norm(mag_raw)
        if mag_norm > 1e-6:
            mag_normed = mag_raw / mag_norm
        else:
            mag_normed = np.zeros(3)
        z = np.array([pressure, mag_normed[0], mag_normed[1], mag_normed[2]], dtype=np.float64)

        # Update
        eskf.update(z, initial_pressure, initial_mag)

        elapsed += dt
        state_num = 0 if in_init else 1

        # Check transition
        if in_init and elapsed >= INIT_DURATION:
            # Compute biases
            if n_samples > 0:
                mean_acc = accel_accum / n_samples
                mean_gyr = gyro_accum / n_samples
                quat_obj = q.from_float_array(eskf.x_nom[6:10]).normalized()
                from UKF.eskf_functions import quat_to_rotation_matrix
                R_v2w = quat_to_rotation_matrix(quat_obj)
                gravity_vehicle = R_v2w.T @ np.array([0, 0, GRAVITY])
                gravity_sensor = R_IMU_TO_VEHICLE.T @ (gravity_vehicle / GRAVITY)
                accel_bias = mean_acc - gravity_sensor
                gyro_bias = mean_gyr
            in_init = False
            state_num = 1
            print(f"Python ESKF: init -> running at t={elapsed:.4f}s, "
                  f"accel_bias={accel_bias}, gyro_bias={gyro_bias}")

        results.append([
            idx, dt, state_num,
            eskf.x_nom[0], eskf.x_nom[1], eskf.x_nom[2],
            eskf.x_nom[3], eskf.x_nom[4], eskf.x_nom[5],
            eskf.x_nom[6], eskf.x_nom[7], eskf.x_nom[8], eskf.x_nom[9],
            eskf.mahalanobis_dist,
        ])

    cols = ["step", "dt", "state", "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z", "qw", "qx", "qy", "qz", "mahal"]
    return pd.DataFrame(results, columns=cols)


def compile_c():
    """Compile the C test harness."""
    inc = str(C_DIR / "ukf_data_processing")
    srcs = [str(s) for s in C_SOURCES]
    cmd = ["gcc", "-O2", f"-I{inc}", *srcs, "-lm", "-o", str(C_EXE)]
    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("COMPILE FAILED:")
        print(result.stderr)
        sys.exit(1)
    print("Compilation successful.")


def run_c(input_csv: Path, output_csv: Path):
    """Run the compiled C test binary."""
    cmd = [str(C_EXE), str(input_csv), str(output_csv)]
    print(f"\nRunning C: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        print("C test FAILED with code", result.returncode)
        sys.exit(1)


def compare_results(py_df: pd.DataFrame, c_df: pd.DataFrame):
    """Compare Python vs C ESKF output."""
    n = min(len(py_df), len(c_df))
    print(f"\n{'='*60}")
    print(f"Comparing {n} timesteps")
    print(f"{'='*60}")

    # Align indices
    py = py_df.iloc[:n].reset_index(drop=True)
    c = c_df.iloc[:n].reset_index(drop=True)

    fields = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
              "qw", "qx", "qy", "qz"]

    print(f"\n{'Field':<10} {'Max Abs Err':>14} {'RMS Err':>14} {'Py Final':>14} {'C Final':>14}")
    print("-" * 70)

    for f in fields:
        diff = py[f].values.astype(np.float64) - c[f].values.astype(np.float64)
        max_err = np.max(np.abs(diff))
        rms_err = np.sqrt(np.mean(diff ** 2))
        py_final = py[f].iloc[-1]
        c_final = c[f].iloc[-1]
        print(f"{f:<10} {max_err:>14.6f} {rms_err:>14.6f} {py_final:>14.4f} {c_final:>14.4f}")

    # Summary
    py_max_alt = py["pos_z"].max()
    c_max_alt = c["pos_z"].max()
    py_max_vel = np.sqrt(py["vel_x"]**2 + py["vel_y"]**2 + py["vel_z"]**2).max()
    c_max_vel = np.sqrt(c["vel_x"]**2 + c["vel_y"]**2 + c["vel_z"]**2).max()

    print(f"\n{'Metric':<20} {'Python':>12} {'C':>12} {'Diff':>12}")
    print("-" * 58)
    print(f"{'Max altitude (m)':<20} {py_max_alt:>12.2f} {c_max_alt:>12.2f} {abs(py_max_alt-c_max_alt):>12.4f}")
    print(f"{'Max speed (m/s)':<20} {py_max_vel:>12.2f} {c_max_vel:>12.2f} {abs(py_max_vel-c_max_vel):>12.4f}")

    # Warn if large discrepancies
    pos_err = np.sqrt(
        (py["pos_x"] - c["pos_x"])**2 +
        (py["pos_y"] - c["pos_y"])**2 +
        (py["pos_z"] - c["pos_z"])**2
    ).max()
    if pos_err < 1.0:
        print(f"\n✔ Position error < 1m — GOOD (max 3D pos err = {pos_err:.4f} m)")
    elif pos_err < 10.0:
        print(f"\n⚠ Position error moderate (max 3D pos err = {pos_err:.2f} m) — expected due to float32 vs float64")
    else:
        print(f"\n✘ Position error large (max 3D pos err = {pos_err:.2f} m) — investigate!")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("Pre-processing sensor data...")
    input_df = preprocess_data(LAUNCH_FOLDER, MIN_T, MAX_T)
    input_csv = C_DIR / "test_input.csv"
    input_df.to_csv(input_csv, index=False, float_format="%.12f")
    print(f"Wrote {len(input_df)} rows to {input_csv}")

    print("\n--- Running Python ESKF ---")
    py_results = run_python_eskf(input_df)
    py_csv = C_DIR / "test_output_python.csv"
    py_results.to_csv(py_csv, index=False, float_format="%.12f")
    py_max_alt = py_results["pos_z"].max()
    py_max_vel = np.sqrt(py_results["vel_x"]**2 + py_results["vel_y"]**2 +
                         py_results["vel_z"]**2).max()
    print(f"Python ESKF: max alt = {py_max_alt:.2f} m, max speed = {py_max_vel:.2f} m/s")

    print("\n--- Compiling C ESKF ---")
    compile_c()

    c_output_csv = C_DIR / "test_output.csv"
    run_c(input_csv, c_output_csv)

    print("\n--- Loading C results ---")
    c_results = pd.read_csv(c_output_csv)

    compare_results(py_results, c_results)


if __name__ == "__main__":
    main()
