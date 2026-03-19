import numpy as np
import quaternion as q
from UKF.eskf_functions import (
    measurement_function,
    measurement_jacobian,
    R_MAG_TO_BOARD_V2,
)

# Simple sanity checks for magnetometer measurement model
p0 = 101325.0  # arbitrary reference pressure
mag_world = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def state_from_quat(quat: q.quaternion) -> np.ndarray:
    x = np.zeros(6, dtype=np.float32)
    x[2:6] = q.as_float_array(quat)
    return x


def run_cases():
    cases = {
        "identity": q.quaternion(1, 0, 0, 0),
        "+90yaw": q.from_rotation_vector(np.array([0.0, 0.0, np.pi / 2], dtype=np.float32)),
        "-90yaw": q.from_rotation_vector(np.array([0.0, 0.0, -np.pi / 2], dtype=np.float32)),
        "+180yaw": q.from_rotation_vector(np.array([0.0, 0.0, np.pi], dtype=np.float32)),
    }

    print("R_MAG_TO_BOARD_V2:\n", R_MAG_TO_BOARD_V2)
    for name, quat in cases.items():
        quat_n = quat.normalized()
        x = state_from_quat(quat_n)
        z = measurement_function(x, p0, mag_world, R_mag=R_MAG_TO_BOARD_V2)
        H = measurement_jacobian(x, p0, mag_world, R_mag=R_MAG_TO_BOARD_V2)
        print(f"\nCase: {name}")
        print("quat:", q.as_float_array(quat_n))
        print("mag_pred:", z[1:])
        print("H mag block:\n", H[1:, 2:])


def finite_diff_identity():
    base_quat = q.quaternion(1, 0, 0, 0)
    x0 = state_from_quat(base_quat)
    z0 = measurement_function(x0, p0, mag_world, R_mag=R_MAG_TO_BOARD_V2)[1:]
    H = measurement_jacobian(x0, p0, mag_world, R_mag=R_MAG_TO_BOARD_V2)[1:, 2:]

    print("\nFinite diff around identity:")
    for axis in range(3):
        dtheta = np.zeros(3, dtype=np.float32)
        dtheta[axis] = 1e-4
        dq = q.from_rotation_vector(dtheta)
        x1 = state_from_quat((base_quat * dq).normalized())
        z1 = measurement_function(x1, p0, mag_world, R_mag=R_MAG_TO_BOARD_V2)[1:]
        fd = (z1 - z0) / 1e-4
        print(f"axis {axis}: fd {fd}, Hcol {H[:, axis]}")


def main():
    run_cases()
    finite_diff_identity()


if __name__ == "__main__":
    main()
