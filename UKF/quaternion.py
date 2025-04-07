import numpy as np

def quat_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def quat_inv(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

def eul2quat(eul, dt=1):
    angle = np.linalg.norm(eul) * dt
    if angle < 1e-20:
        print("divide by zero eul2quat")
        return np.array([1.0, 0, 0, 0])
    else:
        axis = eul / angle
        w = np.cos(angle / 2)
        xyz = axis * np.sin(angle / 2)
        return np.concatenate(([w], xyz))
