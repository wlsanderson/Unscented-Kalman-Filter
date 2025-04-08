import numpy as np
import numpy.typing as npt

def quat_multiply(q1, q0):
    q1 = np.atleast_2d(q1)
    q0 = np.atleast_2d(q0)
    if q1.shape != q0.shape and q0.shape[0] > 1:
        raise ValueError("Input lists of quaternions must be same size arrays")
    if q1.shape != q0.shape and q0.shape[0] == 1:
        q0 = np.full([q1.shape[0], 4], q0[0])
    w0, x0, y0, z0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    result = np.stack([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    ], axis=1)
    return result[0] if result.shape[0] == 1 else result

def quat_inv(quat):
    q = np.atleast_2d(quat)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    result = np.stack([w, -x, -y, -z], axis=1)
    return result[0] if result.shape[0] == 1 else result

def rotvec2quat(rotvec, dt=1):
    rotvec = np.atleast_2d(rotvec)
    angles = np.linalg.norm(rotvec, axis=1) * dt

    result = np.empty([rotvec.shape[0], 4], dtype=np.float64)
    small_angles = angles < 1e-20

    result[small_angles] = np.array([1.0, 0.0, 0.0, 0.0])

    valid = ~small_angles
    axis = np.zeros_like(rotvec)
    axis[valid] = rotvec[valid] / angles[valid, None]
    w = np.cos(angles[valid] / 2)
    xyz = axis[valid] * np.sin(angles[valid] / 2)[:, None]
    result[valid] = np.concatenate([w[:, None], xyz], axis=1)

    return result[0] if result.shape[0] == 1 else result
    
def quat2rotvec(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    quat = np.atleast_2d(quat)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    angles = 2 * np.arccos(quat[:, 0])
    result = np.zeros([quat.shape[0], 3], dtype=np.float64)

    small_angles = angles < 1e-20
    valid = ~small_angles

    axis = np.zeros_like(result)
    axis[valid] = quat[valid, 1:] / np.sin(angles[valid] / 2)[:, None]
    result[valid] = axis[valid] * angles[valid, None]
    return result[0] if quat.shape[0] == 1 else result

def quat_rotate(
        quat: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
    """
    Need to check if this is rotating from global frame -> vehicle frame, or
    vehicle frame -> global frame.
    """
    v = np.concatenate([[0], v])
    return quat_multiply(quat, quat_multiply(v, quat_inv(quat)))[1:]