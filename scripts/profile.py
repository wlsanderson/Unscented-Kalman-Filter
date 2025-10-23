import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R
import timeit

# Generate input data
N = 10000
gyro_vecs = np.random.randn(N, 3) * 0.01  # small angular velocities
quats_npq = quaternion.as_float_array(quaternion.as_quat_array(np.random.rand(N, 4)))
quats_npq /= np.linalg.norm(quats_npq, axis=1, keepdims=True)

# Convert for scipy
quats_scipy = quats_npq[:, [1, 2, 3, 0]]  # to [x, y, z, w]

# -------------- Function with numpy-quaternion -----------------
def quat_ops_npq():
    qs = quaternion.from_float_array(quats_npq)
    delta_qs = quaternion.from_rotation_vector(gyro_vecs)
    result = delta_qs * qs
    rotvecs = quaternion.as_rotation_vector(result)
    return rotvecs

# -------------- Function with scipy -----------------
def quat_ops_scipy():
    qs = R.from_quat(quats_scipy)
    delta_qs = R.from_rotvec(gyro_vecs)
    result = delta_qs * qs
    rotvecs = result.as_rotvec()
    return rotvecs

# -------------- Timeit Benchmark -----------------
t_npq = timeit.timeit(quat_ops_npq, number=50)
t_scipy = timeit.timeit(quat_ops_scipy, number=50)

print(f"numpy-quaternion: {t_npq:.4f} sec")
print(f"scipy.spatial.transform: {t_scipy:.4f} sec")
