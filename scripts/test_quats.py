import quaternion as q
import numpy as np

# 1.43789645e-02 4.90348274e-03 9.81337592e+00

a = np.array([-9.8056965, -0.28149602, -0.2676829], dtype=np.float32)
quat_float = np.array([-0.153586767399188, -0.693747247404336, -0.177729150723393, 0.680836405153188], dtype=np.float32)
quat = q.from_float_array(quat_float)
a_q = q.from_float_array([0, a[0], a[1], a[2]])
vec = quat * a_q * quat.conjugate()
print(vec)