import numpy as np
import quaternion as q

a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
a = a / np.linalg.norm(a)

print(q.from_rotation_vector(a))
