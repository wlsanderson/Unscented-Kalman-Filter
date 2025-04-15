import numpy as np
import numpy.typing as npt
import quaternion as q

def measurement_function(sigmas):
    return np.array([sigmas[0], sigmas[1], sigmas[2]])

def state_transition_function(sigmas, dt) -> npt.NDArray:
    delta_theta = sigmas[0:3] * dt

    # delta_theta[0] = delta_theta[0]*np.cos(-delta_theta[2])-delta_theta[1]*np.sin(-delta_theta[2])
    # delta_theta[1] = delta_theta[0]*np.sin(-delta_theta[2])+delta_theta[1]*np.cos(-delta_theta[2])

    quat = q.from_float_array(sigmas[3:7])
    q_next = quat * q.from_rotation_vector(delta_theta)
    q_next = q_next.normalized()
    if np.any(np.isnan(q_next.components)) or abs(q_next.norm() - 1.0) > 1e-2:
        print("Quaternion error detected")

    return np.array([
        delta_theta[0]/dt,
        delta_theta[1]/dt,
        delta_theta[2]/dt,
        q_next.w,
        q_next.x,
        q_next.y,
        q_next.z
        ])

