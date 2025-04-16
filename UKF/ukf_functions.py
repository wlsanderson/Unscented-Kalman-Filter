from UKF.constants import GRAVITY, DRAG_COEFFICIENT, ROCKET_MASS, REFERENCE_AREA, AIR_DENSITY
import numpy as np
import numpy.typing as npt
import quaternion as q

def measurement_function(sigmas, init_alt, adjust_gyro_w_acc: bool = False):
    n = len(sigmas)
    quat_sigmas = sigmas[n-4:n]

    quat_sigmas /= np.linalg.norm(quat_sigmas)
    gyro_sigmas = sigmas[n-7:n-4]
    global_acc = np.array([sigmas[2], sigmas[3], sigmas[4]])
    global_acc = q.from_vector_part(global_acc / GRAVITY)
    quat = q.from_float_array(quat_sigmas)
    acc = quat.conjugate() * global_acc * quat
    alt = sigmas[0] + init_alt
    gyro = gyro_sigmas

    return np.array([alt, acc.x, acc.y, acc.z, gyro[0], gyro[1], gyro[2]])

def state_transition_function(sigmas, dt, drag_option: bool = False) -> npt.NDArray:
    n = len(sigmas)
    next_accs = sigmas[2:5]
    next_vel = sigmas[1] + (-next_accs[2] - GRAVITY) * dt
    if drag_option:
        next_accs[2] = next_accs[2] + dt * calc_drag(next_vel) / ROCKET_MASS
        next_vel = sigmas[1] + (-next_accs[2] - GRAVITY) * dt
    next_alt = sigmas[0] + (next_vel * dt) + 0.5 * (next_accs[2] * dt**2)

    delta_theta = sigmas[n-7:n-4] * dt

    #delta_theta[0] = delta_theta[0]*np.cos(-delta_theta[2])-delta_theta[1]*np.sin(-delta_theta[2])
    #delta_theta[1] = delta_theta[0]*np.sin(-delta_theta[2])+delta_theta[1]*np.cos(-delta_theta[2])

    quat = q.from_float_array(sigmas[n-4:n])
    q_next = quat * q.from_rotation_vector(delta_theta)
    q_next = q_next.normalized()
    if np.any(np.isnan(q_next.components)) or abs(q_next.norm() - 1.0) > 1e-2:
        print("Quaternion error detected")

    return np.array([
        next_alt,
        next_vel, 
        next_accs[0],
        next_accs[1],
        next_accs[2],
        delta_theta[0]/dt,
        delta_theta[1]/dt,
        delta_theta[2]/dt,
        q_next.w,
        q_next.x,
        q_next.y,
        q_next.z
        ])

def calc_drag(velocity):
    return 0.5 * DRAG_COEFFICIENT * REFERENCE_AREA * AIR_DENSITY * velocity**2
