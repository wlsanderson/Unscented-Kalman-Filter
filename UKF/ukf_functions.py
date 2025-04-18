from UKF.constants import GRAVITY, DRAG_COEFFICIENT, ROCKET_MASS, REFERENCE_AREA, AIR_DENSITY
import numpy as np
import numpy.typing as npt
import quaternion as q

def measurement_function(sigmas, init_alt):
    alt = sigmas[0] + init_alt
    acc = np.array([sigmas[2], sigmas[3], sigmas[4]]) / GRAVITY
    return np.array([alt, acc[0], acc[1], acc[2], sigmas[5], sigmas[6], sigmas[7]])

def state_transition_function(sigmas, dt, X, drag_option: bool = False) -> npt.NDArray:
    n = len(sigmas)
    quat = q.from_float_array(sigmas[n-4:n])
    x_quat = q.from_float_array(X[n-4:n])
    delta_theta = sigmas[n-7:n-4] * dt
    q_next = quat * q.from_rotation_vector(delta_theta)
    q_next = q_next.normalized()
    next_accs = sigmas[2:5]
    rot_acc = x_quat * q.from_vector_part(next_accs) * x_quat.conjugate()
    
    linear_accel = (-rot_acc.z - GRAVITY)
    # print(f"rot_acc.z: {rot_acc.z}")
    # print(f"linear_accel: {linear_accel}")
    next_vel = sigmas[1] + linear_accel * dt
    # if drag_option:
    #     rot_acc.z = rot_acc.z + dt * calc_drag(next_vel) / ROCKET_MASS
    #     next_vel = sigmas[1] + linear_accel * dt
    next_alt = sigmas[0] + (next_vel * dt)

    

    #delta_theta[0] = delta_theta[0]*np.cos(-delta_theta[2])-delta_theta[1]*np.sin(-delta_theta[2])
    #delta_theta[1] = delta_theta[0]*np.sin(-delta_theta[2])+delta_theta[1]*np.cos(-delta_theta[2])

    
    
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
