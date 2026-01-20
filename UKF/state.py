"""Module for the finite state machine that represents which state of flight the rocket is in."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt


from UKF.constants import (
    GROUND_ALTITUDE_METERS,
    LANDED_ACCELERATION_GS,
    MAX_ALTITUDE_THRESHOLD,
    MAX_VELOCITY_THRESHOLD,
    StateProcessCovariance,
    StateMeasurementNoise,
    StateControlInput,
)

if TYPE_CHECKING:
    from UKF.context import Context
from UKF.ukf_functions import state_transition_function, measurement_function



class State(ABC):
    """
    Abstract Base class for the states of the rocket. Each state will have an update
    method that will be called every loop iteration and a next_state method that will be called
    when the state is over.

    1. Standby - when the rocket is on the rail on the ground
    2. Motor Burn - when the motor is burning and the rocket is accelerating
    3. Coast - after the motor has burned out and the rocket is coasting
    4. Free Fall - when the rocket is falling back to the ground after apogee
    5. Landed - when the rocket lands on the ground.
    """

    __slots__ = (
        "context",
        "transient_time",
    )

    def __init__(self, context: "Context"):
        """
        :param context: The UKF Context managing the state machine.
        """
        self.context = context
        self.context.ukf.F = self.state_transition_function
        self.context.ukf.Q = self.process_covariance_function
        self.context.ukf.H = self.measurement_function
        
        # standby state init will add to this, but first state change timestamp is removed before plotting
        self.context.set_state_time() 

        self.transient_time = 0.5


    @property
    @abstractmethod
    def qvar(self) -> np.float64:
        """Process noise variance to be defined in each state subclass"""

    @property
    @abstractmethod
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        """Measurement noise covariance diagonals defined in each state subclass"""

    @property
    @abstractmethod
    def control_input(self) -> npt.NDArray[np.float64]:
        """Control input vector for the state transition function"""

    @abstractmethod
    def update(self):
        """
        Called every loop iteration. Decides when to move to next state
        """

    @abstractmethod
    def next_state(self):
        """
        We never expect/want to go back a state e.g. We're never going to go
        from Flight to Motor Burn, so this method just goes to the next state.
        """

    @abstractmethod
    def state_transition_function(self, sigma_points, dt, u):
        """
        State transition function for Unscented Kalman Filter
        """

    @abstractmethod
    def measurement_function(self, sigmas, init_pressure, init_mag):
        """
        Measurement function for Unscented Kalman Filter
        """
    
    def process_covariance_function(self, dt):
        """
        Process noise covariance matrix
        """
        qvar = self.qvar * dt
        return np.diag(qvar)


class StandbyState(State):
    """
    When the rocket is on the launch rail on the ground.
    """

    __slots__ = ()

    @property
    def qvar(self) -> np.float64:
        return StateProcessCovariance.STANDBY.array
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        return StateMeasurementNoise.STANDBY.matrix
    
    @property
    def control_input(self) -> npt.NDArray[np.float64]:
        return StateControlInput.STANDBY.array


    def update(self):
        """
        Checks if the rocket has launched, based on our velocity.
        """

        # If the velocity of the rocket is above a threshold, the rocket has launched.
        if np.abs(self.context.ukf.mahalanobis_dist) > 30:
            self.next_state()
            return

    def next_state(self):
        print("standby -> motor burn")
        self.context._flight_state = MotorBurnState(self.context)

    def state_transition_function(self, sigma_points, dt, u):
        return state_transition_function(sigma_points, dt, u)
    
    def measurement_function(self, sigmas, init_pressure, init_mag):
        return measurement_function(sigmas, init_pressure, init_mag)    




class MotorBurnState(State):
    """
    When the motor is burning and the rocket is accelerating.
    """

    __slots__ = (
    )

    @property
    def qvar(self) -> np.float64:
        noise = StateProcessCovariance.MOTOR_BURN.array
        return noise
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        noise = StateMeasurementNoise.MOTOR_BURN.matrix
        if np.abs(self.context.data_processor.measurements[1]) > 19:
            noise[1] *= 1e3
        if np.abs(self.context.data_processor.measurements[2]) > 19:
            noise[2] *= 1e3
        if np.abs(self.context.data_processor.measurements[3]) > 19:
            noise[3] *= 1e3

        # an attempt to filter transonic effects
        noise[0] *= max(self.context.ukf.X[5], 1)
        return noise

    @property
    def control_input(self) -> npt.NDArray[np.float64]:
        return StateControlInput.MOTOR_BURN.array


    def __init__(self, context: "Context"):
        super().__init__(context)

    def update(self):
        """Checks to see if the velocity has decreased lower than the maximum velocity, indicating
        the motor has burned out."""


        if self.context.ukf.X[2] > 15 and self.context.ukf.X[8] < -0.1:
            self.next_state()
            return

    def next_state(self):
        print("motor burn -> coast")
        self.context._flight_state = CoastState(self.context)

    def state_transition_function(self, sigma_points, dt, u):
        return state_transition_function(sigma_points, dt, u)
    def measurement_function(self, sigmas, init_pressure, init_mag):
        return measurement_function(sigmas, init_pressure, init_mag)    


class CoastState(State):
    """
    When the motor has burned out and the rocket is coasting to apogee.
    """

    __slots__ = (
        "pressure_uncertainty"
    )

    @property
    def qvar(self) -> np.float64:
        noise = StateProcessCovariance.COAST.array
        return noise
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        noise = StateMeasurementNoise.COAST.matrix
        
        noise[0] *= max(self.context.ukf.X[5], 1)
        return noise

    @property
    def control_input(self) -> npt.NDArray[np.float64]:
        return StateControlInput.COAST.array

    def __init__(self, context: "Context"):
        super().__init__(context)

    def update(self):
        """Checks to see if the rocket has reached apogee, indicating the start of free fall."""

        if (self.context.ukf.X[5] <= 0):
            self.next_state()
            return

    def next_state(self):
        print("coast -> freefall")
        self.context._flight_state = FreeFallState(self.context)

    def state_transition_function(self, sigma_points, dt, u):
        return state_transition_function(sigma_points, dt, u)
    def measurement_function(self, sigmas, init_pressure, init_mag):
        return measurement_function(sigmas, init_pressure, init_mag)    


class FreeFallState(State):
    """
    When the rocket is falling back to the ground after apogee.
    """

    @property
    def qvar(self) -> np.float64:
        process_noise = StateProcessCovariance.FREEFALL.array
        # start lowering certainty in acceleration and gyro predictions the closer the rocket
        # gets to landing, because hitting the ground throws off expected values
        if self.context.ukf.X[2] < 100:
            process_noise[6:12] *= (101 - self.context.ukf.X[2])**1.5
        return process_noise
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        return StateMeasurementNoise.FREEFALL.matrix

    @property
    def control_input(self) -> npt.NDArray[np.float64]:
        return StateControlInput.FREEFALL.array

    __slots__ = ()

    def __init__(self, context: "Context"):
        super().__init__(context)

    def update(self):
        """Check if the rocket has landed, based on our altitude and a spike in acceleration."""


        # If our altitude is around 0, and we have an acceleration spike, we have landed
        if (
            self.context.ukf.X[2] <= GROUND_ALTITUDE_METERS
            and np.abs(self.context.data_processor.measurements[3]) >= LANDED_ACCELERATION_GS
        ):
            self.next_state()


    def next_state(self):
        print("freefall -> landed")
        self.context._flight_state = LandedState(self.context)

    def state_transition_function(self, sigma_points, dt, u):
        return state_transition_function(sigma_points, dt, u)
    def measurement_function(self, sigmas, init_pressure, init_mag):
        return measurement_function(sigmas, init_pressure, init_mag)    


class LandedState(State):
    """
    When the rocket has landed.
    """

    @property
    def qvar(self) -> np.float64:
        return StateProcessCovariance.LANDED.array
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        return StateMeasurementNoise.LANDED.matrix

    @property
    def control_input(self) -> npt.NDArray[np.float64]:
        return StateControlInput.LANDED.array

    __slots__ = ()

    def __init__(self, context: "Context"):
        super().__init__(context)

    def update(self):
        pass

    def next_state(self):
        # Explicitly do nothing, there is no next state
        pass

    def state_transition_function(self, sigma_points, dt, u):
        return state_transition_function(sigma_points, dt, u)
    def measurement_function(self, sigmas, init_pressure, init_mag):
        return measurement_function(sigmas, init_pressure, init_mag)    
