"""Module for the finite state machine that represents which state of flight the rocket is in."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt

from UKF.constants import (
    STATE_DIM,
    GROUND_ALTITUDE_METERS,
    LANDED_ACCELERATION_GS,
    MAX_ALTITUDE_THRESHOLD,
    MAX_VELOCITY_THRESHOLD,
    TAKEOFF_ACCELERATION_GS,
    StateProcessCovariance,
    StateMeasurementNoise,
)

if TYPE_CHECKING:
    from UKF.context import Context
from UKF.ukf_functions import base_state_transition



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

    __slots__ = ("context")

    def __init__(self, context: "Context"):
        """
        :param context: The Airbrakes Context managing the state machine.
        """
        self.context = context
        self.context.ukf.F = self.state_transition_function
        self.context.ukf.Q = self.process_covariance_function
        

        # standby state init will add to this, but first state change timestamp is removed before plotting
        self.context.set_state_time() 

    @property
    @abstractmethod
    def qvar(self) -> np.float64:
        """Process noise variance to be defined in each state subclass"""

    @property
    @abstractmethod
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        """Measurement noise covariance diagonals defined in each state subclass"""

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
    def state_transition_function(self, sigma_points, dt, *F_args):
        """
        State transition function for Unscented Kalman Filter
        """
    
    def process_covariance_function(self, dt):
        """
        Process noise covariance matrix
        """
        q_covariance_matrix = np.zeros([STATE_DIM, STATE_DIM])
        q_covariance_matrix[-1][-1] = self.qvar
        return q_covariance_matrix
        



class StandbyState(State):
    """
    When the rocket is on the launch rail on the ground.
    """

    __slots__ = ()

    @property
    def qvar(self) -> np.float64:
        return StateProcessCovariance.STANDBY.value
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        return StateMeasurementNoise.STANDBY.matrix


    def update(self):
        """
        Checks if the rocket has launched, based on our velocity.
        """

        # If the velocity of the rocket is above a threshold, the rocket has launched.
        if self.context.measurement[2] < -TAKEOFF_ACCELERATION_GS:
            self.next_state()
            return

    def next_state(self):
        print("standby -> motor burn")
        self.context._flight_state = MotorBurnState(self.context)

    def state_transition_function(self, sigma_points, dt, *F_args):
        return base_state_transition(sigma_points, dt, False, F_args)



class MotorBurnState(State):
    """
    When the motor is burning and the rocket is accelerating.
    """

    __slots__ = ()

    @property
    def qvar(self) -> np.float64:
        return StateProcessCovariance.MOTOR_BURN.value
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        return StateMeasurementNoise.MOTOR_BURN.matrix


    def __init__(self, context: "Context"):
        super().__init__(context)
        self.context.ukf.P = np.add(self.context.ukf.P, 0.01)

    def update(self):
        """Checks to see if the velocity has decreased lower than the maximum velocity, indicating
        the motor has burned out."""


        # If our current velocity is less than our max velocity, that means we have stopped
        # accelerating. This is the same thing as checking if our accel sign has flipped
        # We make sure that it is not just a temporary fluctuation by checking if the velocity is a
        # bit less than the max velocity
        if (self.context.ukf.X[1] < self.context._max_velocity * MAX_VELOCITY_THRESHOLD) and (
            self.context._max_velocity > 20
        ):
            self.next_state()
            return

    def next_state(self):
        print("motor burn -> coast")
        self.context._flight_state = CoastState(self.context)

    def state_transition_function(self, sigma_points, dt, *F_args):
        return base_state_transition(sigma_points, dt, True, F_args)



class CoastState(State):
    """
    When the motor has burned out and the rocket is coasting to apogee.
    """

    __slots__ = ()

    @property
    def qvar(self) -> np.float64:
        return StateProcessCovariance.COAST.value
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        return StateMeasurementNoise.COAST.matrix
    
    def __init__(self, context: "Context"):
        super().__init__(context)
        self.context.ukf.P = np.add(self.context.ukf.P, 0.01)

    def update(self):
        """Checks to see if the rocket has reached apogee, indicating the start of free fall."""

        # If our velocity is less than 0 and our altitude is less than 96% of our max altitude, we
        # are in free fall.
        if (
            self.context.ukf.X[1] <= 0
            and self.context.ukf.X[0] <= self.context._max_altitude * MAX_ALTITUDE_THRESHOLD
        ):
            self.next_state()
            return

    def next_state(self):
        print("coast -> freefall")
        self.context._flight_state = FreeFallState(self.context)

    def state_transition_function(self, sigma_points, dt, *F_args):
        return base_state_transition(sigma_points, dt, True, F_args)



class FreeFallState(State):
    """
    When the rocket is falling back to the ground after apogee.
    """

    @property
    def qvar(self) -> np.float64:
        return StateProcessCovariance.FREEFALL.value
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        return StateMeasurementNoise.FREEFALL.matrix


    __slots__ = ()

    def __init__(self, context: "Context"):
        super().__init__(context)

    def update(self):
        """Check if the rocket has landed, based on our altitude and a spike in acceleration."""


        # If our altitude is around 0, and we have an acceleration spike, we have landed
        if (
            self.context.ukf.X[0] <= GROUND_ALTITUDE_METERS
            and -self.context.measurement[2] >= LANDED_ACCELERATION_GS
        ):
            self.next_state()


    def next_state(self):
        print("freefall -> landed")
        self.context._flight_state = LandedState(self.context)

    def state_transition_function(self, sigma_points, dt, *F_args):
        return base_state_transition(sigma_points, dt, False, F_args)



class LandedState(State):
    """
    When the rocket has landed.
    """

    @property
    def qvar(self) -> np.float64:
        return StateProcessCovariance.LANDED.value
    
    @property
    def measurement_noise_diagonals(self) -> npt.NDArray[np.float64]:
        return StateMeasurementNoise.LANDED.matrix


    __slots__ = ()

    def __init__(self, context: "Context"):
        super().__init__(context)

    def update(self):
        pass

    def next_state(self):
        # Explicitly do nothing, there is no next state
        pass

    def state_transition_function(self, sigma_points, dt, *F_args):
        return base_state_transition(sigma_points, dt, False, F_args)

