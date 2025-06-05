"""
Parameters Configuration Module for Distributed Formation Control

This module defines and manages all simulation parameters for the distributed formation
control system. It provides a centralized configuration that can be easily modified
to adjust simulation behavior.

Configuration Parameters:
- Formation Parameters:
    - FORMATION_POINTS_NUM: Number of nodes in the formation
    - FORMATION_RADIUS: Radius of the circular formation
    - FORMATION_CENTER_X/Y: Center coordinates of the formation
    - MISSION_ALTITUDE: Operating altitude of the formation

- Communication Parameters:
    - COMMUNICATION_TRANSMISSION_RANGE: Maximum communication range
    - COMMUNICATION_DELAY: Communication delay between nodes
    - COMMUNICATION_FAILURE_RATE: Probability of communication failure

- Simulation Parameters:
    - SIMULATION_DURATION: Total simulation time
    - SIMULATION_DEBUG: Debug mode flag
    - SIMULATION_REAL_TIME: Real-time simulation flag
    - UPDATE_RATE: Rate of simulation updates

- State Vectors:
    - SHAPE_VECTOR: Current formation shape
    - STATUS_VECTOR: Node status information
    - CORRECT_VECTOR: Correctness indicators

Usage:
    from parameters import Parameters
    params = Parameters()
    # Modify parameters as needed
"""

class Parameters:
    """
    Configuration class for simulation parameters.
    
    This class holds all the parameters needed for the simulation and provides
    default values that can be overridden as needed.
    """

    def __init__(self):

        # --------------------------------------------------------------
        # PARAMETERS TO CONFIGURE THE SIMULATION
        # --------------------------------------------------------------

        # Simulation parameters
        self.SIMULATION_DURATION = 301 # Simulation duration in seconds
        self.SIMULATION_DEBUG = False # Simulation debug mode
        self.SIMULATION_REAL_TIME = False # Simulation real time mode
        self.UPDATE_RATE = 0.05 # Simulation update rate in seconds

        # Mobility parameters
        self.SPEED_LEADER = 10 # Speed of the leader in meters per second
        self.SPEED_FOLLOWER = 20 # Speed of the followers in meters per second

        # Formation parameters
        self.FORMATION_RADIUS = 10 # Formation radius in meters
        self.FORMATION_CENTER_X = 0 # Formation center x coordinate initial value in meters
        self.FORMATION_CENTER_Y = 0 # Formation center y coordinate initial value in meters
        self.FORMATION_ALTERNATE = True # Formation alternate mode (True: alternate formation between circle and line, False: do not alternate formation, only circle)

        # Mission parameters
        self.MISSION_RADIUS = 20 # Mission circle radius in meters
        self.MISSION_POINTS_NUM = 36 # Number of way points in the mission for each half of the mission
        self.MISSION_ALTITUDE = 0 # Mission altitude in meters
        self.MISSION_STEP_TIME = 0.05 # Mission step time in seconds
        self.MISSION_TIMER_STD_DEV = 0.0 # Mission timer standard deviation in percentage of the designated time
        self.MISSION_FIX = True # Mission fix mode (True: fix mission, False: do not fix mission)

        # Visualization parameters
        self.VISUALIZATION_OPEN_BROWSER = False # Visualization open browser
        self.VISUALIZATION_SLOW_DOWN = False # Visualization slow down factor
        self.TERMINAL_MESSAGES = False # Terminal messages control (True: show terminal messages, False: do not show terminal messages)

        # Plot parameters
        self.PLOT_RESULTS = False # Plot results flag (True: plot results, False: do not plot results)

        # Recharge battery follower parameters
        self.RECHARGE_PROBABILITY = 0.05 # Probability of the battery recharge event occurring each second.
        self.RECHARGE_TIME = 5 # Recharge battery follower time in seconds
        self.RECHARGE_POSITION = (0,0,0) # Recharge battery follower position

        # Heartbeat parameters ( HEARTBEAT_FACTOR * election_timeout = heartbeat_timeout)
        self.HEARTBEAT_FACTOR = 0.1 # Heartbeat factor

        # Communication parameters that can be changed according to the simulation run
        self.COMMUNICATION_TRANSMISSION_RANGE = 100 # Communication transmission range in meters
        self.COMMUNICATION_DELAY = 0.0 # Communication delay in seconds
        self.COMMUNICATION_FAILURE_RATE = 0.0 # Communication failure rate in percentage per second
        self.FORMATION_POINTS_NUM = 8 # Number of points in the formation

        # File path to save the metrics
        # The file path is defined in the format "PxxxRxxxDxxxFxxx.csv"
        # where xxx is the value of the parameters used in the simulation
        # Pxxx: Number of points in the formation
        # Rxxx: Communication transmission range in meters
        # Dxxx: Communication delay in microseconds
        # Fxxx: Communication failure rate in percentage per second
        self.FILE_PATH = "P000R000D000F000.csv"

        # --------------------------------------------------------------
        # GLOBAL VARIABLES TO SUPPORT THE CALCULATION OF METRICS
        # --------------------------------------------------------------

        # Variables to calculate the presence error
        # presence_error indicates the discrepancy between expected nodes in a formation
        # and the nodes actually identified as present in that formation.
        # The presence error is calculated as the average of the combined errors of the formation.
        # And is limited between 0% and 100%., where 0% indicates no error and 100% indicates maximum error.
        self.STATUS_VECTOR = [[1]*self.FORMATION_POINTS_NUM]*self.FORMATION_POINTS_NUM # Status vector to store the position of the formation in each node
        self.CORRECT_VECTOR = [1]*self.FORMATION_POINTS_NUM # Standard vector to store the position of the correct formation

        # Variables to calculate the shape error of the formation
        # shape_error indicates the discrepancy between the expected formation shape
        # and the actual formation shape.
        # The shape error is calculated as the average of the combined errors of the formation.
        # And is limited between 0% and 100%., where 0% indicates no error and 100% indicates maximum error.
        self.FORMATION_TYPE = "circle" # Formation type (circle or line)
        self.SHAPE_VECTOR = [(0,0,0)]*self.FORMATION_POINTS_NUM # Shape vector to store the position of each node in the formation

        # Variable to calculate the percentage of time with no leader elected by RAFT
        self.LEADER_OFF = 1 # Flag to indicate if a leader isn´t elected by RAFT



