"""
Data Generation Module for Distributed Formation Control Simulation

This module is responsible for generating simulation data for training, validation, and testing
of distributed formation control algorithms. It uses the GradySim simulator framework to run
multiple simulations with different parameter combinations in parallel.

Key Features:
- Supports three types of datasets: training, validation, and test
- Configurable parameters for formation points, communication range, delay, and failure rate
- Parallel execution of simulations using ProcessPoolExecutor
- Progress tracking with tqdm
- Automatic file naming based on simulation parameters

Usage:
    python data_generation.py
    # Then select the dataset type (1: training, 2: validation, 3: test)

Dependencies:
    - gradysim: Simulation framework
    - numpy: Numerical computations
    - tqdm: Progress bar
    - concurrent.futures: Parallel processing

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.0
"""

from parameters import Parameters
import math
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from protocol import BaseStationProtocol, RAFTProtocol
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np

def simulation(formation_points_num, communication_transmission_range, communication_delay, communication_failure_rate):
    
    # Instanciar os parâmetros
    params = Parameters()

    # Configurar os parâmetros
    params.FORMATION_POINTS_NUM = formation_points_num
    params.COMMUNICATION_TRANSMISSION_RANGE = communication_transmission_range
    params.COMMUNICATION_DELAY = communication_delay
    params.COMMUNICATION_FAILURE_RATE = communication_failure_rate

    # Usar numpy arrays para melhor performance
    params.SHAPE_VECTOR = np.zeros((formation_points_num, 3))
    params.STATUS_VECTOR = np.ones((formation_points_num, formation_points_num))
    params.CORRECT_VECTOR = np.ones(formation_points_num)
    
    # Create the path to the file
    params.FILE_PATH = "P{:03d}R{:03d}D{:03d}F{:03d}.csv".format(
        formation_points_num,
        communication_transmission_range,
        int(communication_delay*1e3),
        int(communication_failure_rate*100)
    )

    builder = SimulationBuilder(SimulationConfiguration(
        duration=params.SIMULATION_DURATION,
        debug=params.SIMULATION_DEBUG,
        real_time=params.SIMULATION_REAL_TIME
    ))
    
    # Criar o medium com configurações otimizadas
    medium = CommunicationMedium(
        transmission_range=params.COMMUNICATION_TRANSMISSION_RANGE,
        delay=params.COMMUNICATION_DELAY,
        failure_rate=params.COMMUNICATION_FAILURE_RATE
    )
    builder.add_handler(CommunicationHandler(medium))
    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler(MobilityConfiguration(update_rate=params.UPDATE_RATE)))
    builder._configuration.execution_logging = False # Disable logging

    # Add the base station positioned at the origin (0,0)
    center_x, center_y = 0, 0  # Base Station position
    builder.add_node(create_base_station_protocol(params), (center_x, center_y, 0))

    # Add follower nodes starting at the circle formation
    radius = params.FORMATION_RADIUS  # Formation circle radius
    points_num = params.FORMATION_POINTS_NUM  # This will create a circle with FORMATION_POINTS_NUM points
    center_x, center_y = params.FORMATION_CENTER_X, params.FORMATION_CENTER_Y  # Formation Center
    for i in range(points_num - 1, -1, -1):  # Generating points for the circle formation (relative to the leader)
        angle = (2 * math.pi / points_num) * i
        x = radius * math.cos(angle) + center_x
        y = radius * math.sin(angle) + center_y
        builder.add_node(create_raft_protocol(params), (x, y, params.MISSION_ALTITUDE))

    simulation = builder.build()
    simulation._logger.disabled = True # Able logging
    simulation.start_simulation()

# Custom BaseStationProtocol creation function to pass parameters
def create_base_station_protocol(parameters):
    class CustomBaseStationProtocol(BaseStationProtocol):
        def __init__(self):
            super().__init__(parameters)    
    return CustomBaseStationProtocol

# Custom RAFTProtocol creation function to pass parameters
def create_raft_protocol(parameters):
    class CustomRAFTProtocol(RAFTProtocol):
        def __init__(self):
            super().__init__(parameters)
    return CustomRAFTProtocol

# Wrapper function to unpack parameters for simulation
def run_simulation(params):
    simulation(*params)

def main():

    # Define datasets
    datasets = {
        "training": {
            "formation_points_num": [8, 12, 16, 24, 32, 48, 64, 96, 128],
            "communication_transmission_range": [20, 26, 32, 38, 44, 50, 56, 60],
            "communication_delay": [0.00, 0.02, 0.05, 0.07, 0.09, 0.11, 0.14, 0.16],
            "communication_failure_rate": [0.00, 0.03, 0.06, 0.09, 0.11, 0.14, 0.17, 0.20]
        },
        "validation": {
            "formation_points_num": [10, 20, 40, 80, 124],
            "communication_transmission_range": [19, 29, 39, 49, 59],
            "communication_delay": [0.01, 0.03, 0.06, 0.12, 0.17],
            "communication_failure_rate": [0.01, 0.05, 0.10, 0.15, 0.21]
        },
        "test": {
            "formation_points_num": [14, 28, 56, 90, 120],
            "communication_transmission_range": [21, 31, 41, 51, 61],
            "communication_delay": [0.04, 0.08, 0.10, 0.13, 0.15],
            "communication_failure_rate": [0.02, 0.04, 0.07, 0.12, 0.19]
        }
    }
    
    # Ask user to choose dataset
    print("Choose the dataset to run:")
    print("1. Training")
    print("2. Validation")
    print("3. Test")
    
    # Loop until a valid choice is entered
    while True:
        choice = input("Enter option number (1-3): ")
        if choice in ["1", "2", "3"]:
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")
    
    # Set parameters based on user choice
    if choice == "1":
        dataset_name = "training"
        print("Training Dataset being generated...")
    elif choice == "2":
        dataset_name = "validation"
        print("Validation Dataset being generated...")
    elif choice == "3":
        dataset_name = "test"
        print("Test Dataset being generated...")
    
    # Set the parameter values based on the selected dataset
    formation_points_num_values = datasets[dataset_name]["formation_points_num"]
    communication_transmission_range_values = datasets[dataset_name]["communication_transmission_range"]
    communication_delay_values = datasets[dataset_name]["communication_delay"]
    communication_failure_rate_values = datasets[dataset_name]["communication_failure_rate"]

    # Create all parameter combinations using itertools.product
    all_combinations = list(product(formation_points_num_values, 
                                    communication_transmission_range_values, 
                                    communication_delay_values, 
                                    communication_failure_rate_values))

    # Count the number of CPUs to use
    num_workers = multiprocessing.cpu_count()

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Progress bar to monitor simulation progress
        list(tqdm(executor.map(run_simulation, all_combinations), total=len(all_combinations)))

if __name__ == "__main__":
    main()

