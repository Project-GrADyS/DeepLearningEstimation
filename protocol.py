"""
Protocol Implementation Module for Distributed Formation Control

This module implements the communication and control protocols for the distributed
formation control system. It includes implementations of the RAFT consensus protocol
and base station protocol for managing the formation of nodes.

Key Components:
- BaseStationProtocol: Manages the base station node that coordinates the formation
- RAFTProtocol: Implements the RAFT consensus algorithm for distributed nodes

Protocol Features:
- Distributed consensus using RAFT algorithm
- Formation control and maintenance
- Communication handling between nodes
- State management and synchronization
- Error handling and recovery
- Presence error calculation for formation monitoring

Usage:
    from protocol import BaseStationProtocol, RAFTProtocol
    # Create protocol instances with parameters
    base_station = BaseStationProtocol(params)
    raft_node = RAFTProtocol(params)

Dependencies:
    - gradysim: Simulation framework
    - numpy: Numerical computations
    - scipy: Scientific computations for distance calculations

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.1
"""

from parameters import Parameters
import logging
import math
import random
import json
from enum import Enum, auto
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.mobility import SetSpeedMobilityCommand
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.communication import CommunicationCommand, CommunicationCommandType
from gradysim.protocol.position import squared_distance, Position

# Node states for the Raft Consensus Algorithm
class NodeState(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()

# RAFT messages definitions
class RequestVote:
    def __init__(self, term, candidate_id):
        self.message_type = 'RequestVote'
        self.term = term
        self.candidate_id = candidate_id

    def to_json(self):
        message_dict = {
            'message_type': self.message_type,
            'term': self.term,
            'candidate_id': self.candidate_id
        }
        return json.dumps(message_dict)

    @staticmethod
    def from_json(json_str):
        message_dict = json.loads(json_str)
        return RequestVote(
            term=message_dict['term'],
            candidate_id=message_dict['candidate_id']
        )

class RequestVoteResponse:
    def __init__(self, term, vote_granted, voter_id):
        self.message_type = 'RequestVoteResponse'
        self.term = term
        self.vote_granted = vote_granted
        self.voter_id = voter_id

    def to_json(self):
        message_dict = {
            'message_type': self.message_type,
            'term': self.term,
            'vote_granted': self.vote_granted,
            'voter_id': self.voter_id
        }
        return json.dumps(message_dict)

    @staticmethod
    def from_json(json_str):
        message_dict = json.loads(json_str)
        return RequestVoteResponse(
            term=message_dict['term'],
            vote_granted=message_dict['vote_granted'],
            voter_id=message_dict['voter_id']
        )

class AppendEntries:
    def __init__(self, term, leader_id, value, consensual_sequence):
        self.message_type = 'AppendEntries'
        self.term = term
        self.leader_id = leader_id
        self.value = value
        self.consensual_sequence = consensual_sequence

    def to_json(self):
        message_dict = {
            'message_type': self.message_type,
            'term': self.term,
            'leader_id': self.leader_id,
            'value': self.value,
            'consensual_sequence': self.consensual_sequence
        }
        return json.dumps(message_dict)

    @staticmethod
    def from_json(json_str):
        message_dict = json.loads(json_str)
        return AppendEntries(
            term=message_dict['term'],
            leader_id=message_dict['leader_id'],
            value=message_dict['value'],
            consensual_sequence=message_dict['consensual_sequence']
        )

class AppendEntriesResponse:
    def __init__(self, term, success, follower_id, sequence):
        self.message_type = 'AppendEntriesResponse'
        self.term = term
        self.success = success
        self.follower_id = follower_id
        self.sequence = sequence

    def to_json(self):
        message_dict = {
            'message_type': self.message_type,
            'term': self.term,
            'success': self.success,
            'follower_id': self.follower_id,
            'sequence': self.sequence
        }
        return json.dumps(message_dict)

    @staticmethod
    def from_json(json_str):
        message_dict = json.loads(json_str)
        return AppendEntriesResponse(
            term=message_dict['term'],
            success=message_dict['success'],
            follower_id=message_dict['follower_id'],
            sequence=message_dict['sequence']
        )

class RAFTProtocol(IProtocol):
    """
    RAFT Consensus Protocol implementation for distributed nodes.
    
    This protocol implements the RAFT consensus algorithm for distributed
    formation control. It handles node communication, state management,
    and formation maintenance.
    """

    def __init__(self, params: Parameters):
        super().__init__()  # Call the base class constructor
        self.params = params  # Store the params object
        self._logger = logging.getLogger(__name__)
       
        self.is_recharge_movement = False
        self.recharge_probability_scan_time = 1 #scan time for the recharge probability in seconds 
        self.mission_timer_vector = [] #vector to store the time between the points of the mission
        self.relative_position: Position = (0, 0, 0) # Relative position of the follower to the virtual leader
        self.virtual_leader_position: Position = (0, 0, 0) # Position of the virtual leader
        self.virtual_leader_start_position: Position = (0, 0, 0) # Start position of the virtual leader between two waypoints
        self.virtual_leader_end_position: Position = (0, 0, 0) # End position of the virtual leader between two waypoints

        # Recharge battery follower parameters
        self.recharge_probability = self.params.RECHARGE_PROBABILITY  # Recharge battery follower probability
        self.recharge_time = self.params.RECHARGE_TIME  # Recharge battery follower time in seconds
        self.recharge_position = self.params.RECHARGE_POSITION  # Recharge battery follower position

        # Formation circle parameters
        self.radius = self.params.FORMATION_RADIUS  # Formation circle radius
        self.points_num = self.params.FORMATION_POINTS_NUM  # This will create a circle with n points
        self.center_x, self.center_y = 0,0  # Circle center

        # Generating points for the circle (relative to the virtual leader)
        self.circle_points = []
        for i in range(self.points_num):
            angle = (2 * math.pi / self.points_num) * i
            x = self.radius * math.cos(angle) + self.center_x
            y = self.radius * math.sin(angle) + self.center_y
            self.circle_points.append((x, y, self.params.MISSION_ALTITUDE))

    def initialize(self) -> None:
        self.node_id = self.provider.get_id() # get the node id
        self.set_relative_position(self.circle_points[self.node_id-1]) #set the relative position of the node to the virtual leader

        self.peers = [i for i in range(1, self.params.FORMATION_POINTS_NUM + 1) if i != self.node_id] #get its peers: all the nodes_id except itself
        self.current_term = 0  # Current term of the node
        self.voted_for = None  # Node ID for which the node has voted
        self.role = NodeState.FOLLOWER  # Initial role of the node
        self.commit_value = [1]*self.params.FORMATION_POINTS_NUM  # Committed value
        self.nodes_status_temp = [1]*self.params.FORMATION_POINTS_NUM  # Vector to store the measure of the state of the nodes
        self.votes_received = set()  # Received votes
        self.election_timeout = random.uniform(0.15, 0.30) # Random election timeout between 150 and 300ms
        self.heartbeat_timeout = self.params.HEARTBEAT_FACTOR * self.election_timeout  # Heartbeat timeout
        self.waypoint = 0 #index of the first waypoint of the virtual mission
        self.line_step = 0 #counter that varies along the line defined by two waypoints
        self.line_step_timeout = self.params.MISSION_STEP_TIME #timeout for the line step
        self.total_time = 0 #total time between two waypoints
        self.total_steps = 0 #total number of steps between two waypoints

        self.sequence = -1 #current sequence
        self.consensual_sequence = -1 #consensual sequence
        self.sequence_list = [-1]*self.params.FORMATION_POINTS_NUM #list of sequences for each node
        self.sequence_list_temp = [-1]*self.params.FORMATION_POINTS_NUM #temporary list of sequences for each node
        
        # Set the speed of the nodes of the formation
        command = SetSpeedMobilityCommand(self.params.SPEED_FOLLOWER)
        self.provider.send_mobility_command(command)

        # Parameters of the mission of the virtual leader
        radius = self.params.MISSION_RADIUS  # Mission circle radius
        points_num = self.params.MISSION_POINTS_NUM  # Mission points number

        # Points of the virtual mission
        self.mission_points = []  # List of mission points

        # Upper Half (clockwise)
        center_x, center_y = self.params.FORMATION_CENTER_X, self.params.FORMATION_CENTER_Y+radius
        for i in range(1, points_num + 1):
            angle = 1.5 * math.pi - (2 * math.pi / points_num) * i  # Clockwise: decrease the angle
            x = radius * math.cos(angle) + center_x
            y = radius * math.sin(angle) + center_y
            self.mission_points.append((x, y, self.params.MISSION_ALTITUDE))

        # Lower Half (anti-clockwise)
        center_x, center_y = self.params.FORMATION_CENTER_X, self.params.FORMATION_CENTER_Y-radius
        for i in range(1, points_num + 1):
            angle = 0.5 * math.pi + (2 * math.pi / points_num) * i  # Anti-clockwise: increase the angle
            x = radius * math.cos(angle) + center_x
            y = radius * math.sin(angle) + center_y
            self.mission_points.append((x, y, self.params.MISSION_ALTITUDE))

        # Calculate the timer vector between points of the mission (distance/speed_leader)
        self.mission_timer_vector = [math.sqrt(squared_distance(self.mission_points[i], self.mission_points[i+1])) / self.params.SPEED_LEADER for i in range(len(self.mission_points)-1)]
        self.mission_timer_vector.append(math.sqrt(squared_distance(self.mission_points[-1], self.mission_points[0])) / self.params.SPEED_LEADER) # Add the time between the last and the first point

        # Schedule the timers for the first time
        self.schedule_recharge_probability_timer()
        self.schedule_election_timer()
        self.schedule_heartbeat_timer()
        self.go_to_next_sequence()

    def schedule_return_to_formation_timer(self):
        self.provider.schedule_timer("return_to_formation", self.provider.current_time() + self.recharge_time) 

    def schedule_recharge_probability_timer(self):
        self.provider.schedule_timer("recharge_probability", self.provider.current_time() + self.recharge_probability_scan_time)

    def schedule_election_timer(self):
        self.provider.schedule_timer("start_election", self.provider.current_time() + self.election_timeout)

    def schedule_heartbeat_timer(self):
        if self.role == NodeState.LEADER:
            self.provider.schedule_timer("heartbeat", self.provider.current_time() + self.heartbeat_timeout)

    def schedule_mission_timer(self):
        mission_noise = max(-1, min(np.random.normal(1, self.params.MISSION_TIMER_STD_DEV), 1)) #random noise in the mission timer limited to [-1,1]
        self.provider.schedule_timer("mission_timer", self.provider.current_time() + self.mission_timer_vector[self.waypoint]*mission_noise)
       
    def schedule_line_step_timer(self):
        self.provider.schedule_timer("line_step_timer", self.provider.current_time() + self.line_step_timeout)
        
    def handle_timer(self, timer: str) -> None:
        if timer == "return_to_formation": 
            self.return_to_formation()
            return

        if timer == "recharge_probability":
            if random.random() < self.recharge_probability:
                self.recharge_movement()
            self.schedule_recharge_probability_timer()
            return
        
        if timer == "start_election" and not self.is_recharge_movement:
            self.start_election()
            return
        
        if timer == "heartbeat" and not self.is_recharge_movement:
            self.send_heartbeats()
            self.schedule_heartbeat_timer()
            return
        
        if timer == "mission_timer":
            self.go_to_next_sequence()
            return
        
        if timer == "line_step_timer":
            self.move_node_along_line()
            return
    
    def move_node_along_line(self):
        self.line_step += 1 #increment the counter that varies along the line defined by the waypoints
        t = self.line_step * self.line_step_timeout / self.total_time #calculate the parameter t
        #schedule the line step timer if the counter is less than the total steps
        if self.line_step < self.total_steps:
            self.schedule_line_step_timer() 
        # Update the virtual leader position
        self.virtual_leader_position = tuple(p1 + (p2 - p1) * t for p1, p2 in zip(self.virtual_leader_start_position, self.virtual_leader_end_position))
        # Updating the relative position to the virtual leader
        self.update_relative_position(self.commit_value, self.node_id, self.params.FORMATION_RADIUS)
        # Go to the virtual leader's position at relative coordinates
        destination = (coord + relative_coord 
                        for coord, relative_coord in zip(self.virtual_leader_position, self.relative_position))
        mobility_command = GotoCoordsMobilityCommand(*destination)
        if not self.is_recharge_movement: #only if the node is not recharging
            self.provider.send_mobility_command(mobility_command)

    def go_to_next_sequence(self): 
        self.sequence_list_temp = [-1] * self.params.FORMATION_POINTS_NUM  # reset the sequence list
        self.schedule_mission_timer() # Fire the mission timer again to the next sequence
        self.waypoint = self.sequence % len(self.mission_points) # current waypoint
        self.virtual_leader_start_position = self.mission_points[self.waypoint] # virtual leader start position between two waypoints
        self.total_time = self.mission_timer_vector[self.waypoint] # calculate the total time between the two waypoints
        self.total_steps = int(self.total_time / self.line_step_timeout) # calculate the total number of steps between the two waypoints
        self.sequence += 1 # next sequence
        self.waypoint = self.sequence % len(self.mission_points) # next waypoint
        self.virtual_leader_end_position = self.mission_points[self.waypoint] # virtual leader end position between two waypoints
        self.line_step = 0 # re-initialize the counter that varies along the line defined by the waypoints start and end
        self.provider.cancel_timer("line_step_timer") # cancel the timer if it is already scheduled
        self.schedule_line_step_timer() # fire the line step timer

    def set_relative_position(self, position: Position) -> None:
        self.relative_position = position

    def update_relative_position(self, nodes_status, nodeID, radius) -> None:
        if nodes_status[nodeID-1] == 1:
            active_nodes = [i for i, x in enumerate(nodes_status, start=1) if x == 1]
            relative_position = active_nodes.index(nodeID) + 1
            actives_nodes_count = len(active_nodes)
            i = actives_nodes_count-relative_position
            angle = (2 * math.pi / actives_nodes_count) * i
            # circle coordinates
            x_circle = radius * math.cos(angle) 
            y_circle = radius * math.sin(angle) 
            # line coordinates
            y_line = 0
            if actives_nodes_count < 2: # To avoid division by zero
                x_line = 0
            else:
                x_line = radius*(1-2*i/(actives_nodes_count-1))
            #alternate between shapes
            if self.waypoint >= self.params.MISSION_POINTS_NUM and self.params.FORMATION_ALTERNATE:
                x = x_line
                y = y_line
                self.params.FORMATION_TYPE = "line"
            else:
                x = x_circle
                y = y_circle
                self.params.FORMATION_TYPE = "circle"
            z = self.params.MISSION_ALTITUDE
            self.set_relative_position((x, y, z))

    def start_election(self):
        if self.role == NodeState.LEADER:  # if the role is leader
            return
        self.current_term += 1  # increment the current term
        self.voted_for = self.node_id  # vote for itself
        self.role = NodeState.CANDIDATE  # change role to candidate
        self.params.LEADER_OFF = 1  # indicate that there is no RAFT leader
        self.votes_received = {self.node_id}  # Add own vote to the received votes set
        message = RequestVote(self.current_term, self.node_id) # Create a RequestVote message
        json_message = message.to_json() # Convert the message to JSON
        for peer in self.peers: # Send the message to all peers
            command = CommunicationCommand(CommunicationCommandType.SEND,json_message,peer)
            self.provider.send_communication_command(command)
        return

    def handle_packet(self, message_str:str) -> None:    
        # Do not consider broadcast messages from the leader of the formation OR if the node is recharging its battery
        if self.is_recharge_movement:
            return        
        # Deserializes the message, identifies its type and forwards it for appropriate processing
        try:
            message_dict = json.loads(message_str)
            message_type = message_dict['message_type']
            if message_type == 'RequestVote':
                message = RequestVote.from_json(message_str)
                self.handle_request_vote(message)
            elif message_type == 'RequestVoteResponse':
                message = RequestVoteResponse.from_json(message_str)
                self.handle_request_vote_response(message)
            elif message_type == 'AppendEntries':
                message = AppendEntries.from_json(message_str)
                self.handle_append_entries(message)
            elif message_type == 'AppendEntriesResponse':
                message = AppendEntriesResponse.from_json(message_str)
                self.handle_append_entries_response(message)
        except json.JSONDecodeError:
            self._logger.error("Failed to decode message")

    # handle a request vote message
    def handle_request_vote(self, message: RequestVote) -> None:
        candidate_term = message.term
        candidate_id = message.candidate_id 
        if candidate_term > self.current_term:
            self.current_term = candidate_term
            self.role = NodeState.FOLLOWER  # change the role to follower
            self.voted_for = None
        vote_granted = False
        if candidate_term == self.current_term and (self.voted_for is None or self.voted_for == candidate_id):
            self.voted_for = candidate_id
            vote_granted = True
        json_message = RequestVoteResponse(self.current_term, vote_granted, self.node_id).to_json()
        command = CommunicationCommand(CommunicationCommandType.SEND,json_message,candidate_id)
        self.provider.send_communication_command(command)

    # handle a request vote response message
    def handle_request_vote_response(self, message: RequestVoteResponse) -> None:
        if message.term > self.current_term:  # if the term is greater than the current term
            self.current_term = message.term  # update the current term
            self.role = NodeState.FOLLOWER  # change the role to follower
            self.voted_for = None  # reset the voted for
            self.provider.cancel_timer("start_election")  # reset the current election timer
            self.schedule_election_timer()  # schedule the election timer
            return
        if (self.role == NodeState.CANDIDATE and message.term == self.current_term and message.vote_granted):  # if the role is candidate and the term is the same as the current term and the vote is granted
            self.votes_received.add(message.voter_id)  # add the voter id to the votes received set
            if len(self.votes_received) > self.commit_value.count(1) // 2:  # if the votes received is greater than half of the active peers
                self.role = NodeState.LEADER  # change the role to leader
                self.params.LEADER_OFF = 0  # indicate that there is a RAFT leader
                self._logger.info(f"Node {self.node_id} is the leader") if self.params.TERMINAL_MESSAGES else None # Log the message
                self.send_heartbeats()  # send heartbeats to maintain leadership
                self.provider.cancel_timer("start_election")  # reset the current election timer
                self.schedule_heartbeat_timer()

    # send heartbeat to every peer to maintain leadership
    def send_heartbeats(self) -> None:
        if self.role != NodeState.LEADER:  # if the role is not leader
            return
        self.commit_value = self.nodes_status_temp  # update the commit value
        self.sequence_list = self.sequence_list_temp  # update the sequence list
        self.consensual_sequence = self.find_majority_element(self.sequence_list)  # find the majority element in the sequence list
        if self.consensual_sequence != -1 and self.consensual_sequence != self.sequence:  # if the consensual sequence is different from -1 and different from the current sequence
            self.fix_sequence(self.sequence, self.consensual_sequence)  # fix the sequence
        self.params.STATUS_VECTOR[self.node_id-1] = self.nodes_status_temp  # update the status vector
        self.nodes_status_temp = [0] * self.params.FORMATION_POINTS_NUM  # reset the vector
        self.nodes_status_temp[self.node_id-1] = 1  # update the vector indicating that the leader is active
        self.sequence_list_temp = [-1] * self.params.FORMATION_POINTS_NUM  # reset the sequence list
        self.sequence_list_temp[self.node_id-1] = self.sequence  # update the sequence list
        for peer in self.peers:
            json_message = AppendEntries(
                self.current_term, 
                self.node_id, 
                self.commit_value, 
                self.consensual_sequence
            ).to_json()
            command = CommunicationCommand(CommunicationCommandType.SEND,json_message,peer)
            self.provider.send_communication_command(command)  # send the message to all peers

    # handle an append entries message
    def handle_append_entries(self, message: AppendEntries) -> None:
        if message.term >= self.current_term:  # if the term is greater or equal to the current term
            self.current_term = message.term  # update the current term
            self.role = NodeState.FOLLOWER  # change the role to follower
            self.provider.cancel_timer("start_election")  # reset the current election timer
            self.schedule_election_timer()  # schedule a new election timer
            success=True # success of the append entries
        else:
            success=False # failure of the append entries
        if self.role != NodeState.LEADER:   
            self.commit_value = self.nodes_status_temp = message.value  # update the commit value and the temporary nodes status vector
            self.params.STATUS_VECTOR[self.node_id-1] = self.commit_value  # update the status vector
            if message.consensual_sequence != -1 and message.consensual_sequence != self.sequence:  # if the consensual sequence is different from -1 and different from the current sequence
                self.fix_sequence(self.sequence, message.consensual_sequence)  # fix the sequence

        json_message = AppendEntriesResponse(self.current_term, success, self.node_id,self.sequence).to_json()
        command = CommunicationCommand(CommunicationCommandType.SEND,json_message,message.leader_id)
        self.provider.send_communication_command(command)

    def handle_append_entries_response(self, message: AppendEntriesResponse) -> None:
        if message.term > self.current_term:  # if the term is greater than the current term
            self.current_term = message.term  # update the current term
            self.role = NodeState.FOLLOWER  # change the role to follower
            self.voted_for = None  # reset the voted for
            self.provider.cancel_timer("start_election")  # reset the current election timer
            self.schedule_election_timer()  # schedule a new election timer
            
        # update the vector indicating that the follower is active
        self.nodes_status_temp[message.follower_id-1] = 1  

        # update the sequence list
        self.sequence_list_temp[message.follower_id-1] = message.sequence  
        
    def find_majority_element(self, nums):
        """
        Boyer-Moore algorithm for finding the majority element in a list of elements. O(n) complexity.
        """
        candidate = None
        count = 0
        # Step 1: Find a possible candidate for majority
        for num in nums:
            if num < 0:
                continue  # Skip negative numbers, including -1
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1) 
        # Step 2: Check if the candidate is really the majority and non negative
        count = sum(1 for num in nums if num == candidate)       
        if count > self.commit_value.count(1) // 2 and candidate >= 0 and self.role == NodeState.LEADER:    
            return candidate
        else:
            return -1

    def fix_sequence(self, sequence, consensual_sequence):
        if self.params.MISSION_FIX:
            # if consensual sequence is less than the current sequence
            if consensual_sequence < sequence: 
                self.line_step = 0  # reset the line step
                #self.sequence = consensual_sequence + 1  # update the sequence
            else:
                self.provider.cancel_timer("mission_timer")  # cancel the mission timer
                self.provider.schedule_timer("mission_timer", self.provider.current_time() + self.params.UPDATE_RATE/2) # fire the mission timer again with a very small delay
                #self.sequence = consensual_sequence - 1  # update the sequence

    def recharge_movement(self):
        if self.is_recharge_movement == False:
            self.is_recharge_movement = True
            mobility_command = GotoCoordsMobilityCommand(*self.recharge_position)
            self.provider.send_mobility_command(mobility_command)
            self.schedule_return_to_formation_timer()
            self.role = NodeState.FOLLOWER  # change the role to follower
            self.nodes_status_temp[self.node_id-1] = 0  # update the vector indicating that the node is recharging
            self.provider.cancel_timer("heartbeat")  # reset the current heartbeat timer to ensure that the leader does not send heartbeats while recharging
            self.provider.cancel_timer("start_election") # reset the current election timer to ensure that the leader does not start an election while recharging
            self.params.CORRECT_VECTOR[self.node_id-1] = 0 # update the standard vector to indicate that the node is recharging; metrics purposes
            self._logger.info(f"Node {self.node_id} is recharging") if self.params.TERMINAL_MESSAGES else None # Log the message

    def return_to_formation(self):
        if self.is_recharge_movement == True:
            self.is_recharge_movement = False
            self.nodes_status_temp[self.node_id-1] = 1  # update the vector indicating that the node is active
            self.schedule_election_timer()  # schedule a new election timer because the node is back to the formation
            self.schedule_heartbeat_timer()  # schedule a new heartbeat timer because the node is back to the formation
            self.params.CORRECT_VECTOR[self.node_id-1] = 1 # update the standard vector to indicate that the node is active again; metrics purposes
            self._logger.info(f"Node {self.node_id} is back to the formation") if self.params.TERMINAL_MESSAGES else None # Log the message

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        # Update the position of the node in the global variable to calculate the shape error
        self.params.SHAPE_VECTOR[self.node_id-1] = telemetry.current_position

    def finish(self) -> None:
        pass

class BaseStationProtocol(IProtocol):
    """
    Base Station Protocol for formation control.
    
    This protocol manages the base station node that coordinates the formation.
    It handles initialization, communication, and formation control tasks.
    """

    def __init__(self, params: Parameters):
        super().__init__()  # Call the base class constructor
        self.params = params  # Store the params object
        self._logger = logging.getLogger(__name__)
        
        # Cache para cálculos vetoriais
        self._shape_vector_cache = None
        self._correct_vector_cache = None
        self._active_nodes_cache = None
        self._center_of_mass_cache = None
        self._neighbor_distances_cache = None
        self._center_distances_cache = None
        self._last_calculation_time = 0
        self._cache_valid = False

    def initialize(self) -> None:
        if not hasattr(self, 'df'):
            self.df = pd.DataFrame(columns=["timestamp", "shape_error", "presence_error", "leader_off"])

        # Initialize the shape error variable
        self.shape_error = 0
        # Initialize the leader_off variable from params
        self.leader_off = self.params.LEADER_OFF
        # Initialize presence error variable
        self.presence_error = 0
    
    def handle_timer(self, timer: str) -> None:
        pass

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        # Calculate the presence error
        self.calculate_presence_error(self.params.CORRECT_VECTOR, self.params.STATUS_VECTOR)
        # Calculate the shape error and update the global variables
        self.calculate_shape_error(self.params.CORRECT_VECTOR, self.params.SHAPE_VECTOR, self.params.FORMATION_TYPE, self.params.FORMATION_RADIUS)
        # Log the metrics in a dataframe
        self.log()

    def calculate_presence_error(self, correct_vector, status_vector) -> None:
        # Calculate the point-wise error for each vector in STATUS_VECTOR
        point_errors = [hamming(correct_vector, status) for status in status_vector]
        # Calculate the average error (error_pontual)
        self.presence_error = sum(point_errors)/len(point_errors)

    def calculate_shape_error(self, correct_vector, shape_vector, formation_type, radius) -> None:
        # Pré-alocação dos arrays com tipo específico para melhor performance
        shape_vector = np.asarray(shape_vector, dtype=np.float64)
        correct_vector = np.asarray(correct_vector, dtype=np.float64)
        
        # Filtragem eficiente usando máscara booleana
        active_mask = correct_vector == 1
        if not np.any(active_mask):
            self.shape_error = 0
            return
            
        # Extração de nós ativos usando máscara
        active_nodes = shape_vector[active_mask]
        num_points = len(active_nodes)
        
        # Tratamento para casos com menos de 2 pontos
        if num_points < 2:
            self.shape_error = 0
            return

        # Cálculo do centro de massa usando operação vetorizada
        center_of_mass = np.mean(active_nodes, axis=0)
        
        # Cálculo de distâncias usando operações vetorizadas
        next_points = np.roll(active_nodes, -1, axis=0)
        neighbor_distances = np.linalg.norm(active_nodes - next_points, axis=1)
        center_distances = np.linalg.norm(active_nodes - center_of_mass, axis=1)
        
        # Pré-alocação de arrays para distâncias ideais
        ideal_neighbor_distances = np.empty(num_points, dtype=np.float64)
        ideal_center_distances = np.empty(num_points, dtype=np.float64)
        
        # Cálculo das distâncias ideais otimizado
        if formation_type == "circle":
            # Cálculo otimizado para formação circular
            ideal_neighbor_distance = 2 * radius * np.sin(np.pi / num_points)
            ideal_neighbor_distances.fill(ideal_neighbor_distance)
            ideal_center_distances.fill(radius)
        else:  # line
            # Cálculo otimizado para formação linear
            if num_points > 1:
                ideal_neighbor_distances.fill(2 * radius / (num_points-1))
                ideal_neighbor_distances[-1] = 2 * radius
                line_center = (active_nodes[0] + active_nodes[-1]) / 2.0
                ideal_center_distances = np.linalg.norm(active_nodes - line_center, axis=1)
                ideal_center_distances[ideal_center_distances == 0] = 1e-9
            else:
                self.shape_error = 0
                return

        # Combinação de arrays usando operação vetorizada
        distances = np.concatenate((neighbor_distances, center_distances))
        ideal_distances = np.concatenate((ideal_neighbor_distances, ideal_center_distances))
        
        # Cálculo do erro normalizado usando operações vetorizadas
        error = np.mean(np.abs(distances - ideal_distances) / ideal_distances)
        error_norm = error / (1 + error)

        # Atualiza o shape_error
        self.shape_error = error_norm

    def log(self) -> None:
        timestamp = self.provider.current_time()
        new_row = pd.DataFrame({
            "timestamp":[timestamp], 
            "shape_error":[self.shape_error],
            "presence_error":[self.presence_error],
            "leader_off":[self.params.LEADER_OFF]})
        
        if not new_row.empty:  # Ensure that `new_row` is not empty
            if self.df.empty:
                self.df = new_row  # Initialize `self.df` if empty
            else:
                self.df = pd.concat([self.df, new_row], ignore_index=True)  # Add `new_row` to `self.df`

    def finish(self) -> None:
        self._logger.info(f"Average Shape Error: {self.shape_error}") if self.params.TERMINAL_MESSAGES else None # Log the average shape error
        self._logger.info(f"Average Leader Off: {self.leader_off}") if self.params.TERMINAL_MESSAGES else None # Log the average leader off
        self._logger.info(f"Average Presence Error: {self.presence_error}") if self.params.TERMINAL_MESSAGES else None # Log the average presence error
        self.df.to_csv(self.params.FILE_PATH, index=False)
        
