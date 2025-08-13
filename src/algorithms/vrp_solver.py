"""
Vehicle Routing Problem Solver using Constraint Programming
"""

import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VRPSolver:
    """
    Vehicle Routing Problem (VRP) solver using Google OR-Tools
    
    Implements constraint programming solution for vehicle routing problems
    with multiple constraints including capacity, time windows, and distance limits.
    """
    
    def __init__(self, 
                 depot: Tuple[float, float] = (0, 0),
                 num_vehicles: int = 5,
                 vehicle_capacity: int = 100):
        """
        Initialize VRP solver
        
        Args:
            depot: Depot coordinates (x, y)
            num_vehicles: Number of available vehicles
            vehicle_capacity: Capacity of each vehicle
        """
        self.depot = depot
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        
        # Problem data
        self.locations = []
        self.demands = []
        self.distance_matrix = []
        
        # Solution data
        self.routes = []
        self.total_distance = 0
        self.vehicle_loads = []
        
    def create_sample_data(self, num_customers: int = 20):
        """Create sample VRP data"""
        logger.info(f"Creating sample VRP data with {num_customers} customers")
        
        # Generate random customer locations
        np.random.seed(42)  # For reproducible results
        
        # Add depot
        self.locations = [self.depot]
        
        # Generate customer locations
        for i in range(num_customers):
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-50, 50)
            self.locations.append((x, y))
        
        # Generate customer demands
        self.demands = [0]  # Depot has no demand
        for i in range(num_customers):
            demand = np.random.randint(5, 25)
            self.demands.append(demand)
        
        # Calculate distance matrix
        self._calculate_distance_matrix()
        
        logger.info(f"Created {len(self.locations)} locations with demands: {self.demands}")
    
    def _calculate_distance_matrix(self):
        """Calculate distance matrix between all locations"""
        num_locations = len(self.locations)
        self.distance_matrix = []
        
        for i in range(num_locations):
            row = []
            for j in range(num_locations):
                if i == j:
                    row.append(0)
                else:
                    # Euclidean distance
                    x1, y1 = self.locations[i]
                    x2, y2 = self.locations[j]
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    row.append(int(distance * 100))  # Convert to integer for OR-Tools
            self.distance_matrix.append(row)
    
    def solve_vrp(self, 
                  time_limit: int = 30,
                  use_guided_local_search: bool = True) -> Dict:
        """
        Solve the Vehicle Routing Problem
        
        Args:
            time_limit: Maximum solving time in seconds
            use_guided_local_search: Whether to use guided local search
            
        Returns:
            Dictionary containing solution and performance metrics
        """
        logger.info("Solving Vehicle Routing Problem...")
        
        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(self.distance_matrix), 
            self.num_vehicles, 
            0  # Depot index
        )
        
        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)
        
        # Define cost of each arc
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.demands[from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.vehicle_capacity] * self.num_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        if use_guided_local_search:
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
        
        search_parameters.time_limit.seconds = time_limit
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        # Extract solution
        if solution:
            self._extract_solution(manager, routing, solution)
            performance = self._calculate_performance()
            
            logger.info(f"VRP solved successfully!")
            logger.info(f"Total distance: {self.total_distance:.2f}")
            logger.info(f"Number of routes: {len(self.routes)}")
            
            return {
                'solution': solution,
                'routes': self.routes,
                'total_distance': self.total_distance,
                'vehicle_loads': self.vehicle_loads,
                'performance': performance,
                'status': 'OPTIMAL' if solution.ObjectiveValue() > 0 else 'FEASIBLE'
            }
        else:
            logger.error("No solution found!")
            return {
                'solution': None,
                'routes': [],
                'total_distance': float('inf'),
                'vehicle_loads': [],
                'performance': {},
                'status': 'INFEASIBLE'
            }
    
    def _extract_solution(self, manager, routing, solution):
        """Extract solution from the solver"""
        self.routes = []
        self.vehicle_loads = []
        self.total_distance = 0
        
        for vehicle_id in range(self.num_vehicles):
            route = []
            route_distance = 0
            route_load = 0
            
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                route_load += self.demands[node_index]
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            
            # Add depot at the end
            route.append(manager.IndexToNode(index))
            
            if len(route) > 1:  # Only add non-empty routes
                self.routes.append(route)
                self.vehicle_loads.append(route_load)
                self.total_distance += route_distance
    
    def _calculate_performance(self) -> Dict:
        """Calculate performance metrics"""
        if not self.routes:
            return {}
        
        # Calculate metrics
        num_routes = len(self.routes)
        avg_route_length = np.mean([len(route) - 1 for route in self.routes])  # -1 for depot
        avg_route_distance = self.total_distance / num_routes
        avg_vehicle_load = np.mean(self.vehicle_loads)
        load_utilization = (avg_vehicle_load / self.vehicle_capacity) * 100
        
        # Calculate route efficiency (lower is better)
        total_demand = sum(self.demands[1:])  # Exclude depot
        route_efficiency = self.total_distance / total_demand
        
        return {
            'num_routes': num_routes,
            'avg_route_length': avg_route_length,
            'avg_route_distance': avg_route_distance,
            'avg_vehicle_load': avg_vehicle_load,
            'load_utilization': load_utilization,
            'route_efficiency': route_efficiency,
            'total_demand': total_demand
        }
    
    def add_time_windows(self, 
                        time_windows: List[Tuple[int, int]],
                        service_times: List[int] = None):
        """
        Add time window constraints to the VRP
        
        Args:
            time_windows: List of (earliest, latest) time windows for each location
            service_times: Service time at each location
        """
        if not time_windows:
            return
        
        if len(time_windows) != len(self.locations):
            raise ValueError("Number of time windows must match number of locations")
        
        self.time_windows = time_windows
        self.service_times = service_times or [0] * len(self.locations)
        
        logger.info("Added time window constraints to VRP")
    
    def add_distance_constraints(self, max_distance: int):
        """
        Add maximum distance constraints for vehicles
        
        Args:
            max_distance: Maximum distance each vehicle can travel
        """
        self.max_distance = max_distance
        logger.info(f"Added distance constraint: max {max_distance} units per vehicle")
    
    def get_route_details(self, route_id: int) -> Dict:
        """Get detailed information about a specific route"""
        if route_id >= len(self.routes):
            return {}
        
        route = self.routes[route_id]
        route_load = self.vehicle_loads[route_id]
        
        # Calculate route distance
        route_distance = 0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            route_distance += self.distance_matrix[from_node][to_node]
        
        # Get location details
        locations = [self.locations[node] for node in route]
        demands = [self.demands[node] for node in route]
        
        return {
            'route_id': route_id,
            'nodes': route,
            'locations': locations,
            'demands': demands,
            'total_load': route_load,
            'total_distance': route_distance,
            'num_stops': len(route) - 1  # Exclude depot
        }
    
    def optimize_routes(self, 
                       num_customers: int = 20,
                       **kwargs) -> Dict:
        """
        Main method to optimize vehicle routes
        
        Args:
            num_customers: Number of customers to serve
            **kwargs: Additional parameters for optimization
            
        Returns:
            Optimization results
        """
        # Create sample data if not already created
        if not self.locations:
            self.create_sample_data(num_customers)
        
        # Solve VRP
        return self.solve_vrp(**kwargs)
