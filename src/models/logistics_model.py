"""
Logistics Optimization Model using Mixed-Integer Programming
"""

import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LogisticsOptimizer:
    """
    Logistics optimization model using Mixed-Integer Programming (MIP)
    
    Implements a comprehensive supply chain optimization model that minimizes
    total operational costs while satisfying demand and capacity constraints.
    """
    
    def __init__(self, 
                 warehouses: Optional[List[str]] = None,
                 customers: Optional[List[str]] = None,
                 products: Optional[List[str]] = None):
        """
        Initialize the logistics optimizer
        
        Args:
            warehouses: List of warehouse locations
            customers: List of customer locations
            products: List of product types
        """
        self.warehouses = warehouses or ['W1', 'W2', 'W3']
        self.customers = customers or ['C1', 'C2', 'C3', 'C4', 'C5']
        self.products = products or ['P1', 'P2', 'P3']
        
        # Initialize cost data
        self.transportation_costs = {}
        self.warehouse_costs = {}
        self.demand_data = {}
        self.capacity_data = {}
        
        self._initialize_sample_data()
        
    def _initialize_sample_data(self):
        """Initialize sample data for demonstration"""
        # Transportation costs (warehouse -> customer)
        for w in self.warehouses:
            for c in self.customers:
                self.transportation_costs[(w, c)] = np.random.uniform(10, 50)
        
        # Warehouse operational costs
        for w in self.warehouses:
            self.warehouse_costs[w] = np.random.uniform(1000, 3000)
        
        # Customer demand
        for c in self.customers:
            for p in self.products:
                self.demand_data[(c, p)] = np.random.randint(50, 200)
        
        # Warehouse capacity
        for w in self.warehouses:
            for p in self.products:
                self.capacity_data[(w, p)] = np.random.randint(300, 800)
    
    def create_mip_model(self) -> pulp.LpProblem:
        """
        Create the Mixed-Integer Programming model
        
        Returns:
            PuLP optimization problem
        """
        # Create optimization problem
        prob = pulp.LpProblem("Logistics_Optimization", pulp.LpMinimize)
        
        # Decision Variables
        # x[w,c,p]: amount of product p shipped from warehouse w to customer c
        x = pulp.LpVariable.dicts("shipment",
                                 [(w, c, p) for w in self.warehouses 
                                  for c in self.customers 
                                  for p in self.products],
                                 lowBound=0,
                                 cat='Continuous')
        
        # y[w]: binary variable indicating if warehouse w is used
        y = pulp.LpVariable.dicts("warehouse_used",
                                 self.warehouses,
                                 cat='Binary')
        
        # Objective Function: Minimize total cost
        # Total cost = Transportation costs + Warehouse operational costs
        prob += pulp.lpSum(
            self.transportation_costs[(w, c)] * x[(w, c, p)]
            for w in self.warehouses
            for c in self.customers
            for p in self.products
        ) + pulp.lpSum(
            self.warehouse_costs[w] * y[w]
            for w in self.warehouses
        )
        
        # Constraints
        
        # 1. Demand satisfaction: Each customer's demand must be met
        for c in self.customers:
            for p in self.products:
                prob += pulp.lpSum(x[(w, c, p)] for w in self.warehouses) >= \
                       self.demand_data[(c, p)], f"Demand_{c}_{p}"
        
        # 2. Capacity constraints: Warehouse capacity cannot be exceeded
        for w in self.warehouses:
            for p in self.products:
                prob += pulp.lpSum(x[(w, c, p)] for c in self.customers) <= \
                       self.capacity_data[(w, p)] * y[w], f"Capacity_{w}_{p}"
        
        # 3. Warehouse activation: If any product is shipped from a warehouse, it must be activated
        for w in self.warehouses:
            for c in self.customers:
                for p in self.products:
                    prob += x[(w, c, p)] <= self.capacity_data[(w, p)] * y[w], \
                           f"Activation_{w}_{c}_{p}"
        
        return prob
    
    def solve_optimization(self, 
                          time_limit: int = 300,
                          gap_tolerance: float = 0.05) -> Dict:
        """
        Solve the logistics optimization problem
        
        Args:
            time_limit: Maximum solving time in seconds
            gap_tolerance: Optimality gap tolerance
            
        Returns:
            Dictionary containing solution and performance metrics
        """
        logger.info("Creating MIP model...")
        prob = self.create_mip_model()
        
        logger.info("Solving optimization problem...")
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=gap_tolerance,
            msg=1
        ))
        
        # Extract solution
        solution = self._extract_solution(prob)
        
        # Calculate performance metrics
        performance = self._calculate_performance(solution)
        
        logger.info(f"Optimization completed. Status: {pulp.LpStatus[prob.status]}")
        logger.info(f"Total cost: ${performance['total_cost']:,.2f}")
        logger.info(f"Cost reduction: {performance['cost_reduction']:.1f}%")
        
        return {
            'problem': prob,
            'solution': solution,
            'performance': performance,
            'status': pulp.LpStatus[prob.status]
        }
    
    def _extract_solution(self, prob: pulp.LpProblem) -> Dict:
        """Extract solution from the solved problem"""
        solution = {
            'shipments': {},
            'warehouse_usage': {},
            'total_cost': pulp.value(prob.objective)
        }
        
        # Extract shipment decisions
        for w in self.warehouses:
            for c in self.customers:
                for p in self.products:
                    var_name = f"shipment_{w}_{c}_{p}"
                    for var in prob.variables():
                        if var.name == var_name:
                            if var.varValue > 0.01:  # Only non-zero shipments
                                solution['shipments'][(w, c, p)] = var.varValue
                            break
        
        # Extract warehouse usage decisions
        for w in self.warehouses:
            var_name = f"warehouse_used_{w}"
            for var in prob.variables():
                if var.name == var_name:
                    solution['warehouse_usage'][w] = var.varValue
                    break
        
        return solution
    
    def _calculate_performance(self, solution: Dict) -> Dict:
        """Calculate performance metrics"""
        # Calculate total cost
        total_cost = solution['total_cost']
        
        # Calculate baseline cost (naive solution)
        baseline_cost = self._calculate_baseline_cost()
        
        # Calculate cost reduction
        cost_reduction = ((baseline_cost - total_cost) / baseline_cost) * 100
        
        # Calculate utilization metrics
        utilization = self._calculate_utilization(solution)
        
        return {
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'cost_reduction': cost_reduction,
            'utilization': utilization
        }
    
    def _calculate_baseline_cost(self) -> float:
        """Calculate baseline cost using naive approach"""
        baseline_cost = 0
        
        # Naive approach: assign each customer to nearest warehouse
        for c in self.customers:
            for p in self.products:
                demand = self.demand_data[(c, p)]
                
                # Find nearest warehouse
                min_cost = float('inf')
                nearest_warehouse = None
                
                for w in self.warehouses:
                    cost = self.transportation_costs[(w, c)]
                    if cost < min_cost:
                        min_cost = cost
                        nearest_warehouse = w
                
                # Add transportation cost
                baseline_cost += demand * min_cost
                
                # Add warehouse operational cost (simplified)
                baseline_cost += self.warehouse_costs[nearest_warehouse] * 0.5
        
        return baseline_cost
    
    def _calculate_utilization(self, solution: Dict) -> Dict:
        """Calculate warehouse and route utilization"""
        utilization = {
            'warehouse_utilization': {},
            'route_utilization': {}
        }
        
        # Calculate warehouse utilization
        for w in self.warehouses:
            if solution['warehouse_usage'].get(w, 0) > 0.5:
                total_capacity = sum(self.capacity_data[(w, p)] for p in self.products)
                total_used = sum(
                    solution['shipments'].get((w, c, p), 0)
                    for c in self.customers
                    for p in self.products
                )
                utilization['warehouse_utilization'][w] = (total_used / total_capacity) * 100
        
        # Calculate route utilization
        for (w, c, p), amount in solution['shipments'].items():
            route_key = f"{w}->{c}"
            if route_key not in utilization['route_utilization']:
                utilization['route_utilization'][route_key] = 0
            utilization['route_utilization'][route_key] += amount
        
        return utilization
    
    def optimize_supply_chain(self, **kwargs) -> Dict:
        """
        Main method to optimize the supply chain
        
        Args:
            **kwargs: Additional parameters for optimization
            
        Returns:
            Optimization results
        """
        return self.solve_optimization(**kwargs)
    
    def get_cost_breakdown(self, solution: Dict) -> Dict:
        """Get detailed cost breakdown"""
        transportation_cost = 0
        warehouse_cost = 0
        
        # Calculate transportation costs
        for (w, c, p), amount in solution['shipments'].items():
            transportation_cost += amount * self.transportation_costs[(w, c)]
        
        # Calculate warehouse costs
        for w, used in solution['warehouse_usage'].items():
            if used > 0.5:
                warehouse_cost += self.warehouse_costs[w]
        
        return {
            'transportation_cost': transportation_cost,
            'warehouse_cost': warehouse_cost,
            'total_cost': transportation_cost + warehouse_cost
        }
