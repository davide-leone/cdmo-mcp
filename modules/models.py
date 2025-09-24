import datetime
import gurobipy
import json
import numpy as np
import os
import pulp
import time
import z3

from minizinc import Instance, Model as MiniZincModel, Solver as MiniZincSolver
from z3 import *
from gurobipy import Model as GurobiModel, GRB, quicksum

from modules.utils import print_result
from modules.utils import extract_values_from_solution, get_tours_and_loads
from modules.utils import extract_tour_smt, print_debug_smt
from modules.utils import extract_tour_mip, print_debug_mip
from modules.heuristics import compute_lower_bound, compute_upper_bound, generate_heuristics, compute_distance_threshold

# ------------------------------------------------------------------------ #
#                                   CP                                     #  
# ------------------------------------------------------------------------ #

def create_and_test_model_smt_cp(solver_name, minizinc_file, approach_name, m, n, l, s, D, min_objective):
    '''
    Create and test the CP model on a particular instance of the problem
    
    '''
    # Create a MiniZinc model
    solver = MiniZincSolver.lookup(solver_name)
    
    model = MiniZincModel()
    model.add_file(minizinc_file)

    # Transform the model into a instance
    inst = Instance(solver, model)

    # Add the variables to the instance
    inst["m"] = m
    inst["n"] = n
    inst["l"] = l
    inst["s"] = s
    inst["D"]= D

    # Solve
    start_time = time.time()
    results = inst.solve(timeout=datetime.timedelta(minutes=5), processes=1, intermediate_solutions=True)
    end_time = time.time()
    
    last_solution = ""
    for sol in results:
        last_solution = str(sol)
    
    result = extract_values_from_solution(input_string=last_solution)
    
    # Additional output info
    approach = approach_name
    elapsed_time = int(end_time - start_time)
    objective = None
    optimal = False    
    loads = []
    distances = []
    tours = []
    solution = []
    
    if (last_solution == ""):
        elapsed_time = 300
        objective = np.inf
        # print_result(approach, elapsed_time, optimal, objective, distances, loads, tours)
        print_result(approach, elapsed_time, optimal, objective, tours)
    else:    
        # Retrieve individual results required for the output
        objective = result["objective"]
        
        if elapsed_time < 300 and objective <= min_objective:
            optimal = True
        else:
            elapsed_time = 300
        
        distances = result["distances"]
            
        loads, tours, solution = get_tours_and_loads(result["predecessor"], result["assigned_courier"], result["loads_int"], m, n)
        
        # Output results
        # print_result(approach, elapsed_time, optimal, objective, distances, loads, tours)
        print_result(approach, elapsed_time, optimal, objective, tours)
    
    return (elapsed_time, optimal, objective, solution)


# ------------------------------------------------------------------------ #
#                                  SMT                                     #  
# ------------------------------------------------------------------------ #
    
def create_and_test_model_smt(m, n, l, s, D, instance_number=None, load_heuristics=False, enable_pruning=False, n_sol=10, enable_debug=False):  
    '''
    Create and test the SMT model on a particular instance of the problem
    
    '''
    # MODEL INITIALIZATION
    start_time = time.time()
    time_limit = 300                                    # Max time limit for solving
    origin = n                                          # The origin node 
    distribution_points = list(range(n))                # Distribution points
    nodes = list(range(n+1))                            # All nodes including depot
    couriers = list(range(m))                           # List of couriers
    sizes = {dp: s[dp] for dp in distribution_points}   # Size demand at each distribution point
    loads = {c: l[c] for c in couriers}                 # Load capacity of each courier
    
    # Decision variable which indicates whether a courier travels from one node to another
    travels = [[[Bool(f"travels_{courier}_{from_dp}_{to_dp}") for to_dp in nodes] for from_dp in nodes] for courier in couriers]
    # Decision variable for subtour elimination (MTZ formulation)
    u = [Int(f"u_{dp}") for dp in range(n)]
    # Objective variable: max route length over all couriers
    max_dist = Int("max_dist")

    solver = Optimize()
    
    # CONSTRAINTS
    # Each courier must leave and return to the origin exactly once
    for courier in couriers:
        solver.add(And(PbEq([(travels[courier][origin][to_dp], 1) for to_dp in distribution_points], 1),
                       PbEq([(travels[courier][to_dp][origin], 1) for to_dp in distribution_points], 1)))

    # Each distribution point must be visited exactly once and exited exactly once
    for dp in distribution_points:
        solver.add(And(PbEq([(travels[courier][from_dp][dp], 1) for courier in couriers for from_dp in nodes], 1),
                       PbEq([(travels[courier][dp][to_dp], 1) for courier in couriers for to_dp in nodes], 1)))
    
    # Number of entries to a node equals number of exits (for each courier)
    for courier in couriers:
        for dp in nodes:
            solver.add(Sum([If(travels[courier][from_dp][dp], 1, 0) for from_dp in nodes]) ==
                       Sum([If(travels[courier][dp][to_dp], 1, 0) for to_dp in nodes]))
    
    # Prevent a courier from visiting the same node twice in a row
    for courier in couriers:
        for dp in distribution_points:
            solver.add(travels[courier][dp][dp] == False)
    
    # Total size of items picked up by a courier must not exceed their capacity
    for courier in couriers:
        solver.add(PbLe([(travels[courier][from_dp][to_dp], s[from_dp]) for from_dp in distribution_points for to_dp in nodes], l[courier]))
    
    # Subtour elimination using Miller-Tucker-Zemlin (MTZ) formulation
    for courier in couriers:
        for from_dp in distribution_points:
            for to_dp in distribution_points:
                if from_dp != to_dp:
                    solver.add(u[from_dp] + 1 <= u[to_dp] + n * (1 - If(travels[courier][from_dp][to_dp], 1, 0)))
        for dp in distribution_points:
            solver.add(u[dp] >= 1)
            solver.add(u[dp] <= n)
    
    # Track the max distance traveled by any courier
    for courier in couriers:
        route_length = Sum([If(travels[courier][from_dp][to_dp], D[from_dp][to_dp], 0) for from_dp in nodes for to_dp in nodes])
        solver.add(max_dist >= route_length)

    end_time = time.time()
    init_time = int(end_time - start_time)
    
    # SEARCH SPACE REDUCTION (Pruning)
    heuristic_time = 0
    pruning_time = 0
    heuristic_upper_bound = np.inf
    heuristic_data_path = None
    removed_arcs = 0
    all_arcs = m*(n+1)*(n+1)
    if enable_pruning:
        approach = "Heuristic+Pruning"
        heuristics_dir = "heuristics/SMT"
        if not os.path.exists(heuristics_dir):
            os.makedirs(heuristics_dir)
        if instance_number is not None:
            heuristic_data_path = os.path.join(heuristics_dir, f'instance_{instance_number}_heuristics.json')
        
        if load_heuristics and heuristic_data_path and os.path.exists(heuristic_data_path):
            # Load precomputed heuristic arcs from file if the parameter -l is specified
            with open(heuristic_data_path, 'r') as f:
                heuristic_info = json.load(f)

            n_sol = heuristic_info.get("n_sol", n_sol)
            heuristic_time = heuristic_info.get("heuristic_time", 0)
            heuristic_upper_bound = heuristic_info.get("heuristic_upper_bound", np.inf)
            candidate_arcs_list = heuristic_info.get("candidate_arcs", [])

            # candidate_arcs was saved as list of tuples, convert to set of tuples with correct types
            candidate_arcs = set()
            for arc in candidate_arcs_list:
                # The arcs are stored as lists in JSON, convert to tuple with int elements
                candidate_arcs.add(tuple(map(int, arc)))
        else:
            # Generate heuristics on-the-fly
            start_heur_time = time.time()
            heuristic_upper_bound, candidate_arcs = generate_heuristics(D, m, n, s, l, n_sol)
            end_heur_time = time.time()
            heuristic_time = end_heur_time - start_heur_time
        
        # Prune arcs not in heuristic set
        if candidate_arcs :
            start_time = time.time()
            for courier in couriers:
                for from_dp in nodes:
                    for to_dp in nodes:
                        if (courier, from_dp, to_dp) not in candidate_arcs :
                            solver.add(Not(travels[courier][from_dp][to_dp]))
                            removed_arcs += 1
            end_time = time.time()
            pruning_time = int(end_time - start_time)
        else:
            print("INFO: Pre-Clustering: No valid clusters found. Skipping pruning.\n")
    else:
        approach = "Basic"
    
    # BOUNDS ON OBJECTIVE
    lower_bound = compute_lower_bound(D, n)
    upper_bound = min(compute_upper_bound(D, n, m), heuristic_upper_bound)
    
    solver.add(max_dist >= lower_bound)
    solver.add(max_dist <= upper_bound)
    
    pruning_time += int(heuristic_time)
    
    # SOLVE THE MODEL
    remaining_time = time_limit - pruning_time
    solver.set("timeout", remaining_time*1000)
 
    start_time = time.time()
    solver.minimize(max_dist)
    result = solver.check()
    end_time = time.time()
    solving_time = int(end_time - start_time)
    total_time = pruning_time + solving_time
    
    # RESULTS EXTRACTION AND OUTPUT
    optimal = False
    tours = []
    solution = []
    objective = None
    elapsed_time = 300
    if result == sat:
        model = solver.model()
        objective = model.evaluate(max_dist).as_long()
        for courier in couriers:
            tour_matrix = [[is_true(model.evaluate(travels[courier][from_dp][to_dp])) for to_dp in nodes] for from_dp in nodes]
            tour = extract_tour_smt(tour_matrix)
            tours.append(tour)
            solution.append(tour[1:-1]) # Remove depot from the start/end of tour
        optimal = total_time <= 300
        elapsed_time = total_time
        print_result(approach, elapsed_time, optimal, objective, tours)
        
        # Save heuristics to file if they were generated
        if enable_pruning and not (load_heuristics and heuristic_data_path and os.path.exists(heuristic_data_path)):
            if heuristic_data_path:
                heuristic_info = {
                    "n_sol": n_sol,
                    "heuristic_time": heuristic_time,
                    "heuristic_upper_bound": heuristic_upper_bound,
                    "candidate_arcs": list(candidate_arcs) if candidate_arcs else []
                }
                with open(heuristic_data_path, 'w') as f:
                    json.dump(heuristic_info, f, indent=4)
        
    else:
        print_result(approach, elapsed_time, False, None, [])

    if enable_debug:
        print_debug_smt(total_time, init_time, pruning_time, solving_time, removed_arcs, all_arcs,lower_bound, upper_bound, result)

    return (elapsed_time, optimal, objective, solution)
    
# ------------------------------------------------------------------------ #
#                                  MIP                                     #  
# ------------------------------------------------------------------------ #

def create_and_test_model_pulp(m, n, l, s, D, instance_number=None, load_heuristics=False, enable_pruning=False, enable_debug=False): 
    '''
    Create and test the MIP model (Pulp-CBC) on a particular instance of the problem
    
    '''
    
    if enable_pruning:
        approach = "Pulp-CBC+Pruning"
    else:
        approach = "Pulp-CBC+Basic"
    
    # MODEL INITIALIZATION
    time_limit = 300
    start_time = time.time()
    
    origin = n
    distribution_points = list(range(n))
    nodes = distribution_points + [origin]
    couriers = list(range(m))
    tours = [(courier, from_dp, to_dp)
             for courier in couriers
             for from_dp in nodes
             for to_dp in nodes if from_dp != to_dp]
    sizes = {dp: s[dp] for dp in distribution_points}
    loads = {courier: l[courier] for courier in couriers}

    cost_matrix = {(from_dp, to_dp): D[from_dp][to_dp] for from_dp in range(n+1) for to_dp in range(n+1) if from_dp != to_dp}

    # Define PuLP model
    model = pulp.LpProblem("MCP", pulp.LpMinimize)

    # Variables
    travels = pulp.LpVariable.dicts("travels", tours, cat='Binary')
    u = pulp.LpVariable.dicts("u", [(courier, dp) for courier in couriers for dp in distribution_points], 
                          lowBound=1, upBound=n, cat='Integer')

    max_dist = pulp.LpVariable("max_dist", lowBound=0, cat='Integer')

    # Objective
    model += max_dist

    # CONSTRAINTS 
    # Each courier must leave and return to the origin exactly once
    for courier in couriers:
        model += pulp.lpSum(travels[(courier, origin, to_dp)] for to_dp in distribution_points) == 1
        model += pulp.lpSum(travels[(courier, from_dp, origin)] for from_dp in distribution_points) == 1
    
    # Each distribution point must be visited exactly once and exited exactly once
    for dp in distribution_points:
        model += pulp.lpSum(travels[(courier, from_dp, dp)]
                            for courier in couriers
                            for from_dp in nodes if from_dp != dp) == 1
        model += pulp.lpSum(travels[(courier, dp, to_dp)]
                            for courier in couriers
                            for to_dp in nodes if dp != to_dp) == 1
    
    # Number of entries to a node equals number of exits (for each courier)
    for courier in couriers:
        for dp in distribution_points:
            model += (pulp.lpSum(travels[(courier, from_dp, dp)] for from_dp in nodes if from_dp != dp) -
                      pulp.lpSum(travels[(courier, dp, to_dp)] for to_dp in nodes if dp != to_dp)) == 0
    
    # Total size of items picked up by a courier must not exceed their capacity
    for courier in couriers:
        model += pulp.lpSum(travels[(courier, from_dp, to_dp)] * sizes[to_dp]
                            for from_dp in nodes for to_dp in distribution_points if from_dp != to_dp) <= loads[courier]

    # Subtour elimination using Miller-Tucker-Zemlin (MTZ) formulation
    for courier in couriers:
        for from_dp in distribution_points:
            for to_dp in distribution_points:
                if from_dp != to_dp:
                    model += u[(courier, from_dp)] + 1 <= u[(courier, to_dp)] + n * (1 - travels[(courier, from_dp, to_dp)])

        for dp in distribution_points:
            model += u[(courier, dp)] >= 1
            model += u[(courier, dp)] <= n

    # Distance bound constraint
    for courier in couriers:
        model += pulp.lpSum(travels[(courier, from_dp, to_dp)] * cost_matrix[from_dp, to_dp]
                            for (from_dp, to_dp) in cost_matrix) <= max_dist

    end_time = time.time()
    init_time = int(end_time - start_time)
    
    # HEURISTIC SOLUTION
    heuristic_upper_bound = np.inf
    heuristic_time = 0
    if enable_pruning:
        heuristics_dir = "heuristics/MIP"
        if not os.path.exists(heuristics_dir):
            os.makedirs(heuristics_dir)
        if instance_number is not None:
            heuristic_data_path = os.path.join(heuristics_dir, f'instance_{instance_number}_heuristics.json')
        
        if load_heuristics and heuristic_data_path and os.path.exists(heuristic_data_path):
            # Load precomputed heuristic arcs from file if the parameter -l is specified
            with open(heuristic_data_path, 'r') as f:
                heuristic_info = json.load(f)

            heuristic_time = heuristic_info.get("heuristic_time", 0)
            heuristic_upper_bound = heuristic_info.get("heuristic_upper_bound", np.inf)
            candidate_arcs_list = heuristic_info.get("candidate_arcs", [])

            # candidate_arcs was saved as list of tuples, convert to set of tuples with correct types
            candidate_arcs = set()
            for arc in candidate_arcs_list:
                # The arcs are stored as lists in JSON, convert to tuple with int elements
                candidate_arcs.add(tuple(map(int, arc)))
        else:
            # Generate heuristics on-the-fly
            start_time = time.time()
            heuristic_upper_bound, candidate_arcs = generate_heuristics(D, m, n, s, l, n_sol=1)
            end_time = time.time()
            heuristic_time = int(end_time - start_time)
    
    # SEARCH SPACE REDUCTION (Pruning + Lower and Upper Bounds)
    start_time = time.time()
    
    # Compute the distance thresholds used to prune arcs
    thresholds, distance_threshold = compute_distance_threshold(m, n, D)
    
    # Remove the arcs between distribution points that are too distant to be in a good solution
    removed_arcs = 0
    if enable_pruning:
        for courier in couriers:
            for from_dp in distribution_points:
                for to_dp in distribution_points:
                    if (courier, from_dp, to_dp) not in candidate_arcs:
                        if from_dp != to_dp and D[from_dp][to_dp] > distance_threshold:
                            model += travels[(courier, from_dp, to_dp)] == 0
                            removed_arcs+=1
    
    # Compute bounds and add them on max_dist  
    lower_bound = compute_lower_bound(D, n)
    upper_bound = min(compute_upper_bound(D, n, m), heuristic_upper_bound)
    
    model += max_dist >= lower_bound
    model += max_dist <= upper_bound
    
    end_time = time.time()
    pruning_time = int(end_time - start_time)
    
    # SOLVE THE MODEL
    remaining_time = time_limit - heuristic_time - pruning_time
    
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=remaining_time)
    start_time = time.time()
    result_status = model.solve(solver)
    end_time = time.time()

    solving_time = int(end_time - start_time)
    total_time = solving_time + heuristic_time + pruning_time
    
    status = pulp.LpStatus[model.status]
    optimal = status == "Optimal"  
    
    # RESULTS EXTRACTION AND OUTPUT
    objective = np.inf
    output_tours = []
    output_solution = []
    
    objective = int(pulp.value(max_dist))
    output_tours, output_solution = extract_tour_mip(travels, origin, solver="pulp")
    
    if total_time >= 300 or optimal == False:
        optimal = False
        elapsed_time = 300
    else:
        elapsed_time = total_time
        # Save heuristics to file if they were generated
        if enable_pruning and not (load_heuristics and heuristic_data_path and os.path.exists(heuristic_data_path)):
            if heuristic_data_path:
                heuristic_info = {
                    "heuristic_time": heuristic_time,
                    "heuristic_upper_bound": heuristic_upper_bound,
                    "candidate_arcs": list(candidate_arcs) if candidate_arcs else []
                }
                with open(heuristic_data_path, 'w') as f:
                    json.dump(heuristic_info, f, indent=4)

    print_result(approach, elapsed_time, optimal, objective, output_tours)
    
    if enable_debug:
        print_debug_mip(total_time, init_time, heuristic_time, pruning_time, solving_time,
                    thresholds, distance_threshold, removed_arcs, lower_bound, upper_bound, status)
    
    return (elapsed_time, optimal, objective, output_solution)


def create_and_test_model_gurobi(m, n, l, s, D, instance_number=None, load_heuristics=False, enable_warm_start=False, enable_pruning=False, enable_debug=False): 
    '''
    Create and test the MIP model (Gurobi) on a particular instance of the problem
    
    '''
    # MODEL INITIALIZATION
    if enable_warm_start:
        if enable_pruning:
            approach = "Gurobi+Warm_Start+Pruning"
        else:
            approach = "Gurobi+Warm_Start"
    elif enable_pruning:
        approach = "Gurobi+Pruning"
    else:
        approach = "Gurobi+Basic"
        
    
    time_limit = 300
    start_time = time.time()
    
    origin = n
    distribution_points = list(range(n))
    nodes = distribution_points + [origin]
    couriers = list(range(m))
    tours = [(courier, from_dp, to_dp)
             for courier in couriers
             for from_dp in nodes
             for to_dp in nodes if from_dp != to_dp]
    sizes = {dp: s[dp] for dp in distribution_points}
    loads = {courier: l[courier] for courier in couriers}

    cost_matrix = {(from_dp, to_dp): D[from_dp][to_dp] for from_dp in nodes for to_dp in nodes if from_dp != to_dp}

    # Initialize Gurobi model
    model = GurobiModel("MultipleCourierProblem")
    model.Params.LogToConsole = 0
    
    # Variables
    travels = model.addVars(tours, vtype=GRB.BINARY, name="travels")
    u = model.addVars(couriers, distribution_points, vtype=GRB.INTEGER, lb=1, ub=n, name="u")
    max_dist = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="max_dist")

    # Objective: minimize max_dist
    model.setObjective(max_dist, GRB.MINIMIZE)

    # CONSTRAINTS
    # Each courier must leave and return to the origin exactly once
    for courier in couriers:
        model.addConstr(quicksum(travels[courier, origin, to_dp] for to_dp in distribution_points) == 1, name=f"start_{courier}")
        model.addConstr(quicksum(travels[courier, from_dp, origin] for from_dp in distribution_points) == 1, name=f"end_{courier}")

    # Each distribution point must be visited exactly once and exited exactly once
    for dp in distribution_points:
        model.addConstr(quicksum(travels[courier, from_dp, dp] for courier in couriers for from_dp in nodes if from_dp != dp) == 1, name=f"in_{dp}")
        model.addConstr(quicksum(travels[courier, dp, to_dp] for courier in couriers for to_dp in nodes if to_dp != dp) == 1, name=f"out_{dp}")

    # Number of entries to a node equals number of exits (for each courier)
    for courier in couriers:
        for dp in distribution_points:
            model.addConstr(
                quicksum(travels[courier, from_dp, dp] for from_dp in nodes if from_dp != dp) ==
                quicksum(travels[courier, dp, to_dp] for to_dp in nodes if to_dp != dp),
                name=f"flow_{courier}_{dp}"
            )

    # Total size of items picked up by a courier must not exceed their capacity
    for courier in couriers:
        model.addConstr(
            quicksum(travels[courier, from_dp, to_dp] * sizes[to_dp] for from_dp in nodes for to_dp in distribution_points if from_dp != to_dp) <= loads[courier],
            name=f"capacity_{courier}"
        )

    # Subtour elimination using Miller-Tucker-Zemlin (MTZ) formulation
    for courier in couriers:
        for from_dp in distribution_points:
            for to_dp in distribution_points:
                if from_dp != to_dp:
                    model.addConstr(
                        u[courier, from_dp] + 1 <= u[courier, to_dp] + n * (1 - travels[courier, from_dp, to_dp]),
                        name=f"mtz_{courier}_{from_dp}_{to_dp}"
                    )

        for dp in distribution_points:
            model.addConstr(u[courier, dp] >= 1, name=f"umin_{courier}_{dp}")
            model.addConstr(u[courier, dp] <= n, name=f"umax_{courier}_{dp}")

    # Per-Courier max distance constraint
    for courier in couriers:
        model.addConstr(
            quicksum(travels[courier, from_dp, to_dp] * cost_matrix[from_dp, to_dp] for (from_dp, to_dp) in cost_matrix) <= max_dist,
            name=f"max_dist_{courier}"
        )
    
    end_time = time.time()
    init_time = int(end_time - start_time)

    # HEURISTIC SOLUTION
    heuristic_upper_bound = np.inf
    heuristic_time = 0
    if enable_warm_start or enable_pruning:
        heuristics_dir = "heuristics/MIP"
        if not os.path.exists(heuristics_dir):
            os.makedirs(heuristics_dir)
        if instance_number is not None:
            heuristic_data_path = os.path.join(heuristics_dir, f'instance_{instance_number}_heuristics.json')
        
        if load_heuristics and heuristic_data_path and os.path.exists(heuristic_data_path):
            # Load precomputed heuristic arcs from file if the parameter -l is specified
            with open(heuristic_data_path, 'r') as f:
                heuristic_info = json.load(f)

            heuristic_time = heuristic_info.get("heuristic_time", 0)
            heuristic_upper_bound = heuristic_info.get("heuristic_upper_bound", np.inf)
            candidate_arcs_list = heuristic_info.get("candidate_arcs", [])

            # candidate_arcs was saved as list of tuples, convert to set of tuples with correct types
            candidate_arcs = set()
            for arc in candidate_arcs_list:
                # The arcs are stored as lists in JSON, convert to tuple with int elements
                candidate_arcs.add(tuple(map(int, arc)))
        else:
            # Generate heuristics on-the-fly
            start_time = time.time()
            heuristic_upper_bound, candidate_arcs = generate_heuristics(D, m, n, s, l, n_sol=1)
            end_time = time.time()
            heuristic_time = int(end_time - start_time)
    
    # WARM START
    start_time = time.time()
    # Set warm start if enabled
    if enable_warm_start:
        for (courier, from_dp, to_dp) in candidate_arcs:
                travels[courier, from_dp, to_dp].start = 1
    end_time = time.time()
    warm_start_time = int(end_time - start_time)

    # SEARCH SPACE REDUCTION (Pruning + Lower and Upper Bounds)
    start_time = time.time()
    
    # Compute the distance thresholds used to prune arcs
    thresholds, distance_threshold = compute_distance_threshold(m, n, D)
    
    # Remove the arcs between distribution points that are too distant to be in a good solution
    removed_arcs = 0
    if enable_pruning:
        for courier in couriers:
            for from_dp in distribution_points:
                for to_dp in distribution_points:
                    if (courier, from_dp, to_dp) not in candidate_arcs:
                        if from_dp != to_dp and D[from_dp][to_dp] > distance_threshold:
                            travels[courier, from_dp, to_dp].ub = 0
                            removed_arcs+=1
    
    # Compute bounds and add them on max_dist  
    lower_bound = float(compute_lower_bound(D, n))
    upper_bound = float(min(compute_upper_bound(D, n, m), heuristic_upper_bound))
    
    model.addConstr(max_dist >= lower_bound, name="max_dist_lower_bound")
    model.addConstr(max_dist <= upper_bound, name="max_dist_upper_bound")
    
    end_time = time.time()
    pruning_time = int(end_time - start_time)
    
    # SOLVE THE MODEL
    remaining_time = time_limit - heuristic_time - warm_start_time - pruning_time
    model.Params.TimeLimit = remaining_time
        
    start_time = time.time()
    model.optimize()
    status = model.status
    end_time = time.time()
    
    solving_time = int(end_time - start_time)
    total_time = init_time + heuristic_time + pruning_time + solving_time
        
    # RESULTS EXTRACTION AND OUTPUT
    optimal = status == GRB.OPTIMAL
    objective = np.inf
    output_tours = []
    output_solution = []
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        travel_values = {(courier, from_dp, to_dp): travels[courier, from_dp, to_dp].X for courier, from_dp, to_dp in tours if travels[courier, from_dp, to_dp].X > 0.5}
        objective = int(model.ObjVal)
        output_tours, output_solution = extract_tour_mip(travel_values, origin, solver="gurobi")
        
        # Save heuristics to file if they were generated
        if (enable_warm_start or enable_pruning) and not (load_heuristics and heuristic_data_path and os.path.exists(heuristic_data_path)):
            if heuristic_data_path:
                heuristic_info = {
                    "heuristic_time": heuristic_time,
                    "heuristic_upper_bound": heuristic_upper_bound,
                    "candidate_arcs": list(candidate_arcs) if candidate_arcs else []
                }
                with open(heuristic_data_path, 'w') as f:
                    json.dump(heuristic_info, f, indent=4)
    else:
        print("No feasible solution found.")

    if total_time >= 300 or optimal == False:
        optimal = False
        elapsed_time = 300
    else:
        elapsed_time = int(total_time)

    print_result(approach, elapsed_time, optimal, objective, output_tours)
    
    if enable_debug:
        print_debug_mip(total_time, init_time, heuristic_time, pruning_time, solving_time,
                    thresholds, distance_threshold, removed_arcs, lower_bound, upper_bound, status)
    
    return (elapsed_time, optimal, objective, output_solution)