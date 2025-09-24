import json
import os
import pulp
import re


# ------------------------------------------------------------------------ #
#                                 COMMON                                   #  
# ------------------------------------------------------------------------ #

def read_dat_file(filename):
    '''
    Read the inputs from the instance file
    
    '''    
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Read the number of couriers
    m = int(lines[0].strip())

    # Read the number of items
    n = int(lines[1].strip())

    # Read the maximum load for each courier
    l = list(map(int, lines[2].strip().split()))

    # Read the sizes of items
    s = list(map(int, lines[3].strip().split()))

    # Read the distance matrix
    D = []
    for i in range(4, 4 + n + 1):
        D.append(list(map(int, lines[i].strip().split())))

    return m, n, l, s, D
    

def print_result(approach, elapsed_time, optimal, objective, tours):
    '''
    Print the results of the model in a nice format
    
    '''
    print("----------------------------------------------------------------------------")
    print("\nApproach: {}\n".format(approach))
    print("Time:     {} s".format(elapsed_time))
    print("Optimal:  {}".format(optimal))
    print()
    print(" --- RESULTS --- \n")
    print("Max Distance: {}\n".format(objective))
    n_c = 1
    for tour in tours:
        tour_string = " -> ".join(str(dp) for dp in tour)
        print("Tour of courier {}: {}".format(n_c, tour_string))
        n_c+=1
    print()
    
    
def create_json(directory, instance, approaches, times, optimalities, objectives, solutions):
    '''
    Save results in a JSON file
    
    '''
    # Create a dictionary to hold the data
    data = {}

    # Iterate over approaches and populate data dictionary
    for i in range(len(approaches)):
        approach = approaches[i]
        time = times[i]
        optimal = optimalities[i]
        obj = objectives[i]
        sol = solutions[i]
        
        data[approach] = {
            "time": time,
            "optimal": optimal,
            "obj": obj,
            "sol": sol
        }

    # Define the path for the JSON file
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f'{instance}.json')

    # Write data to JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# ------------------------------------------------------------------------ #
#                                   CP                                     #  
# ------------------------------------------------------------------------ #

def extract_values_from_solution(input_string):
    '''
    Parse the last solution of solve to get the values necessary for the output
    
    '''
    extracted_values = {}
    
    obj_match = re.search(r'Objective:\s+(\d+)', input_string)
    if obj_match:
        extracted_values['objective'] = int(obj_match.group(1))
        
    bounds_match = re.search(r'Bounds:\s+\[(\d+),\s*(\d+)\]', input_string)
    if bounds_match:
        extracted_values['bounds'] = [int(bounds_match.group(1)), int(bounds_match.group(2))]
        
    distances_match = re.search(r'Distances:\s+\[([0-9,\s]+)\]', input_string)
    if distances_match:
        extracted_values['distances'] = [int(x) for x in distances_match.group(1).split(',')]
        
    successor_match = re.search(r'Successor:\s+\[([0-9,\s]+)\]', input_string)
    if successor_match:
        extracted_values['successor'] = [int(x) for x in successor_match.group(1).split(',')]
        
    predecessor_match = re.search(r'Predecessor:\s+\[([0-9,\s]+)\]', input_string)
    if predecessor_match:
        extracted_values['predecessor'] = [int(x) for x in predecessor_match.group(1).split(',')]
        
    loads_match = re.search(r'Loads\(node\):\s+\[([0-9,\s]+)\]', input_string)
    if loads_match:
        extracted_values['loads_int'] = [int(x) for x in loads_match.group(1).split(',')]
        
    couriers_match = re.search(r'Couriers:\s+\[([0-9,\s]+)\]', input_string)
    if couriers_match:
        extracted_values['assigned_courier'] = [int(x) for x in couriers_match.group(1).split(',')]
    
    return extracted_values


def get_tours_and_loads(predecessor, assigned_courier, loads_int, m, n):
    '''
    Extracts the tour taken by each courier as a sequence of distribution points visited, 
    based on their assigned loads and predecessors
    
    '''
    visited_dp = [[] for x in range(1, m+1)]         # Distribution points visited by each courier
    loads_int_courier = [[] for x in range(1, m+1)]  # Intermediate load of each courier
    tours = [[0] for x in range(1, m+1)]
    loads = []
    solution = []   
    
    # Populate visited_dp and loads_int_courier with predecessors and loads for each courier
    for courier in range(m):
        for node in range(len(assigned_courier)):
            if assigned_courier[node] == courier+1 and loads_int[node] != 0 and predecessor[node] < n+1:
                visited_dp[courier].append(predecessor[node]-1)
                loads_int_courier[courier].append(loads_int[node])
    
    # Sort and format tours, and calculate the final load
    for courier in range(m):
        pairs = zip(visited_dp[courier], loads_int_courier[courier])
        pairs = sorted(pairs, key = lambda x: x[1])
        tours[courier] = [p[0]+1 for p in pairs]
        tours[courier].append(n+1)
        tours[courier].insert(0, n+1)
        if loads_int_courier[courier]:
            loads.append(max(loads_int_courier[courier]))
        else:
            loads.append(0)
    
    # Prepare the solution without origin
    for t in tours:
        solution.append(t[1:-1])
    
    return loads, tours, solution

# ------------------------------------------------------------------------ #
#                                  SMT                                     #  
# ------------------------------------------------------------------------ #
    
def extract_tour_smt(tour_matrix):
    '''
    Extract a tour from a tour matrix
    
    '''
    # Extract the indices where travels is equal to 1
    indices = [(i, j) for i in range(len(tour_matrix)) for j in range(len(tour_matrix[i])) if tour_matrix[i][j] == True]
    # Sort the list to start with the tuple having the highest first element (which is the origin)
    indices.sort(key=lambda x: x[0], reverse=True)

    # Rearrange the list so that the next element have as first dp, the second dp of the previous one
    ordered_indices = [indices.pop(0)]
    while indices:
        next_index = next((i for i, x in enumerate(indices) if x[0] == ordered_indices[-1][1]), None)
        if next_index is not None:
            ordered_indices.append(indices.pop(next_index))
    
    # Flatten the indices and remove consecutive duplicates
    flat_indices = [ordered_indices[0][0]]
    for pair in ordered_indices:
        if pair[1] != flat_indices[-1]:  
            flat_indices.append(pair[1])  
        
    # Increment each element by 1
    tour = [x + 1 for x in flat_indices]
    
    return tour
    

def print_debug_smt(total_time, init_time, pruning_time, solving_time, removed_arcs, all_arcs, lower_bound, upper_bound, status):
    '''
    Print additional informations about the execution of the model 
    
    '''
    print("----- DEBUG -----\n")
    print(f"Total Execution Time = {total_time} s\n")
    print(f"\tInitialization Time = {init_time} s\n")
    print(f"\tPruning Time = {pruning_time} s\n")
    print(f"\tSolving Time = {solving_time} s\n")    
    print(f"Removed Arcs = {removed_arcs}/{all_arcs}\n")
    print(f"Bounds [{lower_bound}, {upper_bound}]\n")
    print(f"Status = {status}")
    print()
    
# ------------------------------------------------------------------------ #
#                                  MIP                                     #  
# ------------------------------------------------------------------------ #

def extract_tour_mip(travels, origin, solver="pulp"):
    """
    Extracts tours from a travel dictionary depending on the solver used.
    
    """
    if solver == "pulp":
        active_trips = [key for key, var in travels.items() if pulp.value(var) == 1.0]
    elif solver == "gurobi":
        active_trips = [key for key, val in travels.items() if val > 0.5]
    else:
        raise ValueError("Unsupported solver. Use 'pulp' or 'gurobi'.")

    tours_by_courier = {}
    for courier, i, j in active_trips:
        tours_by_courier.setdefault(courier, []).append((i, j))

    output_tours = []
    solution = []

    for courier in sorted(tours_by_courier):
        tour_edges = tours_by_courier[courier]
        tour = []
        current_candidates = [i for i, j in tour_edges if i == origin]
        if not current_candidates:
            continue
        current = current_candidates[0]
        tour.append(current)

        while True:
            next_leg = next(((i, j) for i, j in tour_edges if i == current), None)
            if next_leg is None:
                break
            _, current = next_leg
            tour.append(current)
            if current == origin:
                break

        output_tours.append([x + 1 for x in tour])    # Shift for human-readable index
        solution.append([x + 1 for x in tour[1:-1]])  # Exclude origin

    return output_tours, solution
    

def print_debug_mip(total_time, init_time, heuristic_time, pruning_time, solving_time,
                thresholds, distance_threshold, removed_arcs, lower_bound, upper_bound, status):
    
    print("----- DEBUG -----\n")
    print(f"Total Execution Time = {total_time} s\n")
    print(f"\tInitialization Time = {init_time} s\n")
    print(f"\tHeuristic Time = {heuristic_time} s\n")
    print(f"\tPruning Time = {pruning_time} s\n")
    print(f"\tSolving Time = {solving_time} s\n")    
    print(f"Thresholds = {thresholds}\n")
    print(f"Distance Threshold = {distance_threshold}\n")
    print(f"Removed Arcs = {removed_arcs}\n")
    print(f"Bounds [{lower_bound}, {upper_bound}]\n")
    print(f"Status = {status}")
    print()