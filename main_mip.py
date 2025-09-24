import argparse
import numpy as np
    
from modules.utils import read_dat_file, create_json
from modules.models import create_and_test_model_pulp, create_and_test_model_gurobi

def main(enable_gurobi_arg, load_heuristics_arg, enable_debug_arg, skip_arg, inst_arg=None):
    if inst_arg is not None:
        instances = [inst_arg]
    else:
        instances = range(1, 22)
    instances_to_skip = range(11, 22)
    
    for instance_number in instances:
        print("----------------------------------------------------------------------------")
        print("\n                                  INSTANCE {}\n".format(instance_number))
        # Construct the filename
        filename = "DAT/inst{:02d}.dat".format(instance_number)
        # Read the instance from a file
        m, n, l, s, D = read_dat_file(filename)
        
        # json arguments
        approaches = []
        times = []
        optimalities = []
        objectives = []
        solutions = []
        
        if skip_arg and (instance_number in instances_to_skip):
            
            print_result("Pulp-CBC+Basic", 300, False, np.inf, [])
            # Append json arguments
            approaches.append("Pulp-CBC+Basic")
            times.append(300)
            optimalities.append(False)
            objectives.append(np.inf)
            solutions.append([])
            
            print_result("Pulp-CBC+Pruning", 300, False, np.inf, [])
            approaches.append("Pulp-CBC+Pruning")
            times.append(300)
            optimalities.append(False)
            objectives.append(np.inf)
            solutions.append([])
        else:        
            # MODEL 1.1
            elapsed_time, optimal, objective, solution = create_and_test_model_pulp(
                                                                                    m, n, l, s, D, 
                                                                                    instance_number=instance_number, 
                                                                                    load_heuristics=False, 
                                                                                    enable_pruning=False, 
                                                                                    enable_debug=enable_debug_arg)
            
            # Append json arguments
            approaches.append("Pulp-CBC+Basic")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)

            
            # MODEL 1.2
            elapsed_time, optimal, objective, solution = create_and_test_model_pulp(
                                                                                    m, n, l, s, D,
                                                                                    instance_number=instance_number,
                                                                                    load_heuristics=load_heuristics_arg,
                                                                                    enable_pruning=True,
                                                                                    enable_debug=enable_debug_arg)
            
            # Append json arguments
            approaches.append("Pulp-CBC+Pruning")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)
            
        
        if enable_gurobi_arg:
            # MODEL 2.1
            elapsed_time, optimal, objective, solution = create_and_test_model_gurobi(
                                                                                      m, n, l, s, D, 
                                                                                      instance_number=instance_number, 
                                                                                      load_heuristics=False, 
                                                                                      enable_warm_start=False, 
                                                                                      enable_pruning=False, 
                                                                                      enable_debug=enable_debug_arg)
            
            # Append json arguments
            approaches.append("Gurobi+Basic")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)
            
            # MODEL 2.2
            elapsed_time, optimal, objective, solution = create_and_test_model_gurobi(
                                                                                      m, n, l, s, D, 
                                                                                      instance_number=instance_number, 
                                                                                      load_heuristics=load_heuristics_arg, 
                                                                                      enable_warm_start=True, 
                                                                                      enable_pruning=False, 
                                                                                      enable_debug=enable_debug_arg)
            
            # Append json arguments
            approaches.append("Gurobi+Warm_Start")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)
            
            # MODEL 2.3
            elapsed_time, optimal, objective, solution = create_and_test_model_gurobi(
                                                                                      m, n, l, s, D, 
                                                                                      instance_number=instance_number, 
                                                                                      load_heuristics=load_heuristics_arg, 
                                                                                      enable_warm_start=False, 
                                                                                      enable_pruning=True, 
                                                                                      enable_debug=enable_debug_arg)
            
            # Append json arguments
            approaches.append("Gurobi+Pruning")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)
            
            # MODEL 2.4
            elapsed_time, optimal, objective, solution = create_and_test_model_gurobi(
                                                                                      m, n, l, s, D, 
                                                                                      instance_number=instance_number, 
                                                                                      load_heuristics=load_heuristics_arg, 
                                                                                      enable_warm_start=True, 
                                                                                      enable_pruning=True, 
                                                                                      enable_debug=enable_debug_arg)
            
            
            # Append json arguments
            approaches.append("Gurobi+Warm_Start+Pruning")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)
        
        # Create json file
        create_json('res/MIP', instance_number, approaches, times, optimalities, objectives, solutions)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the Capacitated Multiple Courier Problem using MIP.")
    parser.add_argument('-g', '--gurobi', action='store_true', help='Enable Gurobi solver (default: disabled)')
    parser.add_argument('-l', '--load_heuristics', action='store_true', help='Load heuristic results (default: disabled)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output (default: disabled)')
    parser.add_argument('-s', '--skip', action='store_true', help='Skip time-out instances (default: disabled)')
    parser.add_argument('--inst', type=int, help='Instance number to solve (optional, test all if not provided)')

    args = parser.parse_args()

    main(
        enable_gurobi_arg=args.gurobi,
        load_heuristics_arg=args.load_heuristics,
        enable_debug_arg=args.debug,
        skip_arg=args.skip,
        inst_arg=args.inst
    )
