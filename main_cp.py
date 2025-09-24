import numpy as np
import argparse

from modules.utils import read_dat_file, create_json
from modules.models import create_and_test_model_smt_cp

def main(disable_gecode_arg, inst_arg=None):
    if inst_arg is not None:
        instances = [inst_arg]
    else:
        instances = range(1, 22)
    
    for instance_number in instances:
        print("----------------------------------------------------------------------------")
        print("\n                                  INSTANCE {}\n".format(instance_number))
        # Construct the filename
        filename = "DAT/inst{:02d}.dat".format(instance_number)
        # Read the instance from a file
        m, n, l, s, D = read_dat_file(filename)
        
        # Keep track of the best result
        min_objective = np.inf
        
        # json arguments
        approaches = []
        times = []
        optimalities = []
        objectives = []
        solutions = []
        
        # I had an error (with asyncio) I was not able to solve on Windows, so I added the possibility to disable gecode.
        # For this reason, I tested the CP model on Linux, where it worked without problems.
        if not disable_gecode_arg:
            # APPROACH 1.1: "gecode"
            elapsed_time, optimal, objective, solution = create_and_test_model_smt_cp("gecode", "minizinc/mcp_base.mzn", "gecode", m, n, l, s, D, min_objective)
            if objective <= min_objective:
                min_objective = objective
            
            # Append json arguments
            approaches.append("gecode")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)

            # APPROACH 1.2: "gecode_symbreak"
            elapsed_time, optimal, objective, solution = create_and_test_model_smt_cp("gecode", "minizinc/mcp_base_sym.mzn", "gecode_symbreak", m, n, l, s, D, min_objective)
            if objective <= min_objective:
                min_objective = objective
           
            # Append json arguments
            approaches.append("gecode_symbreak")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)
            
            # APPROACH 1.3: "gecode_random"
            elapsed_time, optimal, objective, solution = create_and_test_model_smt_cp("gecode", "minizinc/mcp_rand.mzn", "gecode_random", m, n, l, s, D, min_objective)
            if objective <= min_objective:
                min_objective = objective
           
            # Append json arguments
            approaches.append("gecode_random")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution) 
            
            # APPROACH 1.4: "gecode_random_symbreak"
            elapsed_time, optimal, objective, solution = create_and_test_model_smt_cp("gecode", "minizinc/mcp_rand_sym.mzn", "gecode_random_symbreak", m, n, l, s, D, min_objective)
            if objective <= min_objective:
                min_objective = objective
            
            # Append json arguments
            approaches.append("gecode_random_symbreak")
            times.append(elapsed_time)
            optimalities.append(optimal)
            objectives.append(objective)
            solutions.append(solution)
       
        
        # APPROACH 2.1: "chuffed"
        elapsed_time, optimal, objective, solution = create_and_test_model_smt_cp("chuffed", "minizinc/mcp_base.mzn", "chuffed", m, n, l, s, D, min_objective)
        if objective <= min_objective:
            min_objective = objective
        
        # Append json arguments
        approaches.append("chuffed")
        times.append(elapsed_time)
        optimalities.append(optimal)
        objectives.append(objective)
        solutions.append(solution)
        
        # APPROACH 2.2: "chuffed_symbreak"
        elapsed_time, optimal, objective, solution = create_and_test_model_smt_cp("chuffed", "minizinc/mcp_base_sym.mzn", "chuffed_symbreak", m, n, l, s, D, min_objective)
        if objective <= min_objective:
            min_objective = objective
        
        # Append json arguments
        approaches.append("chuffed_symbreak")
        times.append(elapsed_time)
        optimalities.append(optimal)
        objectives.append(objective)
        solutions.append(solution)
          
        
        # Create json file
        create_json('res/CP', instance_number, approaches, times, optimalities, objectives, solutions)
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the Capacitated Multiple Courier Problem using CP.")
    parser.add_argument('-c', '--chuffed', action='store_true', help='Disable Gecode solver (default: enabled)')
    parser.add_argument('--inst', type=int, help='Instance number to solve (optional, test all if not provided)')

    args = parser.parse_args()

    main(disable_gecode_arg=args.chuffed, inst_arg=args.inst)

