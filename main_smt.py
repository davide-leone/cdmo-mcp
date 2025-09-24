import argparse
import numpy as np

from modules.utils import read_dat_file, create_json
from modules.models import create_and_test_model_smt
    

def main(enable_pruning_arg, load_heuristics_arg, n_sol_arg, enable_debug_arg, inst_arg=None):
    if inst_arg is not None:
        instances = [inst_arg]
    else:
        instances = range(1, 22)
    
    for instance_number in instances:
        print("----------------------------------------------------------------------------")
        print("\n                                  INSTANCE {}\n".format(instance_number))
        filename = "DAT/inst{:02d}.dat".format(instance_number)
        m, n, l, s, D = read_dat_file(filename)

        approaches = []
        times = []
        optimalities = []
        objectives = []
        solutions = []
        
        # Basic approach
        elapsed_time, optimal, objective, solution = create_and_test_model_smt(m, n, l, s, D, enable_debug=enable_debug_arg)

        approaches.append("basic")
        times.append(elapsed_time)
        optimalities.append(optimal)
        objectives.append(objective)
        solutions.append(solution)
        
        # Heuristic + Pruning approach
        if enable_pruning_arg:
            elapsed_time, optimal, objective, solution = create_and_test_model_smt(
                m, n, l, s, D,
                instance_number=instance_number,
                enable_pruning=enable_pruning_arg,
                load_heuristics=load_heuristics_arg,
                n_sol=n_sol_arg,
                enable_debug=enable_debug_arg
            )

        approaches.append("heuristic+pruning")
        times.append(elapsed_time)
        optimalities.append(optimal)
        objectives.append(objective)
        solutions.append(solution)

        create_json('res/SMT', instance_number, approaches, times, optimalities, objectives, solutions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the Capacitated Multiple Courier Problem using SMT.")
    parser.add_argument('-p', '--prune', action='store_true', help='Enable pruning (default: disabled)')
    parser.add_argument('-l', '--load_heuristics', action='store_true', help='Load heuristic results (default: disabled)')
    parser.add_argument('--n_sol', type=int, default=10, help='Number of heuristic solutions (default: 10)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output (default: disabled)')
    parser.add_argument('--inst', type=int, help='Instance number to solve (optional, test all if not provided)')

    args = parser.parse_args()

    main(
        enable_pruning_arg=args.prune,
        load_heuristics_arg=args.load_heuristics,
        n_sol_arg=args.n_sol,
        enable_debug_arg=args.debug,
        inst_arg=args.inst
    )
