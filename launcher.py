import subprocess
import argparse

def run_script(script, args):
    cmd = ["python", script] + args
    print("----------------------------------------------------------------------------")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Launcher for CP, SMT, and MIP pipelines")

    parser.add_argument('--model', choices=['CP', 'SMT', 'MIP'], help='Specify which model to run. If omitted, all models will run.')
    parser.add_argument('--inst', type=int, help='Specify the instance number to test')

    # CP options
    parser.add_argument('-c', '--chuffed', action='store_true', help='Use only Chuffed and not Gecode for CP')

    # SMT options
    parser.add_argument('-p', '--prune', action='store_true', help='Enable pruning for SMT')
    parser.add_argument('--n_sol', type=int, help='Number of heuristic solutions (SMT)')

    # MIP options
    parser.add_argument('-g', '--gurobi', action='store_true', help='Use also Gurobi for MIP')
    parser.add_argument('-s', '--skip', action='store_true', help='Skip time-out instances for MIP')

    # SMT/MIP options
    parser.add_argument('-l', '--load_heuristics', action='store_true', help='Load heuristics for SMT and MIP')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output for SMT and MIP')

    args = parser.parse_args()

    # Add --inst to all
    inst_arg = ['--inst', str(args.inst)] if args.inst is not None else []

    if args.model in [None, 'CP']:
        cp_args = inst_arg.copy()
        if args.chuffed:
            cp_args.append('-c')
        run_script("main_cp.py", cp_args)

    if args.model in [None, 'SMT']:
        smt_args = inst_arg.copy()
        if args.prune:
            smt_args.append('-p')
        if args.load_heuristics:
            smt_args.append('-l')
        if args.n_sol is not None:
            smt_args += ['--n_sol', str(args.n_sol)]
        if args.debug:
            smt_args.append('-d')
        run_script("main_smt.py", smt_args)

    if args.model in [None, 'MIP']:
        mip_args = inst_arg.copy()
        if args.gurobi:
            mip_args.append('-g')
        if args.load_heuristics:
            mip_args.append('-l')
        if args.debug:
            mip_args.append('-d')
        if args.skip:
            mip_args.append('-s')
        run_script("main_mip.py", mip_args)

if __name__ == "__main__":
    main()
