# Modelling and Solving the Multiple Couriers Problem

## Problem Description

The **Multiple Couriers Planning (MCP) problem** is defined as follows:

* We have **m couriers** responsible for distributing **n ≥ m items** to different customer locations.
* Each courier *i* has a maximum load capacity *lᵢ*.
* Each item *j* has:

  * A delivery location *j*.
  * A size *sⱼ* (e.g., weight or volume).
* Couriers must start and end their tour at a common **origin point o**.
* The assignment of items must respect each courier’s load capacity.

**Objective**: Minimize the **maximum distance traveled by any courier**, ensuring a fair division of workload.

---

## Solution Overview

The project provides **three optimization models** for solving MCP:

1. **Constraint Programming (CP)**

   * Uses the **Giant Tour Representation (GTR)** to represent predecessor-successor relationships.
   * Improves scalability compared to a travel-matrix approach, especially on large instances.

2. **Satisfiability Modulo Theories (SMT)**

   * Based on a 3D binary “travel” matrix encoding courier-to-location transitions.

3. **Mixed-Integer Programming (MIP)**

   * Also uses the travel-matrix formulation, similar to classic **Capacitated Vehicle Routing Problem (CVRP)** models.

By unifying CP, SMT, and MIP approaches under a common framework, this repository allows benchmarking and comparing different optimization paradigms.

---

## Installation & Execution

The solution is fully containerized via **Docker**.
From the project root, run the provided PowerShell script:

```powershell
.\run.ps1 -ArgsToPass <arguments>
```

* The `-ArgsToPass` parameter takes the Python script name (`launcher.py`) and additional options.

---

## Arguments

### Common

* `--model {CP, SMT, MIP}` → Selects which model to run. If omitted, runs all sequentially.
* `--inst <int>` → Selects the instance number. If omitted, all instances are tested.

### CP-specific

* `-c, --chuffed` → Use only the **Chuffed** solver (disable Gecode).

  > Note: Gecode may not work on Windows; CP tests were run on Linux.

### SMT-specific

* `-p, --prune` → Enable pruning.
* `--n_sol <int>` → Number of heuristic solutions considered for pruning (default: 10).
* `-l, --load_heuristics` → Load heuristics from file (ensures reproducibility).
* `-d, --debug` → Enable debug output.

### MIP-specific

* `-g, --gurobi` → Enable Gurobi solver (requires valid license).
* `-s, --skip` → Skip instances exceeding the timeout.
* `-l, --load_heuristics` → Load heuristics from file (ensures reproducibility).
* `-d, --debug` → Enable debug output.

---

## Usage Examples

1. **Run all models on all instances** (same configuration as report results):

   ```powershell
   .\run.ps1 -ArgsToPass "launcher.py", "-l", "-p", "-g", "-s"
   ```

2. **Run only the MIP model on instance 1**:

   ```powershell
   .\run.ps1 -ArgsToPass "launcher.py", "-l", "-p", "-g", "-s", "--model", "MIP", "--inst", "1"
   ```

3. **Run the solution checker**:

   ```powershell
   .\run.ps1 -ArgsToPass "check_solution.py", "DAT", "res/"
   ```
