include "globals.mzn";

% ------------------------------------------------------------------- %
%                              VARIABLES                              %
% ------------------------------------------------------------------- %

% --- INSTANCE VARIABLES --- %

int: m;                          % Number of couriers
int: n;                          % Number of items (distribution point)
array[1..m] of int: l;           % Load size limit for each courier
array[1..n] of int: s;           % Size of each item
array[1..n+1, 1..n+1] of int: D; % Distance matrix
int: origin = n + 1;             % Origin location

% To solve the problem we will use the Giant Tour Representation (GTR).
% We will have a total of 1..n+2*m nodes: one node for each distribution point + two nodes for each courier (starts and ends at the origin)
% - The nodes representing the starting point (origin) for each courier will be from n+1 to n+m
% - The nodes representing the ending point (origin) for each courier will be from n+m+1 to n+2*m

% --- DECISION VARIABLES --- %

array[1..n+2*m] of var 1..n+2*m: successor;    % The successor of each node
array[1..n+2*m] of var 1..n+2*m: predecessor;  % The predecessor of each node (redundant)
array[1..n+2*m] of var 1..m: assigned_courier; % Courier assigned to each distribution point
array[1..n+2*m] of var 0..max(l): loads_int;   % Load when arriving at each node
array[1..m] of var int: distances;             % Distance travelled by each courier
var int: objective;                            % Maximum distance travelled by any courier

% --- ADDITIONAL VARIABLES --- %

% D_extended is the GTR adaptation of D
array[1..n+2*m, 1..n+2*m] of int: D_extended = array2d(1..n+2*m, 1..n+2*m,[
  if from_node <= n /\ to_node <= n then 
    D[from_node, to_node]                      % distribution point-distribution point
  elseif from_node <= n /\ to_node > n then   
    D[from_node, n+1]                          % distribution point-origin
  elseif to_node <= n /\ from_node > n then 
    D[n+1, to_node]                            % origin-distribution point
  else 
    D[n+1, n+1]                                % origin-origin
  endif
   | from_node, to_node in 1..n+2*m]); 

% s_extended is the GTR adaptation of s with s_extended = 0 for the nodes representing the origin
array[1..n+2*m] of int: s_extended = [
  if node <= n then 
    s[node]
  else 
    0
  endif
| node in 1..n+2*m]; 

% --- LOWER AND UPPER BOUNDS --- %

% The maximum distance travelled by any courier is at least a round trip from the origin to the furthest distribution point
function int: calculate_lower_bound(array[int, int] of int: D, int: n) =
    let {
        array[int] of int: last_column = [D[i, n+1] | i in 1..n+1],
        array[int] of int: last_row = [D[n+1, i] | i in 1..n+1]
    } in
        min(2*max(last_column), 2*max(last_row));

% A solution for the upper bound can be obtained using average and variance
function int: calculate_upper_bound(array[int, int] of int: D, int: n, int: m) =
    let {
        float: avg_distance = (sum(i, j in 1..n+1) (D[i, j]))/((n+1)*(n+1)),
        float: variance = (sum(i, j in 1..n+1) ((D[i, j] - avg_distance) ^ 2))/((n+1)*(n+1)),
        float: std_dev = sqrt(variance)
    } in
        ceil(avg_distance + 2*std_dev)*ceil(n/m);

% Actually compute the upper and lower bounds
int: lower_bound = calculate_lower_bound(D, n);
int: upper_bound = calculate_upper_bound(D, n, m);

% ------------------------------------------------------------------- %
%                             CONSTRAINTS                             %
% ------------------------------------------------------------------- %

% Successors of end nodes are start nodes
constraint 
   forall(node in (n+m+1..n+2*m-1)) (
     successor[node] = node-m+1 
   );
constraint
   successor[n+2*m] = n+1;

% Predecessor of start nodes are end nodes
constraint
   forall(node in (n+2..n+m)) (
     predecessor[node] = node + m-1
   );
constraint
   predecessor[n+1] = n+2*m;

% Associate a courier to each start nodes
constraint 
   forall(node in n+1..n+m) (
     assigned_courier[node] = node-n
   ); 
% Associate a courier to each end nodes
constraint 
   forall(node in n+m+1..n+2*m) (
     assigned_courier[node] = node-n-m
   );

% Ensure consistency between successor and predecessor variables
constraint
   forall(node in 1..n+2*m) (
      successor[predecessor[node]] = node
   );
constraint
   forall(node in 1..n+2*m) (
      predecessor[successor[node]] = node
   );

% Ensure the courier of a node is the same as its predecessor and successor
constraint
   forall(node in 1..n) (
      assigned_courier[predecessor[node]] = assigned_courier[node]
   );
constraint 
   forall(node in 1..n) (
      assigned_courier[successor[node]] = assigned_courier[node]
   );

% Initial load at start nodes is 0
constraint 
   forall(node in n+1..n+m) (
     loads_int[node] = 0
   );

% Final load at end nodes is more than 0 (ensures that all couriers are utilized)
constraint 
   forall(node in (n+m+1..n+2*m)) (
     loads_int[node] > 0
   );

% Maintain load consistency when moving between nodes
constraint 
   forall(node in 1..n) (
      loads_int[successor[node]] = loads_int[node] + s_extended[node]
   );
constraint
   forall(node in n+1..n+m) (
      loads_int[node] = loads_int[successor[node]] 
   );

% Ensure load does not exceed the courier's capacity at any node
constraint
    forall(node in 1..n)(
        loads_int[node] <= l[assigned_courier[node]]
    );
constraint 
    forall(c in 1..m)(
        loads_int[c+n+m] <= l[c]
    );

% Prevent subtours using circuit constraints on successor and predecessor
constraint 
   circuit(successor);
constraint
   circuit(predecessor);

% Calculate the distance travelled by each courier
constraint 
    forall(c in 1..m)(
        distances[c] = sum([D_extended[node, successor[node]] | node in 1..n+2*m where assigned_courier[node] == c])
    );

% --- SYMMETRY BREAKING --- %

% Checks if the variable a is equal to the integer b
function var int: equality(var int: a, int: b) = 
  if a = b then 
    1 
  else 
    0 
  endif;

% Enforce an ordering among the couriers with the same load
constraint symmetry_breaking_constraint(
  forall(courier_1, courier_2 in 1..m)(
    (l[courier_1] == l[courier_2] /\ courier_1 > courier_2) -> 
    lex_less([equality(assigned_courier[node_1], courier_1) | node_1 in 1..n], 
             [equality(assigned_courier[node_2], courier_2) | node_2 in 1..n])
  )
);


% ------------------------------------------------------------------- %
%                              OBJECTIVE                              %
% ------------------------------------------------------------------- %

% Minimize the maximum distance travelled by any courier
constraint
  objective = max(distances);

% Ensure that the objective is within the bounds
constraint 
  objective >= lower_bound /\
  objective <= upper_bound;

% --- RANDOMIZATION --- %
% Enable randomized search
solve :: seq_search([
    int_search(successor, first_fail, indomain_random, complete),
    int_search(assigned_courier, first_fail, indomain_random, complete),
    int_search(distances, input_order, indomain_min, complete)
]) minimize objective;

% ------------------------------------------------------------------- %
%                                OUTPUT                               %
% ------------------------------------------------------------------- %

output [
  "RESULTS\n--------------------------\n",
  "Objective:   ", show(objective), "\n",
  "Bounds:      [", show(lower_bound), ", ", show(upper_bound), "]\n",
  "Distances:   ", show(distances), "\n",
  "Successor:   ", show(successor), "\n",
  "Predecessor: ", show(predecessor), "\n",
  "Loads(node): ", show(loads_int), "\n",
  "Couriers:    ", show(assigned_courier)
];