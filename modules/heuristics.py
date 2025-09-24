import math
import networkx as nx
import numpy as np
import random
import statistics

from collections import defaultdict

# ------------------------------------------------------------------------ #
#                                SMT+MIP                                   #  
# ------------------------------------------------------------------------ #

def compute_lower_bound(D, n):
    '''
    Compute the lower bound as the longest round trips from the origin to any distribution point
    
    '''
    origin = n
    round_trips = [D[origin][dp] + D[dp][origin] for dp in range(n)]
    
    lower_bound = max(round_trips)
    
    return lower_bound


def compute_upper_bound(D, n, m):
    '''
    Compute the upper bound using a combination of mean and standard deviation
    
    '''
    distances = [D[from_dp][to_dp] for from_dp in range(n+1) for to_dp in range(n+1)]
    avg_distance = statistics.mean(distances)
    std_dev = statistics.stdev(distances)
    
    upper_bound = math.ceil(avg_distance + 2*std_dev) * math.ceil(n/m)

    return upper_bound


def generate_heuristics(D, m, n, s, l, n_sol=10):
    '''
    Tries to generate good heuristic solutions to use for pruning
    '''
    
    def cluster_distribution_points(D, m, n, s, l, origin):
        '''
        Clusters distribution points considering distance from origin and courier capacity constraints.
        '''
        D_array = np.array(D)
        d_origin = D_array[origin, :n]
        
        
        all_indices = list(range(n))

        def attempt_assignment(order):
            '''
            Attempt to assign distribution points to clusters
            '''
            clusters = [[] for _ in range(m)]
            loads = [0] * m
            assigned = [False] * n
            for idx in order:
                item_size = s[idx]
                best_cluster = None
                best_cost = float('inf')
                for cluster_idx in range(m):
                    if loads[cluster_idx] + item_size <= l[cluster_idx]:
                        internal_cost = sum(D_array[idx, p] for p in clusters[cluster_idx]) if clusters[cluster_idx] else 0
                        cost = d_origin[idx] + internal_cost
                        if cost < best_cost:
                            best_cost = cost
                            best_cluster = cluster_idx
                if best_cluster is not None:
                    clusters[best_cluster].append(idx)
                    loads[best_cluster] += item_size
                    assigned[idx] = True
            return clusters, loads, assigned

        success = False
        max_iter = 25
        for _ in range(max_iter):
            shuffled = all_indices.copy()
            random.shuffle(shuffled)
            clusters, loads, assigned = attempt_assignment(shuffled)
            # Ensure no empty clusters
            for i in range(m):
                if len(clusters[i]) == 0:
                    unassigned = [idx for idx, a in enumerate(assigned) if not a]
                    if unassigned:
                        idx = unassigned[0]
                    else:
                        largest = np.argmax([len(c) for c in clusters])
                        idx = clusters[largest].pop()
                        loads[largest] -= s[idx]
                    clusters[i].append(idx)
                    loads[i] += s[idx]
                    assigned[idx] = True

            if all(assigned):
                success = True
                break
        
        # Fallback assignment if clustering fails
        if not success:
            unassigned = [idx for idx, a in enumerate(assigned) if not a]
            for idx in unassigned:
                best_cluster = np.argmin([D[origin][idx]] * m)
                clusters[best_cluster].append(idx)
                loads[best_cluster] += s[idx]

        return clusters

    def find_heuristic_trips(clusters, D, origin):
        '''
        Convert clusters into heuristic delivery trips
        '''
        trips = []
        for courier, dps in enumerate(clusters):
            if not dps:
                continue
            visited = set()
            current = origin
            while len(visited) < len(dps):
                next_dp = min((dp for dp in dps if dp not in visited), key=lambda dp: D[current][dp])
                trips.append((courier, int(current), int(next_dp)))
                visited.add(next_dp)
                current = next_dp
            trips.append((courier, int(current), origin))
        return trips

    def compute_tour_distance(tour, D):
        '''
        Compute the distance of a tour
        '''
        if len(tour) < 2:
            return 0
        return sum(D[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

    def get_tour(trip_matrix, start, n):
        '''
        Extract a tour from a trip matrix
        '''
        tour = [start]
        current = start
        while True:
            next_node = None
            for j in range(n + 1):
                if trip_matrix[current][j]:
                    next_node = j
                    trip_matrix[current][j] = False
                    break
            if next_node is None or next_node == start:
                break
            tour.append(next_node)
            current = next_node
        tour.append(start)
        return tour

    def is_feasible(heuristic_trips, sizes, loads, origin, n):
        '''
        Check if a heuristic solution is feasible
        '''
        courier_routes = defaultdict(list)
        for courier, from_node, to_node in heuristic_trips:
            courier_routes[courier].append((from_node, to_node))

        visited_points = set()
        for courier in range(m):
            trips = courier_routes[courier]
            if not trips:
                continue

            graph = defaultdict(list)
            in_degree = defaultdict(int)
            out_degree = defaultdict(int)

            for from_node, to_node in trips:
                graph[from_node].append(to_node)
                out_degree[from_node] += 1
                in_degree[to_node] += 1

            if out_degree[origin] != 1 or in_degree[origin] != 1:
                return False

            visited_edges = set()
            current = origin
            for _ in range(len(trips)):
                if current not in graph or not graph[current]:
                    return False
                next_node = graph[current].pop()
                visited_edges.add((current, next_node))
                current = next_node

            if current != origin or len(visited_edges) != len(trips):
                return False

            demand = 0
            for _, to_node in visited_edges:
                if to_node != origin:
                    if to_node in visited_points:
                        return False
                    visited_points.add(to_node)
                    demand += sizes[to_node]
            if demand > loads[courier]:
                return False
        return True

    # Main loop to generate heuristic solutions
    origin = n
    heuristic_solutions = []  # Store (objective, arc_set) pairs
    attempts=1000
    for _ in range(attempts):
        clusters = cluster_distribution_points(D, m, n, s, l, origin)
        heuristic_trips = find_heuristic_trips(clusters, D, origin)

        if is_feasible(heuristic_trips, s, l, origin, n):
            courier_tours = [[] for _ in range(m)]
            for courier, from_dp, to_dp in heuristic_trips:
                courier_tours[courier].append((from_dp, to_dp))

            tour_paths = []
            for ctour in courier_tours:
                trip_matrix = [[False] * (n + 1) for _ in range(n + 1)]
                for from_dp, to_dp in ctour:
                    trip_matrix[from_dp][to_dp] = True
                tour_paths.append(get_tour(trip_matrix, origin, n))

            distances = [compute_tour_distance(tour, D) for tour in tour_paths]
            max_distance = max(distances) if distances else float('inf')
            arcs = set(trip for trip in heuristic_trips)

            heuristic_solutions.append((max_distance, arcs))

    # Select top-n_sol heuristics
    heuristic_solutions.sort(key=lambda x: x[0])
    best_solutions = heuristic_solutions[:n_sol]
    heuristic_upper_bound = best_solutions[0][0] if best_solutions else float('inf')

    arc_set = set()
    for _, arcs in best_solutions:
        arc_set.update(arcs)

    return heuristic_upper_bound, arc_set
    
# ------------------------------------------------------------------------ #
#                                  MIP                                     #  
# ------------------------------------------------------------------------ #

def compute_distance_threshold(m, n, D):
    '''
    Compute the distance threshold used for pruning
    '''
    flat_distances = [D[from_dp][to_dp] for from_dp in range(len(D)) for to_dp in range(len(D)) if from_dp != to_dp]

    avg_distance = np.mean([D[from_dp][to_dp] for from_dp in range(n) for to_dp in range(n) if from_dp != to_dp])
    std_distance = np.std([D[from_dp][to_dp] for from_dp in range(n) for to_dp in range(n) if from_dp != to_dp])
    expected_stops_per_courier = n / m
    
    # Minimum Spanning Tree (MST)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=D[i][j])

    mst_cost = sum(w["weight"] for u, v, w in nx.minimum_spanning_edges(G, data=True))
    avg_mst_edge = mst_cost / (n - 1)
    
    thresholds = [
        round(np.percentile(flat_distances, 85)),
        round(avg_distance + std_distance * math.log(expected_stops_per_courier + 1)),
        round(avg_mst_edge * 2.5)
    ]

    return thresholds, int(np.mean(thresholds))