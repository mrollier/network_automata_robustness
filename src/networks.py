import igraph as ig
import random

def create_2d_torus_lattice(L, degree=8):
    """Create a 2D torus lattice with degree 8 or 4."""
    g = ig.Graph()
    n = L * L
    g.add_vertices(n)
    
    edges = []
    for y in range(L):
        for x in range(L):
            i = _node_index(x, y, L)
            if degree == 8:
                # 8 neighbors (Moore neighborhood: cardinal + diagonal)
                neighbors = [
                    (x+1, y), (x-1, y), (x, y+1), (x, y-1),
                    (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)
                ]
            elif degree == 4:
                # 4 neighbors (von Neumann neighborhood: cardinal only)
                neighbors = [
                    (x+1, y), (x-1, y), (x, y+1), (x, y-1)
                ]
            else:
                raise ValueError("Degree must be either 4 (von Neumann) or 8 (Moore).")
            for nx, ny in neighbors:
                j = _node_index(nx, ny, L)
                if i < j:  # avoid double-adding edges
                    edges.append((i, j))
    g.add_edges(edges)
    return g

def watts_strogatz_rewire(g, p):
    """Rewire edges in a copy of graph g with probability p (Watts-Strogatz style)."""
    if not g.is_connected():
        raise ValueError("Input graph must be connected.")
    try_counter = 0
    max_counter = 10
    while True:
        try_counter += 1
        g_copy = g.copy()  # Make a copy of the original graph
        n = len(g_copy.vs)  # Number of vertices in the graph
        edges_to_rewire = list(g_copy.get_edgelist())  # Get the list of all edges
        for edge in edges_to_rewire:
            if random.random() < p:  # With probability p, rewire the edge
                source, target = edge
                g_copy.delete_edges([edge])  # Remove the current edge
                
                # Avoid self-loops and existing edges
                possible_targets = set(range(n)) - {source} - set(g_copy.neighbors(source))
                if possible_targets:
                    new_target = random.choice(list(possible_targets))  # Choose a new target randomly
                    g_copy.add_edges([(source, new_target)])  # Add the new edge
                else:
                    raise ValueError(f"No possible target for moving the edge from node {source}.")
        if g_copy.is_connected():
            return g_copy
        if try_counter == max_counter:
            raise ValueError(f"It was not possible to construct a connected Watts-Strogatz graph.")

#%% helper functions

def _node_index(x, y, L):
    """Convert 2D coordinates to node index in 1D."""
    return (y % L) * L + (x % L)