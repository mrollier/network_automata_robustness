# %% packages

# specialised
import os
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm

# extra module
import sys
sys.path.insert(0, 'src/network_analysis/lib') 
from containers import GraphAPI
from analytics import DataPool
from characterization import Distances, Connectivity, ClusteringAndCycles, Centrality

# %% generate list of parameters
GLOBAL_SEED = None
GRAPH_PARAMS = {
    'BA': [
        {'n': 100, 'm': 2},
        {'n': 100, 'm': 3},
        {'n': 100, 'm': 4},
    ]
}
SAMPLING = 50
INITS_PER_GRAPH = 5

# %% configure random number generator
np.random.seed(GLOBAL_SEED)
np.random.gauss = np.random.normal # for compatibility with igraph
ig.set_random_number_generator(np.random)

# %% generate networks and metadata
nets = []
for model in GRAPH_PARAMS:
    for params in GRAPH_PARAMS[model]:
        graphs = []
        seeds = np.random.randint(low=0, high=int(1<<31), size=SAMPLING)
        for num in seeds:
            np.random.seed(num)
            graphs.append(GraphAPI.generate('ig', model.lower(), **params))
        nets.append(pd.DataFrame({
            'collection': 'robustness',
            'dataset': 'toy', 
            'label': model,
            'N': [ G.count_nodes() for G in graphs ],
            'M': [ G.count_edges() for G in graphs ], 
            'K': np.nan,
            'graph': graphs, 
            'abbr': [ f'{model}({ repr(params) })_{num}' for num in seeds ]
        }))

nets = pd.concat(nets, ignore_index=True)
nets['source'] = list(map(lambda num: f'network_{num:06d}.txt', nets.index))

# %% evaluate the networks
datapool = DataPool()
datapool.persist_fetched = False
datapool.update(n=nets)
datapool.update(m=Distances.as_dataframe())
datapool.update(m=Connectivity.as_dataframe())
datapool.update(m=ClusteringAndCycles.as_dataframe())
datapool.update(m=Centrality.as_dataframe())
datapool.update(r=datapool.evaluate(log_info=True))
datapool.save('data/info', n=True, m=True, r=True)

# %% create output folder if not exists yet
os.makedirs('data/graphs', exist_ok=True)

for i, row in tqdm(datapool.networks.iterrows(), total=len(datapool.networks), ncols=80):
    # force the graph to be directed so both edges e_ij and e_ji will be writen in the file
    # it will need more storage but will avoid problems when reading the graph later
    graph = row['graph'].data.as_directed()
    qt_nodes = len(graph.vs)
    qt_edges = len(graph.es)

    # the file will have "1 + qt_edges + INITS_PER_GRAPH" lines
    # the first line informs the number of nodes, edges and initial configurations, respectively
    # the next "qt_edges" lines are pairs describing the node ids for each edges
    # the last "INITS_PER_GRAPH" lines are 0-1 lists denoting "qt_nodes" initial states
    # all values are separated by single space
    filename = row['source']
    with open(f'data/graphs/{filename}', 'w') as f:
        f.write( f'{qt_nodes} {qt_edges} {INITS_PER_GRAPH}\n' )
        edge_list = [ e.tuple for e in graph.es ]
        f.write( '\n'.join(map(lambda e: f'{e[0]} {e[1]}', edge_list)) )
        for j in range(INITS_PER_GRAPH):
            h0 = np.random.randint(0, 2, qt_nodes, int)
            f.write( '\n' + ' '.join(map(str, h0)) )


# %% end
