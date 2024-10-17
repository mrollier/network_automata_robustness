# %% packages

# specialised
import os
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm

# %% generate list of parameters
GLOBAL_SEED = None
GRAPHS_PER_MODEL = 100
NODES_PER_GRAPH = 100
INITS_PER_GRAPH = 5

np.random.seed(GLOBAL_SEED)
df_params = pd.DataFrame({
	'model': 'BA',
	'n': NODES_PER_GRAPH,
	'm': 4,
	'seed': np.random.randint(low=0, high=int(1<<31), size=GRAPHS_PER_MODEL)
})
df_params.index.name = 'id'
df_params.to_csv('parameters.csv')


# %% create output folder if not exists yet
os.makedirs('graphs/', exist_ok=True)

# %% generate list of networks
np.random.gauss = np.random.normal # for compatibility with igraph
ig.set_random_number_generator(np.random)

for i, row in tqdm(df_params.iterrows()):
	np.random.seed(row['seed'])
	if row['model'] == 'BA':
		graph = ig.Graph.Barabasi(n=row['n'], m=row['m'], directed=False, power=1) # linear BA model
	else:
		raise
	# force the graph to be directed so both edges e_ij and e_ji will be writen in the file
	# it will need more storage but will avoid problems when reading the graph later
	graph = graph.as_directed()
	qt_nodes = len(graph.vs)
	qt_edges = len(graph.es)
	
	# the file will have "1 + qt_edges + INITS_PER_GRAPH" lines
	# the first line informs the number of nodes, edges and initial configurations, respectively
	# the next "qt_edges" lines are pairs describing the node ids for each edges
	# the last "INITS_PER_GRAPH" lines are 0-1 lists denoting "qt_nodes" initial states
	# all values are separated by single space
	with open(f'graphs/network_{i:04d}.txt', 'w') as f:
		f.write( f'{qt_nodes} {qt_edges} {INITS_PER_GRAPH}\n' )
		edge_list = [ e.tuple for e in graph.es ]
		f.write( '\n'.join(map(lambda e: f'{e[0]} {e[1]}', edge_list)) )
		for j in range(INITS_PER_GRAPH):
			h0 = np.random.randint(0, 2, qt_nodes, int)
			f.write( '\n' + ' '.join(map(str, h0)) )

# %% end
