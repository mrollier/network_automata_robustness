# %% packages

# specialised
import numpy as np
import pandas as pd
import igraph as ig

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


# %% generate list of networks
np.random.gauss = np.random.normal # for compatibility with igraph
ig.set_random_number_generator(np.random)

for i, row in df_params.iterrows():
	np.random.seed(row['seed'])
	if row['model'] == 'BA':
		graph = ig.Graph.Barabasi(n=row['n'], m=row['m'], directed=False, power=1) # linear BA model
	else:
		raise
	qt_nodes = len(graph.vs)
	qt_edges = len(graph.es)
	
	# the file will have "1 + INITS_PER_GRAPH + qt_edges" lines
	# the first line informs the number of nodes, edges and initial configurations, respectively
	# the next "INITS_PER_GRAPH" lines are 0-1 lists denoting "qt_nodes" initial states
	# the next "qt_edges" lines are pairs describing the node ids for each edges
	# all values are separated by single space
	with open(f'input/network_{i:04d}.txt', 'w') as f:
		f.write( f'{qt_nodes} {qt_edges} {INITS_PER_GRAPH}\n' )
		for j in range(INITS_PER_GRAPH):
			h0 = np.random.randint(0, 2, qt_nodes, int)
			f.write( ' '.join(map(str, h0))  + '\n' )
		edge_list = [ e.tuple for e in graph.es ]
		f.write( '\n'.join(map(lambda e: f'{e[0]} {e[1]}', edge_list)) )

# %% end
