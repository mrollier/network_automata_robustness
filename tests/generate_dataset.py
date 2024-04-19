# %% packages

# specialised
import numpy as np
import pandas as pd
import igraph as ig

# %% generate list of parameters
GLOBAL_SEED = None
QT_SAMPLES = 100

np.random.seed(GLOBAL_SEED)
df_params = pd.DataFrame({
	'model': 'BA',
	'n': 100,
	'm': 4,
	'seed': np.random.randint(low=0, high=int(1<<31), size=QT_SAMPLES)
})
df_params.index.name = 'id'
df_params.to_csv('../data/generator_params.csv')


# %% generate list of networks
np.random.gauss = np.random.normal # for compatibility with igraph
ig.set_random_number_generator(np.random)

for i, row in df_params.iterrows():
	np.random.seed(row['seed'])
	if row['model'] == 'BA':
		graph = ig.Graph.Barabasi(row['n'], row['m'], directed=False, power=1)
	else:
		raise
	h0 = np.random.randint(0, 2, row['n'], int)
	N = len(graph.vs)
	M = len(graph.es)
	with open(f'../data/input/network_{i:04d}.txt', 'w') as f:
		f.write( f'{N} {M}\n' )
		f.write( ' '.join(map(str, h0))  + '\n' )
		f.write( '\n'.join(map(lambda e: f'{e[0]} {e[1]}', [ e.tuple for e in graph.es ])) )

# %% end
