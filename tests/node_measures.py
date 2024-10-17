# %% packages

# classic
import os
from tqdm import tqdm

# specialised
import pandas as pd

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.datasets import TEPsDataset
from src.automata import LLNA
from src.simulation import join_path, all_rules_from

# %% generate TEPs' measures at node level and save them at disk

# global parameters
INPUT_PATH = join_path('..', 'data', 'teps')
OUTPUT_PATH = join_path('..', 'data', 'calc')
RESOLUTION = eval(input('Desired resolution values (list): '))
DEVICE = 'cpu'

os.makedirs(OUTPUT_PATH, exist_ok=True)

# utility functions
def result_struct(num_nodes:int=0, **identifiers):
	res = { name: identifiers[name] for name in identifiers }
	res['node'] = list(range(1, 1+num_nodes))
	return res

def delta_avg(original, disturbed):
	delta = (original != disturbed).float()
	return delta[1:].mean(0)

# iterate through all selected datasets
for R in RESOLUTION:
	for X,Y in all_rules_from(R, 2):
		rule_desc = str(LLNA(R, X, Y))
		dataset = TEPsDataset(
			path=INPUT_PATH, 
			rule=rule_desc, 
			cached=False, 
			device=DEVICE
		)
		# join calculated values by rule
		results_df = []
		for data in tqdm(dataset, desc=f'{rule_desc:16s}', ncols=80):
			# data with shape [num_inits, num_scenarios, time_steps, num_nodes]
			# (note that "num_scenarios" stands from the initial configuration + disturbs,
			# which in this case has value of 1 + num_nodes)
			for i, init_config in enumerate(data):
				original = init_config[0]
				# skips first scenario (non-disturbed)
				for j, disturbed in enumerate(init_config[1:]):
					res = result_struct(
						rule = rule_desc, 
						init = i,
						disturb = j+1,
						time_steps = disturbed.shape[-2] - 1,
						num_nodes = disturbed.shape[-1]
					)
					res['delta_avg'] = delta_avg(original, disturbed).numpy()
					results_df.append(pd.DataFrame(res))
		# save joined values in a compressed CSV file
		df = pd.concat(results_df)
		df.to_csv(join_path(OUTPUT_PATH, f'{rule_desc}.csv.xz'), compression='xz')

# %% end
   
