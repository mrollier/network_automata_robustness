# %% packages

# classic
import os
import shutil
import torch as tc
from tqdm import tqdm

# specialised
import torch as tc

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.datasets import NetworksDataset
from src.automata import LLNA
from src.simulation import *

# %% generate time-evolution patterns and save at disk

# global parameters (be careful when changing numeric variables to higher values)
INPUT_PATH = join_path('..', 'data', 'graphs')
OUTPUT_PATH = join_path('..', 'data', 'teps')
RESOLUTION = [2]
NUM_STEPS = 50
MAX_INITS = 1
DEVICE = 'cpu'

# transformation that for each initial configuration generates all possibles 
# orthogonal disturbs and append them to the original initial configuration 
def stack_disturbs(E, h0):
    H0 = tc.stack([ orth_disturbs(h, True) for h in h0 ], 0)
    return E, H0

# open the dataset (when 'cached=True', keeps all graphs in memory)
dataset = NetworksDataset(
	path=INPUT_PATH, 
	cached=False, 
	directed=False, 
	max_inits=MAX_INITS, 
	transform=stack_disturbs, 
	device=DEVICE
)

# create LLNA for all possible rules with same resolution from the resolution list
for R in RESOLUTION:
	for X,Y in all_rules_from(R, 2):
		automaton = LLNA(R, X, Y)
		rule_desc = str(automaton)
		temp_folder = join_path(OUTPUT_PATH, rule_desc)
		pbar = tqdm(total=len(dataset), desc=f'{rule_desc:16s}', position=0, leave=True)
		
		# calculate TEPs for all graphs in dataset, for all possible disturbs
		for i, (E, h0) in enumerate(dataset):
			# converts from original shape [num_inits, num_scenarios, num_nodes]
			# to flattened shape [num_inits X num_scenarios, num_nodes]
			# (note that "num_scenarios" stands from the initial configuration + disturbs,
			# which in this case has value of 1 + num_nodes)
		    H0 = prepare_shape(h0)
			# converts from flattened shape [num_inits X num_scenarios, time_steps, num_nodes]
			# to expanded shape [num_inits, num_scenarios, time_steps, num_nodes]
			# (note that "time_steps" includes the initial configuration, being therefore 1+T)
		    Ht = restore_shape(automaton(E, H0, T=NUM_STEPS))
		    
		    # save TEPs in temporary folder
		    file_name = os.path.basename(dataset.filenames[i]).split('.')[0]
		    save_tensor(temp_folder, file_name, Ht, verbose=False)
		    pbar.update()
		pbar.close()
		
		# compress temporary folder into a zip file named after the automaton rule
		shutil.make_archive(temp_folder, 'zip', temp_folder)
		shutil.rmtree(temp_folder)

# %% end
