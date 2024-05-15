# %% packages

# classic
import os
import torch as tc
from tqdm import tqdm

# specialised
import torch as tc

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.datasets import NetworkDataset
from src.automata import LLNA
from src.simulation import *

# %% generate time-evolution patterns and save at disk

# global parameters (be careful when changing numeric variables to higher values)
INPUT_PATH = '../data/graphs'
OUTPUT_PATH = '../data/teps'
RESOLUTION = 2
NUM_STEPS = 50
MAX_INITS = 1

# open the dataset (when 'cached=True', keeps all graphs in memory)
dataset = NetworkDataset(path=INPUT_PATH, cached=True)

# create LLNA for all possible rules with same resolution
for X,Y in all_rules_from(RESOLUTION, 2):
    automaton = LLNA(RESOLUTION, X, Y)
    rule_desc = str(automaton)
    full_path = os.path.join(OUTPUT_PATH, *rule_desc.split(':'))
    pbar = tqdm(total=len(dataset), desc=f'{rule_desc:16s}', position=0, leave=True)
    
    # calculate TEPs for all graphs in dataset, for all possible disturbs
    for i, (E, h0) in enumerate(dataset):
        l = min(h0.shape[0], MAX_INITS)
        H0 = prepare_shape(tc.stack([ orth_disturbs(h, True) for h in h0[:l] ], 0))
        Ht = restore_shape(automaton(E, H0, T=NUM_STEPS))
        
        # save TEPs
        # NOTE: Michiel changed the split argument to \\ for his local machine
        file_name = dataset.files[i].split('\\')[-1].split('.')[0]
        save_tensor(full_path, file_name, Ht, verbose=False)
        pbar.update()
    pbar.close()

# %% end

