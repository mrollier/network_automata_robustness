# %% import packages

# special packages
import os
import pickle
import torch as tc
import itertools as itt
from functools import reduce

# %% utilitary functions
def join_path(*path_parts):
	joined = reduce(os.path.join, path_parts)
	normalized = os.path.normpath(joined)
	return normalized

def orth_disturbs(h:tc.Tensor, append:bool=True) -> tc.Tensor:
	"""
	Create a set disturbed initial conditions based on N orthogonal perturbations. 
	If 'append=True', returns the initial condition as the first row followed by 
	the genearted disturbs, otherwise returns only the disturbs.
	"""
	N = h.shape[0]
	delta = tc.eye(N, dtype=tc.long)
	ha = [h] if append else [] # if 'append==True', adds 'h' as the first row
	return tc.stack(ha + [ h ^ delta[i] for i in range(N) ], 0)

def stack_disturbs(E:tc.Tensor, h0:tc.Tensor) -> tuple:
	"""
	transformation that for each initial configuration generates all possibles 
	orthogonal disturbs and append them to the original initial configuration
	"""
	H0 = tc.stack([ orth_disturbs(h, True) for h in h0 ], 0)
	return E, H0

def prepare_shape(H:tc.Tensor) -> tc.Tensor:
	"""
	Given a tensor with dimensions (L, C, N), joins all dimensions except the last 
	one to prepare for automata iterations.
	"""
	num_nodes = H.shape[-1]
	return H.reshape(-1, num_nodes)

def restore_shape(H:tc.Tensor) -> tc.Tensor:
	"""
	Given a tensor with dimensions (L*C, T, N), separate the first dimension in two,
	as C=N+1 if running all perturbations, to restore tensor to original shape after 
	automata iterations.
	"""
	num_steps, num_nodes = H.shape[-2:]
	num_inits = H.shape[0] // (num_nodes + 1)
	return H.reshape(num_inits, -1, num_steps, num_nodes)

def all_subsets_from(s, l_min:int=0):
	"""
	Given a set {a, b, ...} returns all possible subsets with lenght >= 'l_min'
	"""
	assert(l_min >= 0 and l_min <= len(s))
	s = list(s)
	ss = itt.chain.from_iterable(itt.combinations(s, r) for r in range(l_min, len(s)+1))
	return ss

def all_rules_from(n_components:int, n_transisions:int):
	"""
	Given a number of rule components and a number of independent transitions, 
	generate all possible combinations of rules per transition.
	"""
	assert(n_components >= 2 and n_transisions >= 1)
	rule_set = list(all_subsets_from(range(n_components), 0))
	return itt.product(rule_set, repeat=n_transisions)

def save_tensor(path, name, data, verbose=False):
	os.makedirs(path, exist_ok=True)
	with open(join_path(path, f'{name}.pkl'), 'wb') as f:
		pickle.dump(data.bool(), f)
	if verbose:
		print(f.name + ' saved in disk')

def load_tensor(path, name, verbose=False):
	with open(join_path(path, f'{name}.pkl'), 'rb') as f:
		data = pickle.load(f).long()
	if verbose:
		print(f.name + ' loaded from disk')
	return data
