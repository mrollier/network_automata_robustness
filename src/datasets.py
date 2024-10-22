import os
import shutil
import pickle
from tqdm import tqdm

import torch as tc
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split



class GenericDataset(Dataset):
	def __init__(self, path:str='.', cached:bool=True, device:str='cpu', transform:callable=None):
		dir_content = os.listdir(path)
		append_path = lambda x: os.path.join(path, x)
		self._files = sorted(filter(os.path.isfile, map(append_path, dir_content)))
		self._cached = cached
		self._device = device
		self._transf = transform
		self._samples = list(map(self._load, tqdm(self._files))) if self._cached else []
	
	def _load(self, filename:str):
		raise NotImplementedError
	
	def _parse(self, f, sep:str, dtype:type, repeat:int=0):
		if repeat == 0:
			return tuple(map(dtype, f.readline().split(sep)))
		elif repeat > 0:
			return [ self._parse(f, sep, dtype, 0) for _ in range(repeat) ]
		else:
			raise ValueError
	
	def __len__(self):
		return len(self._files)
	
	def __getitem__(self, idx:int):
		if idx >= 0 and idx < len(self):
			return self._samples[idx] if self._cached else self._load(self._files[idx])
		else:
			raise IndexError
	
	@property
	def filenames(self):
		return self._files
	
	def split(self, tr_size, *args, **kwargs):
		tr_idx, te_idx = train_test_split(tc.arange(len(self)), train_size=tr_size)
		return {
			'train': DataLoader(Subset(self, tr_idx), batch_size=None, *args, **kwargs), 
			'test': DataLoader(Subset(self, te_idx), batch_size=None, *args, **kwargs)
		}



class NetworksDataset(GenericDataset):
	def __init__(self, as_undirected:bool=False, max_inits:int=0, **kwargs):
		self._as_undir = as_undirected
		self._max_inits = max_inits
		GenericDataset.__init__(self, **kwargs)
	
	def _load(self, filename:str):
		with open(filename, 'r') as f:
			n_nodes, n_edges, n_inits = self._parse(f, ' ', int)
			edges = tc.tensor(self._parse(f, ' ', int, n_edges))
			states = tc.tensor(self._parse(f, ' ', int, n_inits))
		if self._as_undir:
			edges_dir = list( map(tuple, edges.numpy()) )
			edges_rev = list( map(tuple, reversed(edges.T).numpy().T) )
			edges = tc.tensor( sorted(set(edges_dir + edges_rev)) )
		if self._max_inits > 0:
			l = min(states.shape[0], self._max_inits)
			states = states[:l]
		edges = edges.T.to(self._device)
		states = states.to(self._device)
		return (edges, states) if self._transf is None else self._transf(edges, states)
	
	def is_directed(self):
		for (edges, _) in self:
			edge_list = set(map(tuple, edges.numpy().T))
			for e in edge_list:
				if tuple(reversed(e)) not in edge_list:
					return True
		return False



class TEPsDataset(GenericDataset):
	def __init__(self, rule:str=None, path:str='.', **kwargs):
		assert(rule is not None)
		self._rule = rule
		self._workspace = os.path.join(path, self._rule)
		shutil.unpack_archive(self._workspace + '.xztar', self._workspace, 'xztar')
		GenericDataset.__init__(self, path=self._workspace, **kwargs)
		
	def __del__(self):
		shutil.rmtree(self._workspace)
		
	def _load(self, filename:str):
		with open(filename, 'rb') as f:
			teps = pickle.load(f).long().to(self._device)
		return teps if self._transf is None else self._transf(teps)
