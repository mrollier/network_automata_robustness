import os
from tqdm import tqdm

import torch as tc
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


class NetworkDataset(Dataset):
	def __init__(self, path:str='.', cached=True, device:str='cpu'):
		full_path = lambda x: os.path.join(path, x)
		self.files = sorted(filter(os.path.isfile, map(full_path, os.listdir(path))))
		self.cached = cached
		self.init_state = []
		self.edge_list = []
		self.num_nodes = []
		
		if self.cached:
			for filename in tqdm(self.files):
				N, E, h0 = self._load(filename)
				self.num_nodes.append(N)
				self.edge_list.append(E.to(device))
				self.init_state.append(h0.to(device))
	
	def _load(self, filename):
		read_seq = lambda f, sep: map(int, f.readline().split(sep))
		with open(filename, 'r') as f:
			n_nodes, n_edges, n_inits = read_seq(f, ' ')
			h0 = tc.tensor([ list(read_seq(f, ' ')) for _ in range(n_inits) ])
			E = tc.tensor([ list(read_seq(f, ' ')) for _ in range(n_edges) ]).T
		return n_nodes, E, h0
	
	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, idx):
		if self.cached:
			return self.edge_list[idx], self.init_state[idx]
		else:
			filename = self.files[idx]
			return self._load(filename)[1:]
	
	def split(self, tr_size, *args, **kwargs):
		tr_idx, te_idx = train_test_split(tc.arange(len(self)), train_size=tr_size)
		return {
			'train': DataLoader(Subset(self, tr_idx), *args, **kwargs), 
			'test': DataLoader(Subset(self, te_idx), *args, **kwargs)
		}
