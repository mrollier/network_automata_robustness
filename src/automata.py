# %% import packages
import matplotlib.pyplot as plt
import numpy as np

# special packages
import torch as tc
import torch_geometric as tg
from functools import reduce

# %% define class

class LLNA(tc.nn.Module):
	def __init__(self, resolution:int, x:list=None, y:list=None):
		super(LLNA, self).__init__()
		self.conv_gnn = tg.nn.conv.SimpleConv(aggr='mean', combine_root=None)
		self._resolution = resolution
		self.rule = (x, y)
		self.callback = None
	
	def __str__(self):
		#concat = lambda arr: reduce(lambda x,y: f'{x},{y}', map(str, arr), '')
		concat = lambda arr: ','.join(list(map(str, arr)))
		(x, y) = self.rule
		return f'R={self._resolution}:B{concat(x)}_S{concat(y)}'
	
	@property
	def resolution(self):
		return self._resolution
	
	@property
	def rule(self):
		indices = tc.arange(self._resolution).int()
		decode = lambda v: indices[ v.squeeze().bool() ].tolist()
		return (decode(self._x), decode(self._y))
	
	@rule.setter
	def rule(self, indices:tuple):
		x, y = indices
		for v in [x, y]:
			assert(v is None or not len(v) or (min(v) >= 0 and max(v) <= self.resolution-1))
		shape = (self._resolution, 1)
		encode = lambda l: tc.tensor([ float(k in l) for k in range(self._resolution) ]).reshape(shape)
		self._x = encode(x) if x is not None else tc.randint(0, 2, shape).float()
		self._y = encode(y) if y is not None else tc.randint(0, 2, shape).float()
	
	def interval_encoding(self, p:tc.Tensor):
		belongs_to  = lambda x, k: ((k <= self._resolution*x) & (self._resolution*x < k+1))
		is_boundary = lambda x, k: ((k == self._resolution-1) & (x == 1))
		return tc.stack([ 
			(belongs_to(p, k) | is_boundary(p, k)).float() for k in range(self._resolution)
		], 2)
	
	def step(self, E:tc.Tensor, h:tc.Tensor):
		h = tc.atleast_2d(h)
		p = self.conv_gnn(h.T, E).T
		R = self.interval_encoding(p)
		b = tc.matmul(R, self._x).squeeze(2) * (1-h)
		s = tc.matmul(R, self._y).squeeze(2) * h
		if self.callback is not None:
			self.callback({ 'E':E, 'h':h, 'p':p, 'R':R, 'b':b, 's':s })
		return (b + s)
	
	def forward(self, E:tc.Tensor, ht:tc.Tensor, T:int=1):
		"""
		==========
		input
		==========
		E   : long tensor of shape [2 x M], where M is the number of directed edges 
		      (for undirected graphs, ensure the existance of both (vi, vj) and (vj, vi) in E)
		ht  : float tensor of shape [L x N], where N and L are the number of nodes and initial configurations
		T   : number of steps to run the automaton
		
		==========
		output
		==========
		H   : float tensor of shape [L x T+1 x N] (the initial configuration plus T steps)
		"""
		H = [ht]
		for t in range(T):
			ht = self.step(E, ht.detach())
			H.append(ht)
		return tc.stack(H, 1)

	def diagram(self, ax=None):
		"""
		Method to visualise the LLNA update rules in a diagram

		Parameters
		----------
		ax : matplotlib.axes._axes.Axes
			Axes object serves as target for diagram. Default is None (in which case this method creates its own Axes object)

		Returns
		-------
		fig : matplotlib.figure.Figure
			Figure object used to further enhance or customise the diagram
		ax : matplotlib.axes._axes.Axes
			Axes object used to further enhance or customise the diagram
		"""
		# make Axes object if required
		if ax is None:
			fig, ax = plt.subplots(1,1,figsize=(10,1.8))
		# create the desired density subdomains
		born_if = self.rule[0]; survive_if = self.rule[1]
		subdoms = []; born = []; survive = []
		for i in range(self._resolution):
			subdoms += [i, i+1]
			if i in born_if:
				born += [1, 1]
			else:
				born += [0, 0]
			if i in survive_if:
				survive += [1, 1]
			else:
				survive += [0, 0]
		subdoms = np.array(subdoms) / self._resolution
		born = np.array(born)
		survive = np.array(survive)

		# diagram qualities
		offset = 0.005
		born_color = 'red'
		survive_color = 'green'
		# top curves
		ax.plot(subdoms, born+offset, color=born_color)
		ax.plot(subdoms, survive-offset, color=survive_color)
		# hatches
		ax.fill_between(subdoms, 0, born+offset, facecolor=born_color, alpha=.2, hatch='/', edgecolor=born_color, label='dead')
		ax.fill_between(subdoms, 0, survive+offset, facecolor=survive_color, alpha=.2, hatch='\\', edgecolor=survive_color, label='alive')

		# add markers for discontinuous points
		# ax.scatter(subdoms[1], born[1], s=50, color='green')
		# ax.scatter(subdoms[1], survive[1], s=50, color='red')

		# add x tick information
		xticks = np.arange(self._resolution*2+1)/(self._resolution*2)
		xticklabels = [None]*len(xticks)
		xticklabels[0] = 0
		xticklabels[1] = f"$[0, 1/{self._resolution}[$"
		xticklabels[-2] = f"$[{self._resolution-1}/{self._resolution}, 1]$"
		xticklabels[-1] = 1
		for i in range(3, (self._resolution-1)*2, 2):
			xticklabels[i] = f"$[{i//2}/{self._resolution}, {i//2+1}/{self._resolution}[$"
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticklabels)

		# add y tick information
		yticks = [0,1]
		yticklabels = ["dead", "alive"]

		ax.set_yticks(yticks)
		ax.set_yticklabels(yticklabels, rotation=45)

		ax.set_ylabel("Node becomes ...")
		ax.set_xlabel(f"State density $\\rho$ of the node's neighbourhood")

		ax.legend(title='Node was ...', ncol=2, loc='right')
		ax.set_title(f"Diagram for LLNA {self.__str__()}")

		# return
		return fig, ax

