# %% packages

# classic
import matplotlib.pyplot as plt

# specialised
import igraph as ig
import torch as tc

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.na.llna import LLNA
    
# %% make instance of Life-Like NA

automaton = LLNA(resolution=5, x=[1,2,3], y=[0,4])
print('Rule:', automaton.rule)

# %% Make instance of E-R network

initial_configs = 3

G = ig.Graph.Erdos_Renyi(n=100, p=0.2)
N = len(G.vs) # vertices
E = tc.tensor([ e.tuple for e in G.es ]).T # edges
h0 = tc.randint(0, 2, (initial_configs, N)) # initial configuration

# %% add network to LLNA

H = automaton(E, h0, 50) # evolve over 50 timesteps
print('Shape of TEP matrix:', H.shape)

# %% plot TEP
# Q: how are these ordered?

for Hi in H:
    plt.imshow(Hi, cmap='gray')
    plt.show()

    
# %% end
