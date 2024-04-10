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

resolution=5
create = [1,2,3]
destroy = [0,4]
automaton = LLNA(resolution=5, x=create, y=destroy)
automaton.rule

# %% Make instance of E-R network

N = 100 # number of vertices
G = ig.Graph.Erdos_Renyi(n=N, p=0.2)
E = tc.tensor([ e.tuple for e in G.es ]).T # edges

# initial configuration with a single defect
h0 = tc.zeros(2,N)
h0[0] = tc.randint(0,2,(N,))
h0[1] = h0[0]
h0[1, N//2] = 1 - h0[1, N//2]

# %% add network to LLNA

H = automaton(E, h0, 50) # evolve over 50 timesteps
H.shape

# %% plot TEPs
# Q: how are these ordered?

for Hi in H:
    plt.imshow(Hi, cmap='Greys')
    plt.show()

# %% plot defect
Hdef = abs(H[1] - H[0])
plt.imshow(Hdef, cmap='Greys')
plt.show()
    
# %% end