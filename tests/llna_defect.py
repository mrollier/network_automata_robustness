# %% packages

# classic
import matplotlib.pyplot as plt

# specialised
import igraph as ig
import torch as tc

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.automata import LLNA

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
fig, ax = plt.subplots(3,1, sharex=True, sharey=True)
titles = ['original TEPs', 'disturbed TEPs']
for i, Hi in enumerate(H):
    ax[i].imshow(Hi, cmap='Greys')
    ax[i].set(title=titles[i])

# %% plot defect
Hdef = abs(H[1] - H[0])
ax[2].imshow(Hdef, cmap='Greys')
ax[2].set(title='defect propagation')

plt.tight_layout()
plt.show()

# %% end
