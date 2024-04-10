# %% import packages

# special packages
import torch as tc
import torch_geometric as tg

# %% define class

class LLNA(tc.nn.Module):
    def __init__(self, resolution:int, x:list=None, y:list=None):
        super(LLNA, self).__init__()
        self.conv_gnn = tg.nn.conv.SimpleConv(aggr='mean')
        self._resolution = resolution
        self.rule = (x, y)
    
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
            assert(v is None or (min(v) >= 0 and max(v) <= self.resolution-1))
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
        return (b + s)
    
    def forward(self, E:tc.Tensor, ht:tc.Tensor, T:int=1):
        """
        ==========
        input
        ==========
        E   : long tensor of shape [2 x M], where M is the number os edges
        ht  : float tensor of shape [L x N], where N and L are the number of nodes and initial configurations
        T   : number of steps to run the automaton
        
        ==========
        output
        ==========
        H   : float tensor of shape [L x T+1 x N] (the initial configuration plus T steps)
        """
        H = [ht]
        P = []
        for t in range(T):
            ht = self.step(E, ht.detach())
            H.append(ht)
        return tc.stack(H, 1)