#The code below is part of Michiel's code
resolution = 2
model = LLNA(resolution) # takes a random LLNA at this resolution
rule = str(model)

s0 = np.random.randint(2, size=N).astype(float)
s0_defect = s0.copy()
s0_defect[0] = 1-s0_defect[0]

def jacobian(graph, model, states):
    N = len(states)
    edges_from = np.array(graph.get_edgelist())
    edges_to = np.array(graph.get_edgelist())[:,::-1]
    edges_both = np.append(edges_from, edges_to, axis=0)
    edges = tc.tensor(edges_both.T, dtype=tc.int64)
    states_all = np.tile(states, (N,1))
    states_all_next = np.array(model.step(edges, tc.tensor(states_all)))
    states_defect_all = np.abs(np.tile(states, (N,1)) - np.diag(np.ones(N)))
    states_defect_all_next = np.array(model.step(edges, tc.tensor(states_defect_all)))
    J = np.mod(states_all_next + states_defect_all_next, 2)
    return J

#Here, my contribution to the Lyapunov exponent calculation
#Configuration space
def conf_space(G, model, s0, s0_defect, T):
    #Delta x0 calculation:
    N = len(s0)
    edges = G.get_edgelist()
    D0 = np.mod(s0 ^ s0_defect, 2)  #initial difference between the two configurations
    D = D0 #initializing at t = 0
    states_original = np.tile(s0, (T,1)) # all states (will be updated in the loop)
    Js = np.zeros(shape=(T-1,N,N)) #Allocate space for Jacobian
    for t in range(1,T):
        states_original[t] = np.array(model.step(tc.tensor(edges).T, tc.tensor(states_original[t-1])))
        Js[t-1] = jacobian(G, model, states_original[t])
        #Delta x^t = Js * Delta x^t-1
        D = np.mod(np.matmul(Js[t-1], D),2)
    #Here, the Lyapunov exponent
    L = np.log(sum(D)/sum(D0))/T
    return L 

 
#Tangent space
def tangent_space(G, model, s0, s0_defect, T):
    #Allocating space
    N = len(s0)
    edges = G.get_edgelist()
    Js = np.zeros(shape=(T-1,N,N))
    states_original = np.tile(s0, (T,1))
    states_perturbed = np.tile(s0_defect, (T,1))
    deltas = np.tile(s0_defect, (T,1))
    for t in range(1,T):
        states_original[t] = np.array(model.step(tc.tensor(edges).T, tc.tensor(states_original[t-1])))
        states_perturbed[t] = np.array(model.step(tc.tensor(edges).T, tc.tensor(states_perturbed[t-1])))
        Js[t-1] = jacobian(G, model, states_original[t])
        deltas[t] = np.mod(np.matmul(Js[t-1], deltas[t-1]),2)
    #The Lyapunov exponent
    L = np.log(sum(deltas[t]))/T
    return L





    