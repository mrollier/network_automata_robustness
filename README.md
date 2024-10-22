# INTRODUCTION

Here we aim to characterize a family of Life-Like Network Automata (LLNA) in the matter of its robustness. To achieve that, we analyze how a unitary perturbation (disturb) in the initial configuration propagates along the time steps for a particular network and automaton. This experiment is repeated for all node (i.e. all nodes are disturbed) multiple times (more than 1 original initial configuration), for different networks and all possible automata for a given family (resolution number).

The analysis is made by the means of the Lyapunov exponent (LE), which indicates if there is a convergence or exponential divergence in the close trajectories due some infinitesimal (in the discrete state space, unitary) disturb. Usually we only care about the maximal Lyapunov exponent (MLE), but there is a whole spectrum of LEs.

<br>

# RESEARCH DETAILS

## How this Automaton Works?

The LLNA is a cellular automata adapted from the original Game of Life to run into a network topology. It has discrete (binary) states and runs on discrete time steps. The transition rule has two _parts_: **born** and **survive**, and each one can only be triggered if the current node has a specific state _and_ its neighbors have a reative amount (percentage, density) of alive nodes. To compute the density and apply the transition rule we need to define the **resolution** number $R$: the amount of parts (subintervals) we will divide the range $[0; 1]$. A higher resolution means more possible combinations of enabled/disabled subintervals to compose the network, allowing more intrincate transition rules but also increasing the rule space.

- the _born_ condition refers only to nodes that are dead and, if triggered, will change the current node state to alive;
- the _survive_ condition refers only to nodes that are alive and, if triggered, will keep the current node state unchanged;
- only one of each rule part can be triggered at same time, but if none of them were triggered the node state will be kept/changed to dead;
- the rule parts will trigger only if the density of alive neighbors falls inside an enabled subinterval (each subinterval can be referes as a rule component);
- the set of enabled rule components (limited by $R$) for all rule parts (born and survive, so its 2) form the transition rule;
- thus there are $2^R \times 2^R$ possible rules for a given $R$.

## Limitations of the Study

- homogenous networks (only linear BA with 100 nodes and average degree $k = 4$);
- smallest family of LLNA ($R = 2$);


## Difficulties to Overcome

- What are the definitions of LE for discrete state and time and how can we use them?
- How to calculate the LE at $t \to \infty$ if after a finite (and rather small) amount of time the LE stops growing? Maybe analyze along a fixed amount of time, but to determine it?
- How does the MLE distribution changes with the increase in the LLNA resolution?
- How is the LE spectrum related to node properties like degree, centrality...?

## Ideas to Explore

### 1) Impact of the "shape" of the transition rule

**Hypothesis**: a more "discontinuous" transition rule function could make an automaton more sensible to small changes in the neighborhood state density

How the relative amount of contiguos enabled subintervals of born/survive transition rule can impact the robustness of the automaton? Imagine a LLNA's transition rule with a quite high number of resolution (and therefore a lot of subintervals composing the state density rule), half of them enabled and half of them disabled for born rule parts. If they are arranged in a way that the first part has the same enabling state, so each rule part will be in the shape of two contiguous blocks: [0; 0.5[ enabled/disabled and the opposite for [0.5; 1]. Imagine now a second configuration, with the same amount of enabled subintervals but arranged in a way that they are interleaved, thus forming a total of $R$ contiguous blocks.


It is reasonable to think that in case of a variation in neighborhood state density, the first rule configuration is less likely to be impacted as the second one. For the first case there is only one region (around $0.5$) where a disturb could change the outcome of the transition function, but for the later there are $R - 1$ discontinuities in the rule-part function that could cause changes in the outcome value.

<br>

# PROJECT STRUCTURE

## Organization of the Files

<font color="red">...</font>

## Experiment Pipeline

<font color="red">...</font>

## TO-DO List

<font color="red">...</font>



