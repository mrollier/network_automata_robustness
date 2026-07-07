# %% import packages

import matplotlib.pyplot as plt
import numpy as np

# special packages
import torch as tc
import torch_geometric as tg
from scipy.special import comb

# %% define class


class LLNA(tc.nn.Module):
    def __init__(self, resolution: int, x: list = None, y: list = None, iso: bool = False):
        super().__init__()
        self.conv_gnn = tg.nn.conv.SimpleConv(aggr="mean", combine_root=None)
        if iso and not self._is_odd_integer(resolution):
            raise ValueError(f"The resolution {resolution} is not an odd integer.")
        self._iso = iso
        self._resolution = resolution
        self.rule = (x, y)
        self.callback = None

    def __str__(self, latex=False):
        encode = lambda arr: sum(2 ** np.array(arr))
        (x, y) = self.rule
        if latex:
            # return f'$R_{{{self._resolution}}}B_{{{encode(x)}}}S_{{{encode(y)}}}$'
            return f"$\\phi^{{{self._resolution}}}_{{{encode(x)},{encode(y)}}}$"
        return f"R{self._resolution}B{encode(x)}S{encode(y)}"

    @property
    def resolution(self):
        return self._resolution

    @property
    def rule(self):
        indices = tc.arange(self._resolution).int()
        decode = lambda v: indices[v.squeeze().bool()].tolist()
        return (decode(self._x), decode(self._y))

    @rule.setter
    def rule(self, indices: tuple):
        x, y = indices
        for v in [x, y]:
            assert v is None or not len(v) or (min(v) >= 0 and max(v) <= self.resolution - 1)
        shape = (self._resolution, 1)
        encode = lambda idx: tc.tensor([float(k in idx) for k in range(self._resolution)]).reshape(shape)
        self._x = encode(x) if x is not None else tc.randint(0, 2, shape).float()
        self._y = encode(y) if y is not None else tc.randint(0, 2, shape).float()

    def interval_encoding(self, p: tc.Tensor):
        if not self._iso:  # classic psuedo-isomorphic case
            belongs_to = lambda x, k: (k <= self._resolution * x) & (self._resolution * x < k + 1)
            is_boundary = lambda x, k: (k == self._resolution - 1) & (x == 1)
            return tc.stack(
                [(belongs_to(p, k) | is_boundary(p, k)).float() for k in range(self._resolution)], 2
            )
        else:  # altered isomorphic case
            belongs_to_lower = lambda x, k: (
                (k < self._resolution / 2) & (x >= k / self._resolution) & (x < (k + 1) / self._resolution)
            )
            belongs_to_middle = lambda x, k: (
                (k == (self._resolution - 1) / 2)
                & (x >= k / self._resolution)
                & (x <= (k + 1) / self._resolution)
            )
            belongs_to_upper = lambda x, k: (
                (k > self._resolution / 2) & (x > k / self._resolution) & (x <= (k + 1) / self._resolution)
            )
            return tc.stack(
                [
                    (belongs_to_lower(p, k) | belongs_to_middle(p, k) | belongs_to_upper(p, k)).float()
                    for k in range(self._resolution)
                ],
                2,
            )

    def step(self, E: tc.Tensor, h: tc.Tensor):
        h = tc.atleast_2d(h)
        p = self.conv_gnn(h.T, E).T
        R = self.interval_encoding(p)
        # born
        b = tc.matmul(R, self._x).squeeze(2) * (1 - h)
        # survive
        s = tc.matmul(R, self._y).squeeze(2) * h
        if self.callback is not None:
            self.callback({"E": E, "h": h, "p": p, "R": R, "b": b, "s": s})
        return b + s

    def forward(self, E: tc.Tensor, ht: tc.Tensor, T: int = 1):
        """
        ==========
        input
        ==========
        E   : long tensor of shape [2 x M], where M is the number of directed edges
              (for undirected graphs, ensure the existance of both (vi, vj) and (vj, vi) in E)
        ht  : float tensor of shape [L x N], where L is the number of initial configurations and N is the number of nodes
        T   : number of steps to run the automaton

        ==========
        output
        ==========
        H   : float tensor of shape [L x T+1 x N] (the initial configuration plus T steps)
        """
        H = [ht]
        for _ in range(T):
            ht = self.step(E, ht.detach())
            H.append(ht)
        return tc.stack(H, 1)

    def diagram(self, ax=None, degree: int = 0, plot_dist=False):
        """
        Method to visualise the LLNA update rules in a diagram

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes
                Axes object serves as target for diagram. Default is None (in which case this method creates its own Axes object)
        degree : int
                If not zero, plot the number of possible densities that may arise for a particular degree.

        Returns
        -------
        fig : matplotlib.figure.Figure
                Figure object used to further enhance or customise the diagram
        ax : matplotlib.axes._axes.Axes
                Axes object used to further enhance or customise the diagram
        """
        # raise exception if degree does not make sense
        if degree < 0 or degree > 64:
            raise Exception(
                f"Degree d={degree} is not allowed. Choose 0 or a positive integer smaller than 64."
            )
        if degree == 0 and plot_dist:
            raise Exception(
                "Plotting the state density distribution requires entering the degree of the node."
            )
        # make Axes object if required
        return_both = False
        if ax is None:
            return_both = True
            fig, ax = plt.subplots(1, 1, figsize=(10, 1.8))
        # create the desired density subdomains
        born_if = self.rule[0]
        survive_if = self.rule[1]
        subdoms = []
        born = []
        survive = []
        for i in range(self._resolution):
            subdoms += [i, i + 1]
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
        born_color = "green"
        survive_color = "dodgerblue"
        # top curves
        ax.plot(subdoms, born + offset, color=born_color)
        ax.plot(subdoms, survive - offset, color=survive_color)
        # hatches
        ax.fill_between(
            subdoms,
            0,
            born + offset,
            facecolor=born_color,
            alpha=0.2,
            hatch="//",
            edgecolor=born_color,
            label="dead",
        )
        ax.fill_between(
            subdoms,
            0,
            survive + offset,
            facecolor=survive_color,
            alpha=0.2,
            hatch="\\\\",
            edgecolor=survive_color,
            label="alive",
        )

        # add markers for discontinuous points
        # ax.scatter(subdoms[1], born[1], s=50, color='green')
        # ax.scatter(subdoms[1], survive[1], s=50, color='red')

        # add x tick information
        xticks = np.arange(self._resolution * 2 + 1) / (self._resolution * 2)
        xticklabels = [None] * len(xticks)
        xticklabels[0] = 0
        xticklabels[1] = f"$[0, 1/{self._resolution}[$"
        # classic pseudo-isomorphic case
        if not self._iso:
            xticklabels[-2] = f"$[{self._resolution - 1}/{self._resolution}, 1]$"
        # isomorphic case
        else:
            xticklabels[-2] = f"$]{self._resolution - 1}/{self._resolution}, 1]$"
        xticklabels[-1] = 1
        # classic pseudo-isomorphic case
        if not self._iso:
            for i in range(3, (self._resolution - 1) * 2, 2):
                xticklabels[i] = f"$[{i // 2}/{self._resolution}, {i // 2 + 1}/{self._resolution}[$"
        # isomorphic case
        else:
            for i in range(3, self._resolution, 2):  # lhs
                xticklabels[i] = f"$[{i // 2}/{self._resolution}, {i // 2 + 1}/{self._resolution}[$"
            xticklabels[self._resolution] = (
                f"$[{(self._resolution - 1) // 2}/{self._resolution}, {(self._resolution + 1) // 2}/{self._resolution}]$"  # middle
            )
            for i in range(self._resolution + 2, (self._resolution - 1) * 2, 2):  # rhs
                xticklabels[i] = f"$]{i // 2}/{self._resolution}, {i // 2 + 1}/{self._resolution}]$"
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # add y tick information
        yticks = [0, 1]
        yticklabels = ["dead", "alive"]

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, rotation=45)

        ax.set_ylabel("Node becomes ...")
        ax.set_xlabel("State density $\\rho$ of the node's neighbourhood")

        ax.legend(title="Node was ...", ncol=2, loc="right")
        ax.set_title(f"Diagram for LLNA {self.__str__(latex=True)}")

        # add possible densities, if required
        if degree:
            rhos = np.linspace(0, 1, degree + 1)
            for rho in rhos:
                ax.axvline(rho, color="k", ls="--")
            # add binomial distribution, if required
            if plot_dist:
                rho_dens_dist = np.array([comb(degree, k) for k in range(degree + 1)])
                rho_dens_dist = rho_dens_dist / np.max(rho_dens_dist)
                ax.scatter(
                    rhos,
                    rho_dens_dist,
                    marker="o",
                    edgecolors="black",
                    facecolors="white",
                    s=50,
                    zorder=3,
                    clip_on=False,
                )

        # return
        if return_both:
            return fig, ax
        return ax

    def hamming_weight(self, degree: int, norm: bool = True):
        """
        Calculate the (normalised) Hamming weight of the local update rule for a particular node degree.

        Delegates to llna.analysis.hamming_weight (the single implementation).
        The analysis function assigns boundary densities with float64
        arithmetic, whereas this method historically used the float32 torch
        encoding; values can differ where a density q/degree lies exactly on
        an interval edge k/resolution (pinned in the characterization fixtures).

        Parameters
        ----------
        degree : int
                The degree of the node you want to calculate the Hamming weight for
        norm : bool
                True by default. Maps the Hamming weight to a value from 0 to 1.

        Returns
        -------
        HW : float or int
                The (normalised) Hamming weight
        """
        from llna.analysis import hamming_weight

        born_if, survive_if = self.rule
        return hamming_weight(self._resolution, born_if, survive_if, degree, norm=norm, iso=self._iso)

    def boolean_sens(self, degree: int, norm_degree: bool = True):
        """
        Calculate the (normalised) Boolean sensitivity of the local update rule for a particular node degree.

        Delegates to llna.analysis.boolean_sens (the single implementation);
        see hamming_weight for the boundary-density caveat.

        Parameters
        ----------
        degree : int
                The degree of the node you want to calculate the Boolean sensitivity for
        norm_degree : bool
                True by default. Maps the Boolean sensitivity to a value from 0 to 1.

        Returns
        -------
        BS : float
                The (normalised) Boolean sensitivity
        """
        from llna.analysis import boolean_sens

        born_if, survive_if = self.rule
        return boolean_sens(
            self._resolution,
            born_if,
            survive_if,
            degree,
            norm_degree=norm_degree,
            current_dens=0.5,
            iso=self._iso,
        )

    @staticmethod
    def _is_odd_integer(res):
        """Check if the resolution is an odd integer. This is required when the LLNA must have automorphisms."""
        return isinstance(res, int) and res % 2 == 1
