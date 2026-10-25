#!/usr/bin/env python
"""
Age-conditioned hierarchical Cox network for LUAD survival prediction.

Each age subgroup owns connection weights w and an age-specific layer;
all subgroups then pass through one age-shared layer and risk head
giving the Cox risk score E. Training minimises

    L = L_cox + L_prior + L_theta

the Cox risk term, a prior-weighted L1 term on w, and L2 on the network
weights.
"""

import torch
from torch import nn
from torch.nn import functional as F

torch.set_default_dtype(torch.float32)


class AgeSpecificLayer(nn.Module):
    """Age-specific layer: connection weights w, then an affine map (W1, b1)."""

    def __init__(self, p, specific_dim, seed=0):
        super(AgeSpecificLayer, self).__init__()
        # Local generator: reproducible w without touching the global RNG.
        gen = torch.Generator().manual_seed(int(seed))
        self.w = nn.Parameter(torch.ones(p) + 0.01 * torch.randn(p, generator=gen))
        # nn.Linear defaults, with the parameters renamed to W1 / b1.
        lin = nn.Linear(p, specific_dim)
        self.W1 = nn.Parameter(lin.weight)
        self.b1 = nn.Parameter(lin.bias)

    def forward(self, X_m):
        """(n, p) standardized expression -> (n, specific_dim)."""
        return F.gelu(F.linear(X_m * self.w, self.W1, self.b1))


class AgeSharedLayer(nn.Module):
    """Age-shared layer: an affine map (W2, b2) followed by the risk head (W3, b3)."""

    def __init__(self, specific_dim, shared_dim):
        super(AgeSharedLayer, self).__init__()
        lin = nn.Linear(specific_dim, shared_dim)
        self.W2 = nn.Parameter(lin.weight)
        self.b2 = nn.Parameter(lin.bias)
        head = nn.Linear(shared_dim, 1)
        self.W3 = nn.Parameter(head.weight)
        self.b3 = nn.Parameter(head.bias)

    def forward(self, h):
        """(n, specific_dim) -> (n,) risk score E."""
        h = F.gelu(F.linear(h, self.W2, self.b2))
        return F.linear(h, self.W3, self.b3).squeeze(-1)


class AgeConditionedCoxNet(nn.Module):
    """Age-conditioned hierarchical net: one head per subgroup, one shared body."""

    def __init__(self, p, n_subgroups=2, specific_dim=32, shared_dim=32, seed=0):
        super(AgeConditionedCoxNet, self).__init__()
        # p: genes; n_subgroups: age strata (e.g. <65 and >=65)
        self.n_subgroups = int(n_subgroups)
        self.subgroups = nn.ModuleList([
            AgeSpecificLayer(p, specific_dim, seed=seed + 100 * (m + 1))
            for m in range(self.n_subgroups)
        ])
        self.shared = AgeSharedLayer(specific_dim, shared_dim)

    def forward(self, X_list):
        """X_list: per-subgroup (n_m, p) matrices -> list of (n_m,) risk scores."""
        assert len(X_list) == self.n_subgroups, \
            f"expected {self.n_subgroups} subgroup matrices, got {len(X_list)}"
        return [self.shared(self.subgroups[m](X_list[m]))
                for m in range(self.n_subgroups)]

    def theta_parameters(self):
        """Network weights W1/b1, W2/b2, W3/b3; the gene weights w are excluded."""
        params = [self.shared.W2, self.shared.b2, self.shared.W3, self.shared.b3]
        for sg in self.subgroups:
            params += [sg.W1, sg.b1]
        return params


def cox_partial_log_likelihood(E, T, delta):
    """Breslow-style negative Cox partial log-likelihood.

    E: risk scores, T: survival times, delta: 1 marks an event. The result sums
    over events; divide by the event count to compare subgroups.
    """
    order = torch.argsort(T, descending=True)
    E = E[order]
    delta = delta[order]
    risk_sum = torch.cumsum(torch.exp(E), dim=0)
    return -torch.sum(delta * (E - torch.log(risk_sum)))


def total_loss(net, X_list, T_list, delta_list, alpha, phi_list,
               lam=4.5, gamma=0.005):
    """Total objective L = L_cox + L_prior + L_theta.

    alpha: (p,) prior weights. phi_list: one scale scalar per subgroup.
    lam: L1 strength on w, gamma: L2 strength on the network weights.
    """
    M = len(X_list)
    E_list = net(X_list)

    # L_cox: event-normalised partial likelihood, summed over subgroups.
    L_cox = sum(cox_partial_log_likelihood(E_list[m], T_list[m], delta_list[m])
                / torch.clamp(delta_list[m].sum(), min=1.0) for m in range(M))

    # L_prior: prior-weighted L1 on w (high alpha -> stronger penalty),
    # scaled by the subgroup scale phi.
    # w is detached: the L1 shrinkage of w is applied by the proximal step,
    # so this term shapes phi only.
    W = torch.stack([sg.w for sg in net.subgroups]).detach()   # (M, p)
    phi = torch.stack(phi_list).reshape(M, 1)                   # (M, 1)
    phi_norm = phi / (phi.mean().detach() + 1e-6)
    L_prior = lam * torch.sum(alpha * W.abs() / (phi_norm + 1e-6)) / W.shape[1]

    # L_theta: L2 on the network weights (W1/b1, W2/b2, W3/b3); w is excluded.
    L_theta = gamma * sum(torch.sum(q ** 2) for q in net.theta_parameters())

    return L_cox + L_prior + L_theta
