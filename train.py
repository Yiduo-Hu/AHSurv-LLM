#!/usr/bin/env python
"""
Training for the age-conditioned hierarchical Cox network.

theta and phi are updated by plain gradient descent. w is updated by a
proximal gradient step whose soft-threshold mu is set by the prior weights,
the subgroup scale phi, and the running Cox importance s.
"""

import math

import torch

from model import total_loss


def lr_schedule(epoch, max_iter, lr, decay_start, min_lr):
    """Constant lr, then cosine annealing down to min_lr by the last iteration."""
    start = min(int(decay_start * max_iter), max_iter - 1)
    if epoch <= start:
        return lr
    t = (epoch - start) / max(1, max_iter - start)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))


def prox_scale(progress, decay_start, decay_end):
    """Strength of the soft-threshold: 1, then cosine down to 0, then off."""
    if progress <= decay_start:
        return 1.0
    if progress >= decay_end:
        return 0.0
    t = (progress - decay_start) / (decay_end - decay_start)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def train_model(net, X_list, T_list, delta_list, alpha, phi_list,
                lam=4.5, gamma=0.005,
                lr=5e-4, max_iter=2000,
                warmup_ratio=0.15, decay_start=0.4, decay_end=0.7,
                min_lr=1e-6,
                beta=0.9, kappa=1.5, omega=0.85):
    """Train the network and return the final w of every subgroup.

    alpha: (p,) prior weights. phi_list: one learnable scale scalar per
    subgroup, initialised near 1. lam: L1 strength on w, gamma: L2 strength
    on theta. warmup_ratio, decay_start and decay_end are fractions of
    max_iter; the learning rate and the soft-threshold both follow a cosine
    schedule. beta, kappa and omega shape the Cox importance s.
    """
    M = len(X_list)
    theta_params = net.theta_parameters()
    w_params = [sg.w for sg in net.subgroups]
    # s: running Cox importance, the survival contribution of each gene
    s = torch.zeros(M, w_params[0].shape[0], device=w_params[0].device)

    for epoch in range(max_iter):
        loss = total_loss(net, X_list, T_list, delta_list, alpha, phi_list,
                          lam=lam, gamma=gamma)
        net.zero_grad(set_to_none=True)
        for p_ in phi_list:
            p_.grad = None
        loss.backward()

        progress = (epoch + 1) / max_iter
        lr_now = lr_schedule(epoch + 1, max_iter, lr, decay_start, min_lr)
        warm = min(1.0, progress / max(warmup_ratio, 1e-12))
        scale = prox_scale(progress, decay_start, decay_end)

        with torch.no_grad():
            # s: exponential moving average of the absolute Cox gradient of w
            for m, w in enumerate(w_params):
                s[m].mul_(beta).add_(w.grad.abs(), alpha=1.0 - beta)
            # theta: gradient descent on the Cox + L2 gradient
            for p_ in theta_params:
                if p_.grad is not None:
                    p_.data.add_(p_.grad, alpha=-lr_now)
            # phi: gradient descent on the prior-term gradient
            for p_ in phi_list:
                if p_.grad is not None:
                    p_.data.add_(p_.grad, alpha=-lr_now)
            # w: gradient step, then soft-thresholding by mu
            phi = torch.stack([q.detach() for q in phi_list]).reshape(M, 1)
            phi_norm = phi / (phi.mean() + 1e-6)
            for m, sg in enumerate(net.subgroups):
                w = sg.w
                w.data.add_(w.grad, alpha=-lr_now)
                # a large s means a large survival contribution: shrink less
                importance = 1.0 / (s[m] + omega).pow(kappa)
                mu = (lr_now * scale * warm * lam
                      * alpha / (phi_norm[m] + 1e-6) * importance)
                shrunk = torch.clamp(w.data.abs() - mu, min=0.0)
                w.data.copy_(torch.sign(w.data) * shrunk)

    return [sg.w.detach().clone() for sg in net.subgroups]
