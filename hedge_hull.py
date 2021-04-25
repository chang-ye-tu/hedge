import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp, zeros, ones, std, mean, maximum, cumsum, cumprod, reshape, transpose

np.random.seed(1000)
phi = norm(loc=0, scale=1).cdf

def bs_d1(S, dt, r, sigma, K):
    return (log(S / K) + (r + sigma**2 / 2) * dt) / (sigma * sqrt(dt))

def bs_price(S, T, r, sigma, K, t):
    dt = T - t
    d1 = bs_d1(S, dt, r, sigma, K)
    d2 = d1 - sigma * sqrt(dt)
    return S * phi(d1) - K * exp(-r * dt) * phi(d2)

def bs_delta(S, T, r, sigma, K, t):
    dt = T - t
    d1 = bs_d1(S, dt, r, sigma, K)
    return phi(d1)

def mc_paths(s0, T, sigma, r, n_sims, n_steps):
    rv = np.random.randn(n_sims, n_steps)
    dt = T / n_steps
    sT = s0 * cumprod(exp((r - sigma**2 / 2) * dt + sigma * sqrt(dt) * rv), axis=1)
    return reshape(transpose(np.c_[ones(n_sims) * s0, sT]), (n_steps + 1, n_sims))

N = 5         # time disrectization
s0 = 49.      # initial value of the asset
K = 50.       # strike for the call option 
T = 20. / 52  # maturity
sigma = 0.2   # volatility
premium = 3   # option premium
r = 0.05
n_sim = 10 ** 6

def pnl_delta(deltas, paths, K, price, alpha):
    ds = paths[1:, :] - paths[:-1, :]
    hedge = np.sum(deltas * ds, axis=0)
    payoff = maximum(paths[-1, :] - K, 0)
    pnls = -payoff + hedge + price
    return pnls

def pnl_delta_bs(s0, K, r, sigma, T, paths, alpha):
    times = zeros(paths.shape[0])
    times[1:] = T / (paths.shape[0] - 1)
    times = cumsum(times)
    bs_deltas = zeros((paths.shape[0] - 1, paths.shape[1]))
    for i in range(paths.shape[0] - 1):
        t = times[i]
        bs_deltas[i, :] = bs_delta(paths[i, :], T, r, sigma, K, t)
    return pnl_delta(bs_deltas, paths, K, bs_price(s0, T, r, sigma, K, 0), alpha)

table = []
for N in [4, 5, 10, 20, 40, 80]:
    bs = bs_price(s0, T, r, sigma, K, 0)
    paths = mc_paths(s0, T, sigma, r, n_sim, N)
    pnl_stop_loss = zeros((n_sim, 1))
    pnl_covered = zeros((n_sim, 1))
    pnl_naked = zeros((n_sim, 1))
    epsilon = 1e-2
    for i in range(n_sim):
        position, cost = 0, 0
        for price in paths[:, i]:
            if price >= K + epsilon:
                if position == 0:
                    position = 1
                    cost += price
            elif price <= K - epsilon:
                if position == 1:
                    position = 0
                    cost -= price
        final = paths[-1, i]
        pnl_stop_loss[i] = premium - cost + (K if final >= K else (final if position else 0))
        pnl_covered[i] = premium - s0 + (K if final >= K else final)
        pnl_naked[i] = premium - maximum(final - K, 0)
        
    pnl_bs = pnl_delta_bs(s0, K, r, sigma, T, paths, 0.5)
    table.append((20. / N, std(pnl_bs) / bs, mean(pnl_bs), std(pnl_stop_loss) / bs, mean(pnl_stop_loss), std(pnl_covered) / bs, mean(pnl_covered), std(pnl_naked) / bs, mean(pnl_naked)))
