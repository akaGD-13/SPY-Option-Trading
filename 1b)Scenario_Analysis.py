import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

# Black-Scholes pricing function
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Parameters
S0 = 548.62
K_put = 521
K_call = 576
r = 0.0442
T_eval = 7 / 365
put_price_market = 9.97
call_price_market = 4.37
n_put_contracts = 1000

# Range definitions
spot_range = np.linspace(450, 650, 201)
iv_put_range = np.linspace(0.15, 0.5, 36)
iv_call_range = iv_put_range - 0.08  # skew

# Trade setup
cash_raised = n_put_contracts * 100 * put_price_market
n_call_contracts = int(cash_raised / (100 * call_price_market))

# Meshgrid for 3D PnL surface
S_grid, iv_grid = np.meshgrid(spot_range, iv_put_range)
pnl_surface = np.zeros_like(S_grid)

for i in range(S_grid.shape[0]):
    for j in range(S_grid.shape[1]):
        S = S_grid[i, j]
        sigma_p = iv_put_range[i]
        sigma_c = iv_call_range[i]

        put_val = bs_price(S, K_put, T_eval, r, sigma_p, 'put')
        call_val = bs_price(S, K_call, T_eval, r, sigma_c, 'call')

        portfolio_val = -n_put_contracts * 100 * put_val + n_call_contracts * 100 * call_val
        pnl_surface[i, j] = portfolio_val

# 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(iv_grid, S_grid, pnl_surface, cmap='viridis')
ax.set_title('Portfolio PnL vs SPY Spot and Implied Volatility')
ax.set_xlabel('Put IV')
ax.set_ylabel('SPY Spot Price')
ax.set_zlabel('Portfolio PnL ($)')
plt.tight_layout()
plt.show()
