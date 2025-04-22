import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# Black-Scholes pricing and Greeks
def bs_price_and_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    return price, delta, gamma, vega, theta, rho


# Implied volatility solver
def implied_vol(S, K, T, r, market_price, option_type='put'):
    def objective(sigma):
        return bs_price_and_greeks(S, K, T, r, sigma, option_type)[0] - market_price

    return brentq(objective, 1e-6, 5.0)


# Parameters
S0 = 548.62
r = 0.0442
T_orig = 37 / 365
T_t1 = 7 / 365

# Original trade strikes and market data
K_put = 521
put_price_market = 9.97
sigma_put = implied_vol(S0, K_put, T_orig, r, put_price_market, 'put')
K_call = 576
call_price_market = 4.37
sigma_call = implied_vol(S0, K_call, T_orig, r, call_price_market, 'call')

# New deep OTM put (90%)
K_put2 = 494
put2_price_market = 5.54
sigma_put2 = implied_vol(S0, K_put2, T_orig, r, put2_price_market, 'put')

# Build original portfolio
n_put = 1000
cash = n_put * 100 * put_price_market
n_call = int(cash / (100 * call_price_market))

# Build put-spread portfolio
n_put2 = n_put
spread_credit = (put_price_market - put2_price_market) * 100 * n_put
n_call_spread = int(spread_credit / (100 * call_price_market))

# Scenario grid
S_range = np.linspace(480, 600, 121)
pnl_orig, pnl_spread = [], []
greeks_orig, greeks_spread = [], []

for S in S_range:
    # T-1 values
    p_price, dp, gp, vp, tp, rp = bs_price_and_greeks(S, K_put, T_t1, r, sigma_put, 'put')
    c_price, dc, gc, vc, tc, rc = bs_price_and_greeks(S, K_call, T_t1, r, sigma_call, 'call')
    p2_price, d2p, g2p, v2p, t2p, r2p = bs_price_and_greeks(S, K_put2, T_t1, r, sigma_put2, 'put')

    # PnL calculations
    orig_val = -n_put * 100 * p_price + n_call * 100 * c_price
    spread_val = (-n_put * 100 * p_price + n_put2 * 100 * p2_price
                  + n_call_spread * 100 * c_price)
    pnl_orig.append(orig_val)
    pnl_spread.append(spread_val)

    # Greeks at T-1
    greeks_orig.append([
        -n_put * 100 * dp + n_call * 100 * dc,
        -n_put * 100 * gp + n_call * 100 * gc,
        -n_put * 100 * vp + n_call * 100 * vc,
        -n_put * 100 * tp + n_call * 100 * tc,
        -n_put * 100 * rp + n_call * 100 * rc
    ])
    greeks_spread.append([
        -n_put * 100 * dp + n_put2 * 100 * d2p + n_call_spread * 100 * dc,
        -n_put * 100 * gp + n_put2 * 100 * g2p + n_call_spread * 100 * gc,
        -n_put * 100 * vp + n_put2 * 100 * v2p + n_call_spread * 100 * vc,
        -n_put * 100 * tp + n_put2 * 100 * t2p + n_call_spread * 100 * tc,
        -n_put * 100 * rp + n_put2 * 100 * r2p + n_call_spread * 100 * rc
    ])

# Plot PnL
plt.figure(figsize=(10, 6))
plt.plot(S_range, pnl_orig, label=f'Original (1000 P{K_put}, {n_call} C{K_call})')
plt.plot(S_range, pnl_spread, label=(f'Put-Spread (sell P{K_put}, buy P{K_put2}), '
                                     f'{n_call_spread} C{K_call})'))
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('SPY Spot Price at T-1')
plt.ylabel('PnL ($)')
plt.title('T-1 Portfolio PnL: Original vs Put-Spread')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display Greeks at inception
# Original
_, d_po, g_po, v_po, t_po, r_po = bs_price_and_greeks(S0, K_put, T_orig, r, sigma_put, 'put')
_, d_co, g_co, v_co, t_co, r_co = bs_price_and_greeks(S0, K_call, T_orig, r, sigma_call, 'call')
greeks_orig_init = {
    'Delta': -n_put * 100 * d_po + n_call * 100 * d_co,
    'Gamma': -n_put * 100 * g_po + n_call * 100 * g_co,
    'Vega': -n_put * 100 * v_po + n_call * 100 * v_co,
    'Theta': -n_put * 100 * t_po + n_call * 100 * t_co,
    'Rho': -n_put * 100 * r_po + n_call * 100 * r_co
}
# Spread
_, d_p2, g_p2, v_p2, t_p2, r_p2 = bs_price_and_greeks(S0, K_put2, T_orig, r, sigma_put2, 'put')
greeks_spread_init = {
    'Delta': -n_put * 100 * d_po + n_put2 * 100 * d_p2 + n_call_spread * 100 * d_co,
    'Gamma': -n_put * 100 * g_po + n_put2 * 100 * g_p2 + n_call_spread * 100 * g_co,
    'Vega': -n_put * 100 * v_po + n_put2 * 100 * v_p2 + n_call_spread * 100 * v_co,
    'Theta': -n_put * 100 * t_po + n_put2 * 100 * t_p2 + n_call_spread * 100 * t_co,
    'Rho': -n_put * 100 * r_po + n_put2 * 100 * r_p2 + n_call_spread * 100 * r_co
}

df_greeks = pd.DataFrame({
    'Greek': list(greeks_orig_init.keys()),
    'Original': list(greeks_orig_init.values()),
    'Spread': list(greeks_spread_init.values())
})

print(df_greeks)
