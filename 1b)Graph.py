import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


# Black-Scholes and Greeks for European options
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
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)

    return price, delta, gamma, vega, theta, rho


# Parameters
S0 = 548.62
r = 0.0442
T_orig = 37 / 365  # original time to expiry
T_t1 = 7 / 365  # T-1: 1 week to expiry

K_put = 521
K_call = 576
put_price_market = 9.97
call_price_market = 4.37
sigma_put = 0.338875  # 95%
sigma_call = 0.180364  # 105%

# Initial capital from selling 1000 puts
# put_price_initial, *_ = bs_price_and_greeks(S0, K_put, T_orig, r, sigma_put, 'put')
initial_cash = 1000 * 100 * put_price_market

# Call price at initiation
# call_price_initial, *_ = bs_price_and_greeks(S0, K_call, T_orig, r, sigma_call, 'call')

# Number of call contracts bought
call_contracts = int(initial_cash / (100 * call_price_market))

print('Put Position:', 1000)
print('Call Position:', call_contracts)
print('Cash Position:', initial_cash - call_contracts * 100 * call_price_market)

# At T-1 scenario
S_range = np.linspace(500, 600, 101)
pnl = []
put_pnl = []
call_pnl = []
greeks_summary = []

for S_t1 in S_range:
    put_t1, d_p, g_p, v_p, t_p, r_p = bs_price_and_greeks(S_t1, K_put, T_t1, r, sigma_put, 'put')
    call_t1, d_c, g_c, v_c, t_c, r_c = bs_price_and_greeks(S_t1, K_call, T_t1, r, sigma_call, 'call')

    position_value = -1000 * 100 * put_t1 + call_contracts * 100 * call_t1
    initial_cost = 0  # fully funded structure
    pnl.append(position_value - initial_cost)

    # Portfolio Greeks (weighted sums)
    delta = -1000 * 100 * d_p + call_contracts * 100 * d_c
    gamma = -1000 * 100 * g_p + call_contracts * 100 * g_c
    vega = -1000 * 100 * v_p + call_contracts * 100 * v_c
    theta = -1000 * 100 * t_p + call_contracts * 100 * t_c
    rho = -1000 * 100 * r_p + call_contracts * 100 * r_c

    greeks_summary.append([S_t1, delta, gamma, vega, theta, rho])
    put_pnl.append(-100 * (put_t1 - put_price_market))  # single put, shorted
    call_pnl.append(100 * (call_t1 - call_price_market))  # single call, long


# Find breakeven point (closest to zero PnL)
breakeven_index = np.argmin(np.abs(np.array(pnl)))
breakeven_spot = S_range[breakeven_index]
breakeven_pnl = pnl[breakeven_index]

# Create dual-axis plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left axis: Portfolio
ax1.plot(S_range, pnl, label='Portfolio PnL (1000 puts + {} calls)'.format(call_contracts), color='tab:blue')
ax1.annotate(f'Breakeven\n{breakeven_spot:.2f}',
             xy=(breakeven_spot, breakeven_pnl),
             xytext=(breakeven_spot-10, breakeven_pnl+200000),
             arrowprops=dict(arrowstyle='->', color='red'),
             color='red')
ax1.set_xlabel('SPY Spot Price at T-1')
ax1.set_ylabel('Portfolio PnL ($)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.axhline(0, color='gray', linestyle='--')

# Right axis: Single Put and Call
ax2 = ax1.twinx()
ax2.plot(S_range, put_pnl, label='Single Short Put PnL (K=521)', linestyle='--', color='tab:red')
ax2.plot(S_range, call_pnl, label='Single Long Call PnL (K=576)', linestyle='--', color='tab:green')
ax2.set_ylabel('Single Option PnL ($)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('PnL at T-1 (1 Week Before Expiry)')
plt.grid(True)
plt.tight_layout()
plt.show()


# Show Greeks summary
# Calculate the Greeks for the portfolio as of 04/09/2025 (initial trade date)

# Greeks for the 95% put (short)
_, delta_p, gamma_p, vega_p, theta_p, rho_p = bs_price_and_greeks(
    S0, K_put, T_orig, r, sigma_put, 'put'
)

# Greeks for the 105% call (long)
_, delta_c, gamma_c, vega_c, theta_c, rho_c = bs_price_and_greeks(
    S0, K_call, T_orig, r, sigma_call, 'call'
)

# Portfolio Greeks
greeks_today = {
    'Delta': -1000 * 100 * delta_p + call_contracts * 100 * delta_c,
    'Gamma': -1000 * 100 * gamma_p + call_contracts * 100 * gamma_c,
    'Vega':  -1000 * 100 * vega_p + call_contracts * 100 * vega_c,
    'Theta': -1000 * 100 * theta_p + call_contracts * 100 * theta_c,
    'Rho':   -1000 * 100 * rho_p + call_contracts * 100 * rho_c,
    'Calls Bought': call_contracts,
    'Cash Raised from Put Sale': initial_cash
}

greeks_today_df = pd.DataFrame(greeks_today, index=[0])
print(np.round(greeks_today_df.iloc[0, :]))

# # greeks at T-1
# greeks_df = pd.DataFrame(greeks_summary, columns=['Spot', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])
# print(greeks_df)
