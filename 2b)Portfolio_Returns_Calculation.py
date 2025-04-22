import numpy as np
from scipy.stats import norm

# Black-Scholes pricing function
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Given data
S_mean = 544.82
S_mean = 467.73
# S_mean = 617.34
K_put = 521
K_call = 576
T_pred = 7/365
r = 0.0442

# Predicted IVs (in decimal form)
sigma_put_pred = 0.317321
sigma_call_pred = 0.175157

# Market prices at initiation
put_price_init = 9.97
call_price_init = 4.37

# Initial cash from selling 1000 puts
initial_cash = 1000 * 100 * put_price_init

# Number of call contracts bought
call_contracts = int(initial_cash / (100 * call_price_init))

# Value at T-1
put_value_pred = bs_price(S_mean, K_put, T_pred, r, sigma_put_pred, 'put')
call_value_pred = bs_price(S_mean, K_call, T_pred, r, sigma_call_pred, 'call')

put_position_val = -1000 * 100 * put_value_pred
call_position_val = call_contracts * 100 * call_value_pred

portfolio_value = put_position_val + call_position_val
portfolio_return = portfolio_value / initial_cash

# Display results
import pandas as pd

result_df = pd.DataFrame({
    'Metric': ['Put Value ($)', 'Call Value ($)', 'Portfolio Value ($)', 'Portfolio Return'],
    'Value': [put_value_pred, call_value_pred, portfolio_value, portfolio_return]
})

print(result_df)
