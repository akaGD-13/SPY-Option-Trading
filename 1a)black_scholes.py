# price at 4/9/2025: 548.62
# 95%: 521.189 --> 521, SPY US 05/16/2025 C521 = 33.8, SPY US 05/16/25 P521 = 9.97
# 105%: 576.051 --> 576, SPY US 05/16/2025 C576 = 4.37, SPY US 05/16/25 P576 = 36.31
# SOFR: 4.42%

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def implied_rate_put_call_parity(S, K, T, call_price, put_price):
    """
    Compute implied risk-free rate r from put-call parity.

    Parameters:
        S (float): Spot price of the underlying
        K (float): Strike price
        T (float): Time to maturity (in years)
        call_price (float): Market price of call option
        put_price (float): Market price of put option

    Returns:
        r (float): Implied risk-free rate
    """
    parity_value = S - (call_price - put_price)
    if parity_value <= 0:
        raise ValueError("Invalid prices: parity condition violated (S - (C - P) <= 0)")
    return -np.log(parity_value / K) / T


def implied_rate_from_price(S, K, T, sigma, market_price, option_type='call'):
    """
    Solves for the implied risk-free rate r given market price of the option.

    Parameters:
        S (float): Spot price of the underlying
        K (float): Strike price
        T (float): Time to maturity in years
        sigma (float): Volatility (annualized)
        market_price (float): Observed market price of the option
        option_type (str): 'call' or 'put'

    Returns:
        r (float): Implied risk-free rate (annualized)
    """

    def objective(r):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price

    # Try solving for r in a reasonable range: [-10%, 20%]
    return brentq(objective, -0.20, 0.50)


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes formula for European option pricing.

    Parameters:
        S (float): Spot price of the underlying
        K (float): Strike price of the option
        T (float): Time to maturity in years
        r (float): Annualized risk-free interest rate (decimal)
        sigma (float): Annualized volatility (decimal)
        option_type (str): 'call' or 'put'

    Returns:
        price (float): Theoretical option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


# Inputs
S = 548.62               # SPY spot
K_95 = 521              # 95% moneyness put
K_105 = 576             # 105% moneyness call
r = 0.0442               # SOFR (annual)
T = 37 / 365             # Time to maturity = May 16 - Apr 9 ≈ 37 days
# sigma = 0.20
sigma_95 = 0.338875
sigma_105 = 0.180364

# Prices
put_price = black_scholes(S, K_95, T, r, sigma_95, option_type='put')
call_price = black_scholes(S, K_95, T, r, sigma_95, option_type='call')

print(f"Put (K={K_95}) Price: ${put_price:.2f}")
print(f"Call (K={K_95}) Price: ${call_price:.2f}")
# # Solve for implied risk-free rate
# r_put = implied_rate_from_price(S, K_95, T, sigma_95, 9.97, option_type='put')
# print(f"Implied risk-free rate from 95 put: {r_put:.4%}")
# r_call = implied_rate_from_price(S, K_95, T, sigma_95, 33.8, option_type='call')
# print(f"Implied risk-free rate from 95 call: {r_call:.4%}")
r_parity = implied_rate_put_call_parity(S, K_95, T, 33.8, 9.97)
print(f"Implied r from put-call parity: {r_parity:.4%}")


put_price = black_scholes(S, K_105, T, r, sigma_105, option_type='put')
call_price = black_scholes(S, K_105, T, r, sigma_105, option_type='call')
print(f"Put (K={K_105}) Price: ${put_price:.2f}")
print(f"Call (K={K_105}) Price: ${call_price:.2f}")
# # Solve for implied risk-free rate
# r_put = implied_rate_from_price(S, K_105, T, sigma_105, 36.31, option_type='put')
# print(f"Implied risk-free rate from 95 put: {r_put:.4%}")
# r_call = implied_rate_from_price(S, K_105, T, sigma_105, 4.37, option_type='call')
# print(f"Implied risk-free rate from 95 call: {r_call:.4%}")
r_parity = implied_rate_put_call_parity(S, K_105, T, 4.37, 36.31)
print(f"Implied r from put-call parity: {r_parity:.4%}")
