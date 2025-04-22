import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares, brentq
from scipy.stats import norm

# 1. Load your IV data
df_iv = pd.read_excel("data/dataset_guangda_fei.xlsx", sheet_name="IV", index_col=0)
df_iv.index = pd.to_datetime(df_iv.index)
df_iv.columns = [c.strip() for c in df_iv.columns]

# 2. Extract SPY spot and IVs on 2025-04-09
row = df_iv.loc["2025-04-09"]
S0 = row["SPY Close"]
r = 0.0442
q = 0.0
T = 37/365

K_list = [0.95*S0, S0, 1.05*S0]
iv_market = [
    row["95% Moneyness Vol 1m"]/100,
    row["ATM Vol 1m"]/100,
    row["105% Moneyness Vol 1m"]/100
]
v0 = iv_market[1]**2   # initial variance from ATM

# 3. Heston characteristic function
def heston_cf(phi, j, params):
    kappa, theta, sigma_v, rho, v0 = params
    a = kappa*theta
    b = kappa - rho*sigma_v if j==1 else kappa
    d = np.sqrt((rho*sigma_v*phi*1j - b)**2 + sigma_v**2*(phi*1j+phi**2))
    g = (b - rho*sigma_v*phi*1j + d)/(b - rho*sigma_v*phi*1j - d)
    exp_dT = np.exp(-d*T)
    C = (r-q)*phi*1j*T + a/sigma_v**2*((b - rho*sigma_v*phi*1j + d)*T
        - 2*np.log((1-g*exp_dT)/(1-g)))
    D = (b - rho*sigma_v*phi*1j + d)/sigma_v**2 * (1-exp_dT)/(1-g*exp_dT)
    return np.exp(C + D*v0 + 1j*phi*np.log(S0*np.exp(-q*T)))

# 4. Heston price via integration
def heston_price(K, params):
    def integrand(phi, j):
        return np.real(
            np.exp(-1j*phi*np.log(K)) *
            heston_cf(phi - (j-1)*1j, j, params) /
            (1j*phi)
        )
    P1 = 0.5 + 1/np.pi*quad(lambda x: integrand(x,1), 0, 100)[0]
    P2 = 0.5 + 1/np.pi*quad(lambda x: integrand(x,2), 0, 100)[0]
    return S0*np.exp(-q*T)*P1 - K*np.exp(-r*T)*P2

# 5. Implied vol from model price
def implied_vol(price, S, K, T, r, q, kind='call'):
    def bs_diff(vol):
        d1 = (np.log(S/K)+(r-q+0.5*vol**2)*T)/(vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        if kind=='call':
            bs = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            bs = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        return bs - price

    # endpoint values
    low, high = 1e-6, 5.0
    f_low, f_high = bs_diff(low), bs_diff(high)

    # if price is outside the BS bounds, clamp
    if f_low < 0 and f_high < 0:
        # model price is *higher* than any BS price → take high vol
        return high
    if f_low > 0 and f_high > 0:
        # model price is *lower* than any BS price → take low vol
        return low

    # otherwise we have a sign change and can root-find
    return brentq(bs_diff, low, high)

# 6. Calibration residuals
def residuals(x):
    kappa, theta, sigma_v, rho = x
    params = [kappa, theta, sigma_v, rho, v0]
    errs = []
    for K, iv_m in zip(K_list, iv_market):
        price = heston_price(K, params)
        # pass all required args here:
        iv_mod = implied_vol(price, S0, K, T, r, q, kind='call')
        errs.append(iv_mod - iv_m)
    return errs

# 7. Run calibration
x0 = [1.5, v0, 0.3, -0.7]  # initial guess
bounds = ([0.01,1e-6,0.01,-0.99], [10,1.0,2.0,0.99])
sol = least_squares(residuals, x0, bounds=bounds)

kappa, theta, sigma_v, rho = sol.x
print("Calibrated Heston parameters:")
print(f" kappa  = {kappa:.4f}")
print(f" theta  = {theta:.6f}")
print(f" sigma_v= {sigma_v:.4f}")
print(f" rho    = {rho:.4f}")
print(f" v0     = {v0:.6f}")
