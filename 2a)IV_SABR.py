import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Load IV surface data
file_path = "data/dataset_guangda_fei.xlsx"
df_iv = pd.read_excel(file_path, sheet_name="IV", index_col=0)
df_iv.index = pd.to_datetime(df_iv.index)
df_iv.columns = [col.strip() for col in df_iv.columns]

# Extract market data on 2025-04-09
row = df_iv.loc[pd.to_datetime("2025-04-09")]
F = row["SPY Close"] * np.exp(0.0442 * (37/365))  # forward approx
T_full = 37/365
K_list = [521, row["SPY Close"], 576]
iv_list = [
    row["95% Moneyness Vol 1m"] / 100,
    row["ATM Vol 1m"] / 100,
    row["105% Moneyness Vol 1m"] / 100
]

# SABR Hagan formula
def sabr_hagan_vol(F, K, T, alpha, beta, rho, nu):
    if F == K:
        return alpha / (F**(1 - beta)) * (1 + T * ((1 - beta)**2 * alpha**2 / (24 * F**(2 - 2*beta))
                                                + rho * beta * nu * alpha / (4 * F**(1 - beta))
                                                + (2 - 3*rho**2) * nu**2 / 24))
    z = (nu / alpha) * (F * K)**((1 - beta)/2) * np.log(F/K)
    x_z = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho))
    pre = alpha / ((F * K)**((1 - beta)/2))
    return pre * (z / x_z) * (1 + T * ((1 - beta)**2 * alpha**2 / (24 * (F*K)**(1 - beta))
                                       + rho * beta * nu * alpha / (4 * (F*K)**((1 - beta)/2))
                                       + (2 - 3*rho**2) * nu**2 / 24))

# Calibration function
def calibrate_sabr(F, strikes, ivs, T, beta=0.5):
    def resid(params):
        a, r_, nu = params
        return [sabr_hagan_vol(F, K, T, a, beta, r_, nu) - iv for K, iv in zip(strikes, ivs)]
    x0 = [ivs[1], 0.0, 0.5]
    sol = least_squares(resid, x0, bounds=([1e-6, -0.99, 1e-6], [5, 0.99, 5]))
    return sol.x  # alpha, rho, nu

# 1) Calibrate SABR on 04/09 (T_full)
alpha, rho, nu = calibrate_sabr(F, K_list, iv_list, T_full, beta=0.5)

# 2) Predict IV at T-1 (05/09), T_pred = 7/365
T_pred = 7/365
iv_pred = [sabr_hagan_vol(F, K, T_pred, alpha, 0.5, rho, nu) for K in K_list]

# Build result table
result = pd.DataFrame({
    "Strike": K_list,
    "Predicted IV on 2025-05-09": iv_pred
})

print(result)
