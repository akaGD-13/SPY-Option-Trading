import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load IV data
file_path = "data/dataset_guangda_fei.xlsx"
df_iv = pd.read_excel(file_path, sheet_name="IV", index_col=0)
df_iv.index = pd.to_datetime(df_iv.index)
df_iv.columns = [col.strip() for col in df_iv.columns]

# Select calibration date: April 9, 2025
cal_date = pd.to_datetime("2025-04-09")
row = df_iv.loc[cal_date]

# Extract SPY and volatilities (convert % to decimals)
S0 = row["SPY Close"]
sigma_atm = row["ATM Vol 1m"] / 100
sigma_put0 = row["95% Moneyness Vol 1m"] / 100
sigma_call0 = row["105% Moneyness Vol 1m"] / 100

# Heston parameters (calibrated to ATM)
v0 = 0.060282
r = 0.0442
kappa = 1.5       # mean reversion speed
theta = v0        # long-term variance
sigma_v = 0.3     # vol-of-vol
rho = -0.7        # correlation

# Simulation settings
M = 10000
trading_days = 30 # days to 2025-05-09
dt = 1 / 252

# Initialize arrays
S_paths = np.full(M, S0)
v_paths = np.full(M, v0)

# Monte Carlo under Heston
for _ in range(trading_days):
    z1 = np.random.normal(size=M)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=M)
    # variance update
    v_paths = np.clip(
        v_paths + kappa * (theta - v_paths) * dt
        + sigma_v * np.sqrt(np.maximum(v_paths, 0) * dt) * z2,
        1e-8, None
    )
    # spot update
    S_paths *= np.exp((r - 0.5 * v_paths) * dt + np.sqrt(v_paths * dt) * z1)

# Summarize distribution
summary = pd.DataFrame({
    "Statistic": ["Mean", "Median", "5th %ile", "95th %ile"],
    "SPY Price": [
        np.mean(S_paths),
        np.median(S_paths),
        np.percentile(S_paths, 5),
        np.percentile(S_paths, 95)
    ]
})

print(summary)

# Plot histogram
plt.figure(figsize=(8, 4))
plt.hist(S_paths, bins=50)
plt.title("SPY Price Distribution on 2025-05-09 (Heston)")
plt.xlabel("SPY Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
