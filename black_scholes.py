import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# inputs and data
s_price = yf.download("AAPL", period="max", auto_adjust=True)["Close"].squeeze()
log_returns = np.log(s_price / s_price.shift(1)).dropna()
std = float(np.std(log_returns) * np.sqrt(252))

S = float(s_price.iloc[-1])
X = 320     # change to user input
r = 0.05    # fixed rate from fixed income 
T = 1       # change to user input

# calculate call and put prices
def compute_d1_d2(S, X, r, T, std):
    d1 = (np.log(S/X) + (r + (std**2)/2) * T) / (std * np.sqrt(T))
    d2 = d1 - std * np.sqrt(T)
    return d1, d2

def N(x):
    return norm.cdf(x)

def compute_call_put(S, X, r, T, std):
    d1, d2 = compute_d1_d2(S, X, r, T, std)
    call = S * N(d1) - X * np.exp(-r*T) * N(d2)
    put = X * np.exp(-r*T) * N(-d2) - S * N(-d1)
    return call, put

call_price, put_price = compute_call_put(S, X, r, T, std)

# call-put parity check
parity_check = S - X * np.exp(-r*T)
if np.isclose(call_price - put_price, parity_check):
    print(f"Put-call parity holds")
else:
    print(f"Parity difference: {call_price - put_price - parity_check}")

print(f"Call: {call_price:.2f}, Put: {put_price:.2f}")