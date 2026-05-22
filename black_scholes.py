import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# inputs and data
company = {"Apple Inc":"AAPL"}
s_price = yf.download(list(company.values())[0], period="max", auto_adjust=True)["Close"].squeeze()
log_returns = np.log(s_price / s_price.shift(1)).dropna()
std = float(np.std(log_returns) * np.sqrt(252))

X = float(s_price.iloc[-1])
S = np.linspace(X * 0.75, X * 1.25, 100)
r = 0.05    # fixed rate from fixed income 
T = 1       # change to user input

# define formulas for option prices and greeks
def compute_d1_d2(S, X, r, T, std):
    d1 = (np.log(S/X) + (r + (std**2)/2) * T) / (std * np.sqrt(T))
    d2 = d1 - std * np.sqrt(T)
    return d1, d2

def N(x):
    return norm.cdf(x)

def n(x):
    return norm.pdf(x)

def compute_call_put(S, X, r, T, std):
    d1, d2 = compute_d1_d2(S, X, r, T, std)
    call = S * N(d1) - X * np.exp(-r*T) * N(d2)
    put = X * np.exp(-r*T) * N(-d2) - S * N(-d1)
    return call, put

def compute_greeks(S,X,r,T,std):
    d1, d2 = compute_d1_d2(S,X,r,T,std)
    
    delta_call = N(d1)
    delta_put = -N(-d1)

    theta_call = (-(S * n(d1) * std)/(2*np.sqrt(T))) - (r * X * (np.e**(-r*T)) * N(d2))
    theta_put = (-(S * n(d1) * std)/(2*np.sqrt(T))) + (r * X * (np.e**(-r*T)) * N(-d2))

    gamma = n(d1)/(S*std*np.sqrt(T))

    vega = S*np.sqrt(T)*n(d1)

    rho_call = X * T * np.e**(-r*T) * N(d2)
    rho_put = -X * T * np.e**(-r*T) * N(-d2)

    return {
        "delta_call"    : delta_call.round(2),
        "delta_put"     : delta_put.round(2),
        "theta_call"    : theta_call.round(2),
        "theta_put"     : theta_put.round(2),
        "gamma"         : gamma.round(2),
        "vega"          : vega.round(2),
        "rho_call"      : rho_call.round(2),
        "rho_put"       : rho_put.round(2),
    }

# calculate option prices, verify put-call parity, calculate greeks for different stock prices
results = {
    "stock_price": [],
    "call_price": [],
    "put_price": [],
    "delta_call": [],
    "delta_put": [],
    "theta_call": [],
    "theta_put": [],
    "gamma": [],
    "vega": [],
    "rho_call": [],
    "rho_put": []
    }

for S in S:
    results["stock_price"].append(S.round(2))
    
    call_price, put_price = compute_call_put(S, X, r, T, std)

    parity_check = S - X * np.exp(-r*T)
    if np.isclose(call_price - put_price, parity_check) == False:
        print(f"Parity difference: {call_price - put_price - parity_check} (stock: {S})")
        continue

    results["call_price"].append(call_price.round(2))
    results["put_price"].append(put_price.round(2))

    greeks = compute_greeks(S,X,r,T,std)

    results["delta_call"].append(greeks["delta_call"])
    results["delta_put"].append(greeks["delta_put"])
    results["theta_call"].append(greeks["theta_call"])
    results["theta_put"].append(greeks["theta_put"])
    results["gamma"].append(greeks["gamma"])
    results["vega"].append(greeks["vega"])
    results["rho_call"].append(greeks["rho_call"])
    results["rho_put"].append(greeks["rho_put"])

results = pd.DataFrame(results)