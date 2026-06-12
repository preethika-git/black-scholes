import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt

# inputs and data

company = {"Apple Inc":"AAPL"}
s_price = yf.download(list(company.values())[0], period="max", auto_adjust=True)["Close"].squeeze()
log_returns = np.log(s_price / s_price.shift(1)).dropna()
std = float(np.std(log_returns) * np.sqrt(252))

S = float(s_price.iloc[-1])
S_range = np.linspace(int(S) * 0.75, int(S) * 1.25, 100)
r = 0.05

# user input for the following from the same option 

X = 307.5                   # strike price
M = 3.41                    # market price    
expiry_date = "2026-06-08"  # expiry date 

today = pd.Timestamp.now()
T = (pd.Timestamp(expiry_date) - today).days / 365

# define for Black-Scoles pricing and Greeks

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
        "delta_call"    : delta_call,
        "delta_put"     : delta_put,
        "theta_call"    : theta_call,
        "theta_put"     : theta_put,
        "gamma"         : gamma,
        "vega"          : vega,
        "rho_call"      : rho_call,
        "rho_put"       : rho_put,
    }

# calculate option prices and greeks    

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

for stk in S_range:
    results["stock_price"].append(stk.round(2))
    
    call_price, put_price = compute_call_put(stk, X, r, T, std)

    parity_check = stk - X * np.exp(-r*T)
    if np.isclose(call_price - put_price, parity_check) == False:
        print(f"Parity difference: {call_price - put_price - parity_check} (stock: {stk})")
        break

    results["call_price"].append(call_price.round(2))
    results["put_price"].append(put_price.round(2))

    greeks = compute_greeks(stk,X,r,T,std)

    results["delta_call"].append(greeks["delta_call"].round(2))
    results["delta_put"].append(greeks["delta_put"].round(2))
    results["theta_call"].append(greeks["theta_call"].round(2))
    results["theta_put"].append(greeks["theta_put"].round(2))
    results["gamma"].append(greeks["gamma"])
    results["vega"].append(greeks["vega"].round(2))
    results["rho_call"].append(greeks["rho_call"].round(2))
    results["rho_put"].append(greeks["rho_put"].round(2))

results = pd.DataFrame(results)  

# export results to csv

results.to_csv("greeks.csv", index=False)

# define formulas for Implied Volatility solver

def iv_f(S, X, r, T, std):
    call_price, put_price = compute_call_put(S, X, r, T, std)
    return call_price - M

def iv_fdash(S,X,r,T,std):
    greeks = compute_greeks(S,X,r,T,std)
    return greeks["vega"]

def newton_rhapson(S,X,r,T,std):
    
    initial_guess = M/(S*np.sqrt(T)*0.4)  

    max_iterations = 100
    tolerance = 1e-6

    IV_old = initial_guess

    for i in range(max_iterations):
        IV_new = IV_old - (iv_f(S, X, r, T, std=IV_old)/iv_fdash(S, X, r, T, std=IV_old))
        if abs(IV_new-IV_old) < tolerance:
            return IV_new
        IV_old = IV_new
        
    print(f"IV solver did not converge in {max_iterations} iterations")
    return IV_new

# calculate IV and verify

IV = newton_rhapson(S,X,r,T,std)

verification_price, _ = compute_call_put(S, X, r, T, IV)
if not np.isclose(M, verification_price):
    print("Market Price and Recovered price from IV does not match")

# summary 

print("="*50)
print(f"Black-Scholes Option Pricing - {list(company.keys())[0]}")
print("-"*50)
print(f"Current Price: {S:.2f}")
print(f"Strike Price: {X:.2f}")
print(f"Days to Expiry: {T*365} days ({expiry_date})")
print(f"Risk-free Rate: {r:.2f}")
print(f"Option Market Price: {M}")
print(f"Historical Volatility: {std:.2%}")
print(f"Implied Volatility: {IV:.2%}")
print("="*50, end = "\n"*3)

# current greeks

current_greeks = compute_greeks(S,X,r,T,std)

print("="*50)
print(f"Greeks at Current Price (${S:.2f})")
print("-"*50)
for greek, value in current_greeks.items():
    print(f"{greek}: {value:.3f}")
print("="*50)

# plotting charts

fig, axes = plt.subplots(3,2, figsize=(18,20))
axes = axes.flatten()

plots = [
        ([(results["call_price"], "call", "b"),(results["put_price"], "put", "r")], 
        f"{list(company.keys())[0]}: Option Price vs Spot Price", "Option Price"),
        ([(results["delta_call"],"call","b"),(results["delta_put"],"put","r")],
        "Delta vs Spot","Delta"),
        ([(results["theta_call"],"call","b"),(results["theta_put"],"put","r")],
        "Theta vs Spot","Theta"),
        ([(results["gamma"],"","b")],
        "Gamma vs Spot","Gamma"),
        ([(results["vega"],"","b")],
        "Vega vs Spot","Vega"),
        ([(results["rho_call"],"call","b"),(results["rho_put"],"put","r")],
        "Rho vs Spot","Rho")
        ]


          
x_ticks = [i for i in range(int(((results["stock_price"].iloc[0]//10)*10)-10),
                            int(((results["stock_price"].iloc[-1]//10)*10)+20),10)]

for idx, (y_data, title, ylabel) in enumerate(plots):
    ax = axes[idx]

    for y, label, color in y_data:
        ax.plot(results["stock_price"], y, label = label, color = color)
    
    ax.axvline(x=X, linestyle="--", color="grey", lw = .5)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_ticks, labels=x_ticks, rotation = 45)
    ax.set_xlim(xmin = np.min(x_ticks)-10, xmax = np.max(x_ticks)+10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)


    if idx == 0:        # call put price vs spot
        y_min = min(results["call_price"].min(), results["put_price"].min())
        y_max = max(results["call_price"].max(), results["put_price"].max())
        y_ticks = [i for i in range(int(((y_min//10)*10)-10),
                                    int(((y_max//10)*10)+20),10)]

        ax.set_yticks(y_ticks, labels=y_ticks)
        ax.legend(loc = "center left")
    
    elif idx == 1:      # delta vs spot
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("Spot Price", loc="right", labelpad= -33)

        ax.set_yticks(np.arange(-1,1.1,0.2))
        ax.set_ylim(ymin=-1.1, ymax=1.1)

        ax.legend(loc = "upper left")

    elif idx == 2:      # theta vs spot
        ax.spines["bottom"].set_position("zero")

        t_min = min(results["theta_call"].min(), results["theta_put"].min())
        t_max = max(results["theta_call"].max(), results["theta_put"].max())
        t_ticks = np.arange(((t_min//15)*15)-10, 10, 15)
        ax.set_yticks(t_ticks)

        ax.set_xlabel("Spot Price", labelpad= -33)
        ax.legend(loc = "lower left")

    elif idx == 3:      # gamma vs spot
        g_ticks = np.arange(((min(results["gamma"])//0.001)*0.001), max(results["gamma"])+0.001, 0.001)
        g_labels = [f"{i*(10**3):.0f}e-3" for i in g_ticks]
        ax.set_yticks(g_ticks, labels=g_labels)
        ax.set_ylim(ymin=min(g_ticks)-0.00012, ymax=max(g_ticks)+0.00012)

    elif idx == 4:      # vega vs spot
        v_min, v_max = min(results["vega"]), max(results["vega"])
        v_ticks = np.arange(((v_min//5)*5)-5, ((v_max//5)*5)+10, 5)
        ax.set_yticks(v_ticks)
        ax.set_ylim(ymin=v_min-5, ymax=max(v_ticks)+3)

    else:               # rho vs spot
        ax.spines["bottom"].set_position("zero")

        r_min = min(min(results["rho_call"]), min(results["rho_put"]))
        r_max = max(max(results["rho_call"]), max(results["rho_put"]))
        r_ticks = np.arange(((r_min)//5)*5, r_max+5, 5)
        r_label = [f"{i:.0f}" for i in r_ticks]

        ax.set_yticks(r_ticks, labels=r_label)
        ax.set_ylim(ymin=r_min-5, ymax=r_max+5)

        ax.set_xlabel("Spot Price", labelpad= -33)
        ax.legend(loc = "upper left")

plt.show()