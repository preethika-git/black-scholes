# Black-Scholes Options Pricer

European call and put option pricing using the Black-Scholes model, with full Greeks and implied volatility solving.

## Features
- Option price calculation (call & put)
- Greeks: Delta, Gamma, Vega, Theta, Rho
- Implied volatility solver (Newton-Raphson)
- Put-call parity verification
- 6 comprehensive visualization charts in one output
- Historical volatility calculation from market data

## Inputs
Edit these in the code:
- `company`: Stock ticker (default: AAPL)
- `X`: Strike price
- `M`: Market option price
- `expiry_date`: Option expiry date (YYYY-MM-DD)
- `r`: Risk-free rate (default: 0.05)

## Outputs
- `outputs/greeks.csv` — All Greeks and prices across spot price range
- `outputs/greeks_chart.png` — 6 charts showing Greeks vs spot price
- Console summary with current Greeks at market price
- Implied volatility calculation with verification

## How to Use
```python
X = 295                     # Set strike
M = 7.45                    # Set market price
expiry_date = "2026-07-02"  # Set expiry

python black_scholes.py
```

## Dependencies
- numpy
- pandas
- yfinance
- scipy
- matplotlib

## Methodology
Prices European options using Black-Scholes closed-form solution. Greeks computed analytically. IV solved using Newton-Raphson method with Brenner-Subrahmanyam initial guess.