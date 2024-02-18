import math

import math

def european_call_price(S, K, r, T, sigma):
    """
    Calculates the price of a European call option using the Black-Scholes-Merton model.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    r (float): Risk-free interest rate
    T (float): Time to expiration of the option (in years)
    sigma (float): Volatility of the underlying asset. Can be implied or historical, with appropriate interpretation of results.

    Returns:
    float: Price of the European call option
    """
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call_price

def european_put_price(S, K, r, T, sigma):
    """
    Calculates the price of a European put option using the Black-Scholes-Merton model.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    r (float): The risk-free interest rate.
    T (float): The time to expiration of the option in years.
    sigma (float): The volatility of the underlying asset. Can be implied or historical, with appropriate interpretation of results.

    Returns:
    float: The price of the European put option.
    """
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    put_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return put_price

def norm_cdf(x):
    """
    Calculates the cumulative distribution function (CDF) of the standard normal distribution.

    Parameters:
    x (float): The value at which to evaluate the CDF.

    Returns:
    float: The CDF value at x.
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


if __name__=='__main__':
    S = float(input("Enter the current price of the underlying asset: "))
    K = float(input("Enter the strike price of the option: "))
    r = float(input("Enter the risk-free interest rate: "))
    T = float(input("Enter the time to expiration of the option in years: "))
    sigma = float(input("Enter the volatility of the underlying asset: "))
    print('European call option price:', european_call_price(S, K, r, T, sigma))
    print('European put option price:', european_put_price(S, K, r, T, sigma))
