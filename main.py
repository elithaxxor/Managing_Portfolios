import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import traceback, time, math

from fontTools.misc.psLib import endofthingRE

'''
Define the Asset class, access the expected return and standard deviation of the asset
# I made this a class, so that I can easily access the expected return and standard deviation of the asset
'''


@dataclass
class Portofolio_Portion:
    proportion_stock_A: float
    proportion_stock_B: float
    proportion_stock_C: float

    def __init__(self, proportion_stock_A, proportion_stock_B, proportion_stock_C, total_proportion):
        self.proportion_stock_A = proportion_stock_A
        self.proportion_stock_B = proportion_stock_B
        self.proportion_stock_C = proportion_stock_C
        self.clients_weight_in_risky = clients_weight_in_risky
        self.client_investment_t_bills = client_investment_t_bills
        self.total_proportion = total_proportion

    def __str__(self):
        return f"proportion_stock_A: {self.proportion_stock_A}, proportion_stock_B: {self.proportion_stock_B}, proportion_stock_C: {self.proportion_stock_C}, total_proportion: {self.total_proportion}"

    def __repr__(self):
        return f"proportion_stock_A: {self.proportion_stock_A}, proportion_stock_B: {self.proportion_stock_B}, proportion_stock_C: {self.proportion_stock_C}, total_proportion: {self.total_proportion}"

    def proportion_setter(self, value):

        clients_weight_in_risky = float(input("Enter the client's weight in the risky portfolio (e.g., 0.70 for 70%): "))
        client_investment_t_bills = 1 - clients_weight_in_risky

        proportion_stock_A = float(input("Enter the proportion of Stock A (e.g., 0.40 for 40%): "))
        proportion_stock_B = float(input("Enter the proportion of Stock B (e.g., 0.30 for 30%): "))
        proportion_stock_C = float(input("Enter the proportion of Stock C (e.g., 0.30 for 30%): "))

        self.proportion_stock_A = proportion_stock_A
        self.proportion_stock_B = proportion_stock_B
        self.proportion_stock_C = proportion_stock_C
        self.clients_weight_in_risky = clients_weight_in_risky
        self.clients_inv_t_bills = client_investment_t_bills


    @property
    def total_proportion(self):
        total_proportion = self.proportion_stock_A + self.proportion_stock_B + self.proportion_stock_C
        self.total_proportion = total_proportion
        return self.total_proportion

    @property
    def weighted_average1(self):
        return self.proportion_stock_A * self.clients_weight_in_risky

    @property
    def weighted_average2(self):
        return self.proportion_stock_B * self.clients_weight_in_risky

    @property
    def weighted_average3(self):
        return self.proportion_stock_C * self.clients_weight_in_risky
    @property
    def get_proportion(self):
        return self.proportion_stock_A, self.proportion_stock_B, self.proportion_stock_C, self.total_proportion

    @total_proportion.setter
    def total_proportion(self, value):
        self._total_proportion = value


@dataclass # Dataclass to store the expected return and standard deviation of an asset
class Portfolio:
    expected_return: float
    std_deviation: float
    risky_portofolio_return: float
    risky_portfolio_std_dev: float
    risk_free_rate: float
    client_weight_in_risky: float
    total_proportion: float

def get_portfolio_inputs():
    """
    Function to get user input for the risky portfolio return, standard deviation,
    risk-free rate, and client's weight in the risky portfolio.

    Returns:
        tuple: Contains the risky portfolio return, risky portfolio standard deviation,
               risk-free rate, and client weight in the risky portfolio.
    """
    try:
        # Get user inputs
        risky_portfolio_return = float(input("Enter the expected return of the risky portfolio (e.g., 0.18 for 18%): "))
        risky_portfolio_std_dev = float(input("Enter the standard deviation of the risky portfolio (e.g., 0.28 for 28%): "))
        risk_free_rate = float(input("Enter the risk-free rate (e.g., 0.08 for 8%): "))
        client_weight_in_risky = float(input("Enter the client's weight in the risky portfolio (e.g., 0.70 for 70%): "))

        expected_return = float(input("Enter the Expected Return: "))
        std_deviat = float(input("Enter the Standard Deviation: "))

        # Get user inputs
        proportion_stock_A = float(input("Enter the proportion of Stock A (e.g., 0.40 for 40%): "))
        proportion_stock_B = float(input("Enter the proportion of Stock B (e.g., 0.30 for 30%): "))
        proportion_stock_C = float(input("Enter the proportion of Stock C (e.g., 0.30 for 30%): "))

        correlation = correlation_coeficant_calculation(asset_A, t_bills)

        weight_A, weight_B, expected_return, standard_deviation = find_optimal_risky_portfolio(expected_return, std_deviat, correlation, risky_portfolio_return)

        print("weight_A: ", weight_A, "weight_B: ", weight_B, "expected_return: ", expected_return, "standard_deviation: ", standard_deviation)
        print("risky_portfolio_return: ", risky_portfolio_return, "risky_portfolio_std_dev: ", risky_portfolio_std_dev, "risk_free_rate: ", risk_free_rate, "client_weight_in_risky: ", client_weight_in_risky)
        print("proportion_stock_A: ", proportion_stock_A, "proportion_stock_B: ", proportion_stock_B, "proportion_stock_C: ", proportion_stock_C, "total_proportion: ", total_proportion)
        print("expected_return: ", expected_return, "std_deviation: ", std_deviation)
        print("correlation: ", correlation)


        # Ensure the proportions are valid
        total_proportion = proportion_stock_A + proportion_stock_B + proportion_stock_C

        if not (0 <= proportion_stock_A <= 1 and 0 <= proportion_stock_B <= 1 and 0 <= proportion_stock_C <= 1):
            raise ValueError("Proportions must be between 0 and 1.")
        if total_proportion != 1:
            raise ValueError("The total proportion of stocks A, B, and C must sum to 1.")

        # Ensure the weight is valid
        if client_weight_in_risky < 0 or client_weight_in_risky > 1:
            raise ValueError("Client's weight in the risky portfolio must be between 0 and 1.")

        return Portfolio(risky_portfolio_return=risky_portfolio_return, risky_portfolio_std_dev=risky_portfolio_std_dev, risk_free_rate=risk_free_rate,
                         client_weight_in_risky=client_weight_in_risky, proportion_stock_A=proportion_stock_A, proportion_stock_B=proportion_stock_B,
                         proportion_stock_C=proportion_stock_C, total_proportion=total_proportion, expected_return=expected_return, std_deviation=std_deviation)


    except ValueError as e:
        print(f"Invalid input: {e}\n", traceback.print_exc())


@dataclass
class Asset:
    expected_return: float
    std_deviation: float

def get_asset_input(asset_name):
    print("Enter the details for Expectd Return and Standard devation for ", asset_name)
    expected_return = float(input(f"Enter the expected return for {asset_name}: "))
    std_deviation = float(input(f"Enter the standard deviation for {asset_name}: "))
    print("\nexpected_return: ", expected_return, "std_deviation: ", std_deviation)
    print("Asset input: ", Asset(expected_return=expected_return, std_deviation=std_deviation))
    return Asset(expected_return=expected_return, std_deviation=std_deviation)


# Calculate correlation coefficient

def correlation_coeficant_calculation(asset_A, asset_B):
    # Calculate the ›ariance of A and B
    global correlation2, covariance_AB, correlation
    #correlation2 = (asset_A.std_deviation * asset_B.std_deviation)
#    covariance_AB = np.cov(asset_A.expected_return, asset_B.expected_return)[0][1]
#  correlation = covariance_AB / (asset_A.std_deviation * asset_B.std_deviation)
    #correlation3 = np.corrcoef(asset_A.expected_return, asset_B.expected_return)[0][1]

    mean_A = np.mean(asset_A.expected_return)
    mean_B = np.mean(asset_B.expected_return)
    mean_covariance = np.mean((asset_A.expected_return - mean_A) * (asset_B.expected_return - mean_B))
  #  correlation2 = mean_covariance / (asset_A.std_deviation * asset_B.std_deviation)
   # correlation_AB = mean_covariance / (asset_A.std_deviation * asset_B.std_deviation)
    correlation = mean_A - mean_B
   # correlation = np.corrcoef(asset_A.expected_return, asset_B.expected_return)[0][1]
   # covariance_AB = np.mean((asset_A.expected_return - mean_A) * (asset_B.expected_return - mean_B))
    #correlation = covariance_AB / (asset_A.std_deviation * asset_B.std_deviation)
    print("Mean of A: ", mean_A, "Mean of B: ", mean_B)
    print("correlation: ", correlation)
    print("Mean Covariance: ", mean_covariance)
#    print("Covariance of A and B: ", correlation2)
   # print("Correlation of A and B: ", correlation_AB)


    return correlation


def plot_oppurtunity_set(asset_A, asset_B, t_bills):
    #
    # Plotting the opportunity set
    print("************************************\nPlotting the opportunity set for Assets A, B, and T-bills")
    time.sleep(.5)
    fig, ax = plt.subplots()

    # Plotting T-bills as a point
    ax.plot(t_bills.std_deviation, t_bills.expected_return, 'bo', label="T-bills", markersize=10)

    # Plotting Assets A and B as points
    ax.plot(asset_A.std_deviation, asset_A.expected_return, 'ro', label="Asset A", markersize=10)
    ax.plot(asset_B.std_deviation, asset_B.expected_return, 'go', label="Asset B", markersize=10)

    # Connecting the points (A, B) to simulate a basic opportunity set
    std_devs = np.linspace(asset_A.std_deviation, asset_B.std_deviation, 100)
    expected_returns = np.linspace(asset_A.expected_return, asset_B.expected_return, 100)

    ax.plot(std_devs, expected_returns, 'k--', label="Opportunity Set")

    # Adding labels and legend
    ax.set_xlabel('Standard Deviation (Risk)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Opportunity Set for Assets A, B and T-bills')
    ax.legend()

    plt.grid(True)
    plt.show()


def calculate_sharpe_ratio(expected_return, risk_free_rate, std_dev):

    sharp_ratio = (expected_return - risk_free_rate) / std_dev
    print("Sharpe Ratio: ", sharp_ratio)
    return sharp_ratio


def calculate_cal(expected_return, risk_free_rate, std_dev, risk_levels):
    """
    Calculate the returns along the Capital Allocation Line (CAL).

    Args:
        expected_return (float): Expected return of the risky portfolio.
        risk_free_rate (float): Risk-free rate (T-bill return).
        std_dev (float): Standard deviation (risk) of the risky portfolio.
        risk_levels (numpy array): Array of standard deviations (risk) for combined portfolios.

    Returns:
        numpy array: Array of returns along the CAL.
    """
    sharpe_ratio = calculate_sharpe_ratio(expected_return, risk_free_rate, std_dev)
    print("Sharpe Ratio: ", sharpe_ratio)
    return risk_free_rate + sharpe_ratio * risk_levels


def plot_cal(expected_return, risk_free_rate, std_dev):
    """
    Plot the Capital Allocation Line (CAL) for a given risky portfolio and risk-free asset.

    Args:
        expected_return (float): Expected return of the risky portfolio.
        risk_free_rate (float): Risk-free rate (T-bill return).
        std_dev (float): Standard deviation (risk) of the risky portfolio.
    """
    # Generate risk levels (standard deviations) from 0 (100% in T-bills) to 150% of the risky portfolio's risk
    risk_levels = np.linspace(0, 1.5 * std_dev, 100)

    # Calculate returns along the CAL
    cal_returns = calculate_cal(expected_return, risk_free_rate, std_dev, risk_levels)

    # Plot the CAL
    plt.figure(figsize=(10, 6))
    plt.plot(risk_levels, cal_returns, label="Capital Allocation Line (CAL)", color='b')

    # Plot the risk-free asset
    plt.scatter(0, risk_free_rate, color='r', label="Risk-Free Asset (T-bill)", marker='o', s=100)

    # Plot the risky portfolio
    plt.scatter(std_dev, expected_return, color='g', label="Risky Portfolio", marker='o', s=100)

    # Add labels, title, and legend
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Capital Allocation Line (CAL)')
    plt.legend()
    plt.grid(True)
    plt.show()


def correlation_coeficant(asset_A, asset_B):
    # Calculate the covariance of A and B
    returns_A = np.array(asset_A.expected_return)
    returns_B = np.array(asset_B.expected_return)

    # Calculate mean returns
    mean_A = np.mean(asset_A.expected_return)
    mean_B = np.mean(asset_B.expected_return)

    # Calculate the covariance of A and B
    covariance_AB = np.mean((asset_A.expected_return - mean_A) * (asset_B.expected_return - mean_B))

    # Calculate standard deviations
    std_dev_A = np.std(asset_A.expected_return, ddof=0)  # Population standard deviation
    std_dev_B = np.std(asset_B.expected_return, ddof=0)  # Population standard deviation

    # Calcaulte the correlation coefficient
    correlation_AB = covariance_AB / (std_dev_A * std_dev_B)

    print("Mean of A: ", mean_A, "Mean of B: ", mean_B)
    print("std_dev_A: ", std_dev_A, "std_dev_B: ", std_dev_B)
    print("Covariance of A and B: ", covariance_AB)
    print("Correlation Coefficient: ", correlation_AB)

    return correlation_AB


''' takes the expected returns and standard deviations of two assets, calculates the optimal portfolio weights for assets A and B, and provides the expected return and standard deviation of the optimal portfolio.'''
def calculate_optimal_portfolio(asset_A, asset_B, t_bills):
    # Calculating the covariance of A and B

    correlation_coef = correlation_coeficant(asset_A, asset_B)
    covariance_AB = correlation_coef * asset_A.std_deviation * asset_B.std_deviation
    print("Covariance of A and B: ", covariance_AB)
    print("Correlation Coefficient: ", correlation_coef)


    # Calculating the weights of A and B in the optimal portfolio
    weight_A = (asset_B.std_deviation ** 2 - covariance_AB) / ((asset_A.std_deviation ** 2) + (asset_B.std_deviation ** 2) - 2 * covariance_AB)
    weight_B = 1 - weight_A

    # Calculating the expected return and standard deviation of the optimal portfolio
    expected_return_optimal = weight_A * asset_A.expected_return + weight_B * asset_B.expected_return
    std_deviation_optimal = ((weight_A ** 2) * (asset_A.std_deviation ** 2) + (weight_B ** 2) * (asset_B.std_deviation ** 2) + 2 * weight_A * weight_B * covariance_AB) ** 0.5
    print("*****************************************\n"
          "weight_A: ", weight_A, "weight_B: ", weight_B, "expected_return_optimal: ", expected_return_optimal, "std_deviation_optimal: ", std_deviation_optimal)
    return weight_A, weight_B, expected_return_optimal, std_deviation_optimal

'''
#The slope of the Capital Allocation Line (CAL) represents the Sharpe ratio of the optimal risky portfolio  P . The Sharpe ratio is a measure of the risk-adjusted return and is calculated using the following formula:
# Sharpe Ratio = (Expected Return of the Optimal Portfolio - Risk-Free Rate) / Standard Deviation of the Optimal Portfolio
# The Sharpe ratio is maximized when the optimal portfolio is chosen, as it represents the highest risk-adjusted return.
'''

def calculate_cal_slope(optimal_return, optimal_std_dev, t_bill_return):
    cal_slope = (optimal_return - t_bill_return) / optimal_std_dev
    print("optimal_return: ", optimal_return, "optimal_std_dev: ", optimal_std_dev, "t_bill_return: ", t_bill_return, "\ncal_slope: ", cal_slope)
    return (optimal_return - t_bill_return) / optimal_std_dev

def calculate_cal_slope2(sharpe_ratio, t_bill_return, std_dev):
    cal_slope2 = (sharpe_ratio * std_dev) + t_bill_return
    print("cal_slope2: ", cal_slope2)
    return cal_slope2

#################################
# [USER INPUT]
''' [Function] to take input from the user for probabilities that sum to 1'
    this function takes input from the user to create an array of probabilities that sum to 1.
    [USAGE]: It will determine the calculated expected return of a portfolio with given weights and returns of assets
'''

def input_probabilities():
    """
    Takes input from the user to create an array of probabilities that sum to 1.

    Returns:
        list of float: A list of probabilities entered by the user.
    """
    probabilities = []
    num_outcomes = int(input("Enter the number of possible outcomes: "))

    for i in range(num_outcomes):
        prob = float(input(f"Enter the probability for outcome {i + 1}: "))
        if prob < 0 or prob > 1:
            print("Probability must be between 0 and 1. Please enter again.")
            prob = float(input(f"Enter the probability for outcome {i + 1}: "))
        probabilities.append(prob)

    # Check if the sum of probabilities is approximately 1
    if not (0.99 <= sum(probabilities) <= 1.01):
        raise ValueError("The probabilities do not sum to 1. Please re-enter them.")

    return probabilities


''' [Function] to take input from the user for expected returns'
    this function takes input from the user to create an array of expected returns.
    [USAGE]: It will determine the calculated expected return of a portfolio with given weights and returns of assets
'''


def input_expected_returns(): # gets input expected returns fron uyser to calculate the expected return
    """
    Takes input from the user to create an array of expected returns.

    Returns:
        list of float: A list of expected returns entered by the user.
    """
    expected_returns = []
    num_outcomes = int(input("Enter the number of possible outcomes: "))

    for i in range(num_outcomes):
        ret = float(input(f"Enter the expected return for outcome {i + 1} (as decimal, e.g., 0.12 for 12%): "))
        expected_returns.append(ret)

    return expected_returns

###############################
# [CALCULATIONS / ARITHMETIC LOGIC]
'''
#   Calculate the [expected return ]of a portfolio with given probabilities and returns of assets
#   The functions takes two lists as input: probabilities and returns. The probabilities list contains the probabilities of different outcomes, and the returns list contains the returns corresponding to those outcomes.
#   This data is derived from input_probabilities() and input_expected_returns() functions. [USER INPUT]
'''
def expected_return_finance(probabilities, returns):
    # Check if lengths of inputs match
    if len(probabilities) != len(returns):
        raise ValueError("The lengths of probabilities and returns must match.")

    # Check if probabilities sum to 1
    if not (0.99 <= sum(probabilities) <= 1.01):
        raise ValueError("The sum of the probabilities must be approximately 1.")

    # Calculate the expected return
    expected_return = sum([prob * ret for prob, ret in zip(probabilities, returns)])
    print("probabilities: ", probabilities, "returns: ", returns)
    print("expected return: ", expected_return)
    return sum([prob * ret for prob, ret in zip(probabilities, returns)])

def find_optimal_risky_portfolio(returns, std_devs, correlation, risk_free_rate):
    excess_returns = [r - t_bills.expected_return for r in returns]

    # Calculate the weight of asset A in the optimal risky portfolio
    weight_A = (excess_returns[0] * std_devs[1] ** 2 - excess_returns[1] * correlation * std_devs[0] * std_devs[1]) / \
               (excess_returns[0] * std_devs[1] ** 2 + excess_returns[1] * std_devs[0] ** 2 -
                (excess_returns[0] + excess_returns[1]) * correlation * std_devs[0] * std_devs[1])

    weight_B = 1 - weight_A  # Since weights must sum to 1

    # Calculate the expected return and risk (standard deviation) of the optimal risky portfolio
    portfolio_return = calculate_portfolio_return([weight_A, weight_B], returns)
    portfolio_risk = calculate_portfolio_risk([weight_A, weight_B], std_devs, correlation)
    print("weight_A: ", weight_A, "weight_B: ", weight_B, "portfolio_return: ", portfolio_return, "portfolio_risk: ", portfolio_risk)

    return {
        'weight_A': weight_A,
        'weight_B': weight_B,
        'expected_return': portfolio_return,
        'standard_deviation': portfolio_risk
    }

def calculate_portfolio_return(weights, returns):
    return np.dot(weights, returns)

# Calculate the expected return of a portfolio with given weights and returns of assets
def calculate_portfolio_return(weights, asset_A_return, asset_B_return):
    print("weights: ", weights, "asset_A_return: ", asset_A_return, "asset_B_return: ", asset_B_return)
    portfolio_return = weights * asset_A_return + (1 - weights) * asset_B_return
    print("portfolio return = ", portfolio_return)
    return weights * asset_A_return + (1 - weights) * asset_B_return

# Calculate the standard deviation of a portfolio with given weights and standard deviations of assets
def calculate_portfolio_risk(weights, asset_A_STD, asset_B_STD, correlation):
    #return (weights ** 2 * stocks_std_dev ** 2 + (1 - weights) ** 2 * gold_std_dev ** 2 + 2 * weights * (1 - weights) * stocks_std_dev * gold_std_dev * correlation) ** 0

    variance = (weights[0] ** 2 * std_devs[0] ** 2) + (weights[1] ** 2 * std_devs[1] ** 2) + \
               (2 * weights[0] * weights[1] * correlation * std_devs[0] * std_devs[1])
    portfolio_return = calculate_portfolio_return([weight_A, weight_B], returns)
    portfolio_risk = calculate_portfolio_risk([weight_A, weight_B], std_devs, correlation)

    print("weights: ", weights, "asset_A_STD: ", asset_A_STD, "asset_B_STD: ", asset_B_STD, "correlation: ", correlation)
    print("Variance: ", variance)
    print("portfolio_return: ", portfolio_return, "portfolio_risk: ", portfolio_risk)

    print(np.sqrt(
        (weights ** 2 * asset_A_STD ** 2) +
        ((1 - weights) ** 2 * asset_B_STD ** 2) +
        (2 * weights * (1 - weights) * asset_A_STD * asset_B_STD * correlation) ))

    return np.sqrt(
        (weights ** 2 * asset_A_STD ** 2) +
        ((1 - weights) ** 2 * asset_B_STD ** 2) +
        (2 * weights * (1 - weights) * asset_A_STD * asset_B_STD * correlation)
    )

# Calculate the expected return and standard deviation of a portfolio with given weights and returns of assets
def calculate_standard_deviation(probabilities, returns):
    # Step 1: Calculate the expected return (mean)
    expected_return = sum(p * r for p, r in zip(probabilities, returns))

    # Step 2: Calculate the variance
    variance = sum(p * (r - expected_return) ** 2 for p, r in zip(probabilities, returns))

    # Step 3: Calculate the standard deviation
    standard_deviation = np.sqrt(variance)
    print("expected_return: ", expected_return, "standard_deviation: ", standard_deviation, "variance: ", variance)

    return expected_return, standard_deviation

###########################################
def calculate_expected_value(asset_A, weight):
    asset_A = get_asset_input("Asset A")
    expected_value = asset_A.std_deviation * weight
    print("Calculating the expected value of the rate of return on his portfolio")
    print("\n\nasset_A: ", asset_A, "weight: ", weight)
    print("expected_value: ", expected_value)
    return expected_value

def calculate_standard_deviation(t_bills, asset_A,  weight):

    standard_deviation = t_bills.expected_return + weight * (asset_A.expected_return - t_bills.expected_return)
    print("standard_deviation: ", standard_deviation)
    return standard_deviation
##################################
''' TO CALCULATE EXPECTED RETURN AND STANDARD DEVIATION OF A PORTFOLIO WITH GIVEN WEIGHTS AND RETURNS OF ASSETS'''

'''
    [Question 16] Calculate the investment proportion y to ensure the complete portfolio's standard deviation
    does not exceed a specified maximum.
'''

def calculate_proportion_y(target_return, risky_return, risk_free_rate):
    proportion_y = (target_return - risk_free_rate) / (risky_return - risk_free_rate)
    print("proportion_y: ", proportion_y)
    return proportion_y

def calculate_investment_proportion_y(max_std_dev_given, std_dev_risky):
    proportion_y = (max_std_dev_given - std_dev_risky) / (max_std_dev_given - std_dev_risky)

    proprotion_y2 = max_std_dev_complete / std_dev_risky
    print("max_std_dev_given: ", max_std_dev_given, "std_dev_risky: ", std_dev_risky)
    print("proportion_y: ", proportion_y)
    print("proportion_y2: ", proprotion_y2)
    return proprotion_y2

'''    Calculate the expected return of the complete portfolio.'''
def calculate_expected_return_complete(y, risky_return, risk_free_rate):
    expected_portoflio_return = y * risky_return + (1 - y) * risk_free_rate
    print("expected_portoflio_return: ", expected_portoflio_return)
    return y * risky_return + (1 - y) * risk_free_rate
########################################

''' QUESTION 28B  Calculate the maximum fee that can be charged such that the investor's Sharpe ratio
    is at least equal to that of the passive portfolio.
    '''
def calculate_max_fee(risky_return, risk_free_rate, risky_std_dev, passive_return, passive_std_dev):

    # Calculate the Sharpe ratio of the optimal risky portfolio AND     # Calculate the Sharpe ratio of the passive portfolio
    sharpe_ratio = (risky_return - risk_free_rate) / risky_std_dev
    sharpe_passive = (passive_return - risk_free_rate) / passive_std_dev
    max_fee = (sharpe_ratio - sharpe_passive) / sharpe_ratio
    print("\nmax_fee: ", max_fee, "sharpe_ratio: ", sharpe_ratio, "sharpe_passive: ", sharpe_passive)
    return max_fee
############################
''' QUESTION 4: '''

def calculate_present_value(expected_cash_flow, required_return):

    present_value = expected_cash_flow / (1 + required_return)
    print("present_value: ", present_value)
    return expected_cash_flow / (1 + required_return)

def calculate_expected_rate_of_return(expected_cash_flow, purchase_price):
    expected_rate_of_return = (expected_cash_flow / purchase_price) - 1
    print("expected_rate_of_return: ", expected_rate_of_return)
    return (expected_cash_flow / purchase_price) - 1


######################
''' QUESTION 5 - Calculate the maximum level of risk aversion (A) for which the risky portfolio is still preferred to T-bills. '''
def calculate_max_risk_aversion(risky_return, risk_free_rate, risky_std_dev):
    return 2 * (risky_return - risk_free_rate) / (risky_std_dev ** 2)
################
''' QUESTION 6 - Calculate the expected return and standard deviation of the complete portfolio. 
 calculate expected return for given utility level and "
              "plot the indifference curve by calculating the expected return r_P for different values of σ_P (the standard deviation) and plot r_P against σ_P.")
'''

def expected_return_for_indifference_curve(U, A, std_dev):
    expected_return = U + A * std_dev
    print("expected_return: ", expected_return)
    return U + A * std_dev
##########################

''' [CHAPTER 5- QUESTION 1] Calculate the EAR, Quarterly APR, and monthly APR [ when given a principal, time horizon and interest rate.'''
# Function to calculate future value with Effective Annual Rate (EAR)
def calculate_ear(principal, annual_rate, years):
    A_EAR = principal * (1 + annual_rate) ** years
    return A_EAR

# Function to calculate future value with Quarterly APR
def calculate_quarterly_apr(principal, annual_rate, years):
    compounds_per_year_quarterly = 4
    rate_per_period_quarterly = annual_rate / compounds_per_year_quarterly
    total_periods_quarterly = years * compounds_per_year_quarterly
    A_quarterly = principal * (1 + rate_per_period_quarterly) ** total_periods_quarterly
    return A_quarterly

# Function to calculate future value with Monthly APR
def calculate_monthly_apr(principal, annual_rate, years):
    compounds_per_year_monthly = 12
    rate_per_period_monthly = annual_rate / compounds_per_year_monthly
    total_periods_monthly = years * compounds_per_year_monthly
    A_monthly = principal * (1 + rate_per_period_monthly) ** total_periods_monthly
    return A_monthly



######################
''' [CHAPTER 5- QUESTION 2] Calculate Effective Annual Rate (Annually, Monthly, Weekly, daily and contiously) when given a FIXED APR .'''

# Function to calculate EAR for discrete compounding
def calculate_ear(apr, compounding_periods_per_year):
    ear = (1 + apr / compounding_periods_per_year) ** compounding_periods_per_year - 1
    return ear
# (e) Continuous Compounding
def calculate_ear_continuous(apr):
    ear = math.exp(apr) - 1
    return ear


######################
'''' [CHAPTER 5- QUESTION 3] COMPARE TERMINAL VALUES OF TWO INVESTMENTS when Given:'''

# Function to calculate future value with compound interest
def future_value(principal, annual_rate, years, compounds_per_year):
    total_periods = years * compounds_per_year
    rate_per_period = annual_rate / compounds_per_year
    fv = principal * (1 + rate_per_period) ** total_periods
    return fv

####################

''' [CHAPTER 5- QUESTION 4] Find the Total return and determine the asset  Which is the safer investment?'''
# Function to compare investments based on expected inflation rate

def compare_investments(principal, conventional_rate, inflation_plus_base_rate):
    inflation_rate = float(input("Enter the expected inflation rate (as a percentage, e.g., 3 for 3%): ")) / 100

    # Calculate total rates
    inflation_plus_total_rate = inflation_plus_base_rate + inflation_rate

    # Future Values
    fv_conventional = principal * (1 + conventional_rate)
    fv_inflation_plus = principal * (1 + inflation_plus_total_rate)

    # Real Returns
    real_return_conventional = conventional_rate - inflation_rate
    real_return_inflation_plus = inflation_plus_total_rate - inflation_rate

    # Display the results
    print(f"\nFuture Value of Conventional CD: ${fv_conventional:,.2f}")
    print(f"Future Value of Inflation-Plus CD: ${fv_inflation_plus:,.2f}")

    print(f"Real Return of Conventional CD: {real_return_conventional * 100:.2f}%")
    print(f"Real Return of Inflation-Plus CD: {real_return_inflation_plus * 100:.2f}%")

    if fv_conventional > fv_inflation_plus:
        print("\n[ANSWER] The Conventional CD is the better investment based on expected returns.")
    else:
        print("\n[ANSWER] The Inflation-Plus CD is the better investment based on expected returns.")



#########################


''' [CHAPTER 5- QUESTION 9] Calculate the expected return and standard deviation given a set of probabilities of the complete portfolio.'''
def calculate_mean01(values, probabilities):
    mean = sum(value * prob for value, prob in zip(values, probabilities))
    print("mean: ", mean)
    return mean

# Function to calculate variance
def calculate_variance01(values, probabilities, mean):
    variance = sum(((value - mean) ** 2) * prob for value, prob in zip(values, probabilities))
    print("variance: ", variance)
    return variance


##########################
def asset_input():
    print("Enter the details for Expectd Return and Standard devation for  A:")
    asset_A = get_asset_input("Asset A")
    print("\nEnter the details for Expectd Return and Standard devation for  B:")
    asset_B = get_asset_input("Asset B")

    print("\nEnter the details for Expectd Return and Standard devation for T-bills:")

    t_bills = get_asset_input("T-bills")
    print("asset a input: " , t_bills)

    print("**************************\nasset A input: \n Standard Deviation: ", asset_A.std_deviation, "Expected Return: ", asset_A.expected_return,
          "\nasset B input: \n Standard Deviation: ", asset_B.std_deviation, "Expected Return: ", asset_B.expected_return,
          "\nT-bills input: \n Standard Deviation: ", t_bills.std_deviation, "Expected Return: ", t_bills.expected_return, "\n**************************")

    return asset_A, asset_B, t_bills


def main():
    global correlation_coef, user_expected_returns, user_probabilities, covariance_AB, user_expected_returns, user_expected_returns, asset_A, asset_B, t_bills
    label: choices
    print("\nWelcome to the Portfolio Optimization Tool!\n "
          "\n3enter '1' if you like to calculate the optimal portfolio weights for assets A and B, and the expected return "
          "and standard deviation of the optimal portfolio?"
          "\n enter '2' [Problem 16] Calculate CALs slope, and draw grapsh "
          "\n enter '3' if you like to calculate the expected return of a portfolio with given weights and returns of assets"
          "\n enter '4' Calculate the expected return and standard deviation of a portfolio with given weights and returns of assets"
          "\n enter '5' to calculate e expected value and standard and deviation of the rate of return on his portfolio"
          "\n enter '6' [Problem 13] to calculate the Expected rate of return, given 1 asset and Risk free asset "
          "\n enter '7' [Problem 14] Calculate investment proportions of your client's overall portfolio, including the position in T-bills?"
          "\n enter '8' [Problem 15] What is the reward-to-volatility ratio (S) of the optimal risky portfolio?"
          "\n enter '9' [Problem 17] Find 'Y', the proportion of the risky portfolio given a specefic rate of return to  complete portfolio"
          "\n enter '10' [Problem 18] Calclate the investment proportion, expected return, and standard deviation of the complete portfolio"
            "\n enter '11' [Problem 28A] r[eward-to-volatility ratio ](Sharpe ratio) of the optimal risky portfolio\n"
          "\n enter '12' [Problem 28B] mum fee you could charge (as a percentage of the investment in your fund, deducted at the end of the year\n"
          "\n enter '13' [Problem 4] [Calculate Present Value and Expected rate of return] To determine how much you are willing to pay for the risky portfolio, \n"
          "we can [calculate the present value (fair price)] of the portfolio based on the required risk premium and the risk-free rate.\n"
          "\n enter '14'  [Problem 5- CHAPTER 6] Calculate the: \n [maximum level of risk aversion] (A) for which the risky portfolio is still preferred to T-bills. \n"
          "\n enter '15'  [Problem 6 -- Chapter 6]  plot the [indifference curve] \nby calculating the expected return  r_P  for different values of  \sigma_P  (the standard deviation) and plot  r_P  against  \sigma_P .\n"
          "\n enter '16'  [Problem 1- CHAPTER 5]  Calculate the: \n EAR, Quarterly APR, and monthly APR [ when given a principal, time horizon and interest rate."
          "\n enter '17'  [Problem 2- CHAPTER 5]  Calculate Effective Annual Rate \n(Annually, Monthly, Weekly, daily and contiously) when given a FIXED APR .\n"
          "\n enter '18' (INCORRECT)  [Problem 3- CHAPTER 5]  COMPARE TERMINAL VALUES OF TWO INVESTMENTS when Given: \n # Initial principal amount in dollars original_rate \n  # Original annual interest rate (5%)\
          n #\n Reduced annual interest rate due to early withdrawal compounding_periods_per_year  \n# Monthly compounding"
          "\n enter '19'  [Problem 6- CHAPTER 5]  Find the Total return and determine the asset  Which is the safer investment?"
          "\n enter '20'  [Problem 9- CHAPTER 5]  Calculate the expected return and standard deviation given a set of probabilities of the complete portfolio."
          "\n enter '21'  [Problem 4- CHAPTER 6]   how much will you be willing to pay for the portfolio?, given a risk premium"
          "\n enter '22'  [Problem 5- CHAPTER 6]  Calculate the: UTILITY of a risky asset and risk-free assets (tbills) and Compare the two  \n")




    choice = input("Enter your choice: ")
    if choice not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21','22','23','24','25','26','27','28','29','30']:
        print("\nInvalid choice. Please enter a valid choice.\n")
        main()


    if choice == '1':
        print("[Problem 16] plot and calculate the optimal portfolio weights for assets A and B, and the expected return and standard deviation of the optimal portfolio")
        try:
            # Get input for Asset A, Asset B, and T-bills
            print("Enter the details for Expectd Return and Standard devation for  A:")
            asset_A = get_asset_input("Asset A")


            print("\nEnter the details for Expectd Return and Standard devation for  B:")
            asset_B = get_asset_input("Asset B")

            print("\nEnter the details for Expectd Return and Standard devation for T-bills:")

            t_bills = get_asset_input("T-bills")
            print("asset a input: " , t_bills)

            print("**************************\nasset A input: \n Standard Deviation: ", asset_A.std_deviation, "Expected Return: ", asset_A.expected_return,
                  "\nasset B input: \n Standard Deviation: ", asset_B.std_deviation, "Expected Return: ", asset_B.expected_return,
                  "\nT-bills input: \n Standard Deviation: ", t_bills.std_deviation, "Expected Return: ", t_bills.expected_return, "\n**************************")

            # Calculate the correlation coefficient
            coorelation = correlation_coeficant_calculation(asset_A, asset_B)
            print("Correlation Coefficient: ", coorelation)

        except Exception as e:
            print(e)
            print(traceback.print_exc())
            return
            # Plot the opportunity set
        plot_oppurtunity_set(asset_A, asset_B, t_bills)
        plot_cal(asset_A.expected_return, t_bills.expected_return, asset_A.std_deviation)

    # NOTE: OPTION 1-- Calculate the optimal portfolio weights for assets A and B
    if choice == '2':
        print("[QUESTION 16] You have chosen to Draw the CAL of your portfolio on an expected return–standard deviation diagram. Find CALs slope.")
        try:
            # Calculate the optimal portfolio weights for assets A and B
            print("Calculate the optimal portfolio weights for assets A and B\nEnter the correlation coefeciant for the funds s:")
            global correlation_coef, user_expected_returns, user_probabilities
            global covariance_AB
            # correlation_coef = float(input("Enter the correlation coefeciant for the funds s: "))
            # covariance_AB = correlation_coef * asset_A.std_deviation * asset_B.std_deviation
            #
            # print("Covariance of A and B: ", covariance_AB)


            asset_A = get_asset_input("Asset A")
            asset_B = get_asset_input("Asset B")
            t_bills = get_asset_input("T-bills")

            sharpe_ratio = calculate_sharpe_ratio(asset_A.expected_return, t_bills.expected_return, asset_A.std_deviation)
            print(f"Sharpe Ratio (Reward-to-Volatility Ratio): {sharpe_ratio:.4f}")

            res = calculate_optimal_portfolio(asset_A, asset_B, t_bills)
            print("Weight of A in the optimal portfolio: ", res[0])
            print("Weight of B in the optimal portfolio: ", res[1])
            print("Expected return of the optimal portfolio: ", res[2])
            print("Standard deviation of the optimal portfolio: ", res[3])

            # Calculate the Sharpe ratio of the optimal portfolioc


            cal_slope2 = calculate_cal_slope2(sharpe_ratio, t_bills.expected_return, asset_A.std_deviation)
            plot_cal(asset_A.expected_return, t_bills.expected_return, asset_A.std_deviation)

            print("\n\n\n[ANSWER- QUESTION 16] Given the optimal portfolio \n", "Optimal Expected Return is: ", res[2],
                 "\n  Optimal Standard Deviation is: ", res[3], "\n", "cal_slope2: ", cal_slope2)

        except Exception as e:
            print(e)
            print(traceback.print_exc())


    #  expected_return_finance function
    # NOTE: OPTION 2-- Calculate the expected return of a portfolio with given weights and returns of assets

    if choice == '3':
        print("You have chosen to calculate the expected return of a portfolio with given weights and returns of assets")
        try:
            print("Calculate the expected return of a portfolio with given weights and returns of assets \nEnter the probabilities for the outcomes")
            user_probabilities = input_probabilities()
            print("The entered probabilities are:", user_probabilities)

            user_expected_returns = input_expected_returns()
            print("The entered expected returns are:", user_expected_returns)

            expected_return_calculated = expected_return_finance(user_probabilities, user_expected_returns)
            print("The calculated expected return is:", expected_return_calculated)

            find_optimal_risky_portfolio(user_expected_returns, user_probabilities, correlation_coef, t_bills.expected_return)


        except ValueError as e:
            print(e)
            print(traceback.print_exc())



        # Option 3 Calculate the expected return of a portfolio with given weights and returns of assets
        try:
            print("Calculate the expected return of a portfolio with given weights and returns of assets")
            print("Enter the weights for the portfolio")
            weights = float(input("Enter the weight for the stocks (as decimal, e.g., 0.6 for 60%): "))
            stocks_return = float(input("Enter the expected return for stocks: [Enter 0 to use Asset A's expected return] "))

            if stocks_return == 0:
                stocks_return = asset_A.expected_return
                print("The expected return for stocks is set to the expected return of Asset A.", stocks_return)
            gold_return = float(input("Enter the expected return for gold: [enter 0 to use Asset B's expected return] "))
            if gold_return == 0:
                gold_return = asset_B.expected_return
                print("The expected return for gold is set to the expected return of Asset B.", gold_return)

            print("calculating the portfolio return using the weights and expected returns for 2 asset classes\n", "asset_a return: ", stocks_return, "asset_2 return: ", gold_return, "weights: ", weights)
            portfolio_return = calculate_portfolio_return(weights, stocks_return, gold_return)
            print("The expected return of the portfolio is:", portfolio_return)

            portfolio_risk = calculate_portfolio_risk(weights, asset_A.std_deviation, asset_B.std_deviation, correlation_coef)
            print("The risk of the portfolio is:", portfolio_risk)

        except Exception as e:
            print(e)
            print(traceback.print_exc())


    # Option 4 Calculate the expected return and standard deviation of a portfolio with given weights and returns of assets
    if choice == '4':
        print("You have chosen to calculate the expected return and standard deviation of a portfolio with given weights and returns of assets")
        asset_A, asset_B, t_bills = asset_input()
        try:
            user_probabilities = input_probabilities()
          #  user_expected_returns = input_expected_returns()
            expected_return, standard_deviation = calculate_standard_deviation(user_probabilities, user_expected_returns)


            print("The expected return of the portfolio is:", expected_return)
            print("The standard deviation of the portfolio is:", standard_deviation)
        except ValueError as e:
            print(e)
            print(traceback.print_exc())

    # Option 5 Calculate the correlation coefeciant for the funds

    if choice == '5':
        print("\n\nYou have chosen to calculate the correlation coefeciant for the funds")
        try:

            asset_A = get_asset_input("Asset A")
            asset_B = get_asset_input("Asset B")
            t_bills = get_asset_input("T-bills")
            res = calculate_optimal_portfolio(asset_A, asset_B, t_bills)
            print("\n\nWeight of A in the optimal portfolio: ", res[0])
            print("Weight of B in the optimal portfolio: ", res[1])
            print("Expected return of the optimal portfolio: ", res[2])
            print("Standard deviation of the optimal portfolio: ", res[3])
            print("The expected return of the client's portfolio is:", client_portfolio_return)
            print("The standard deviation of the client's portfolio is:", standard_deviation)
            print("The Sharpe ratio of the client's portfolio is:", sharp_ratio)           # user_probabilities = input_probabilities()

            coorelation = correlation_coeficant_calculation(asset_A, asset_B)

            risky_portfolio_return = get_portfolio_inputs()
            risky_portfolio_std_dev = get_portfolio_inputs()
            risk_free_rate =  get_portfolio_inputs()
            client_weight_in_risky= get_portfolio_inputs()
            proportion_stock_A = get_portfolio_inputs()
            proportion_stock_B = get_portfolio_inputs()
            proportion_stock_C = get_portfolio_inputs()
            total_proportion = get_portfolio_inputs()


            sharp_ratio = calculate_sharpe_ratio(Portfolio.risky_portfolio_return, t_bills.expected_return, Portfolio.risky_portfolio_std_dev)

            print("Correlation Coefficient: ", coorelation)
            print("Sharpe Ratio: ", sharp_ratio)
            print(f"\n\nRisky Portfolio Return: {Portfolio.risky_portfolio_return * 100:.2f}%")
            print(f"Risky Portfolio Standard Deviation: {Portfolio.risky_portfolio_std_dev * 100:.2f}%")
            print(f"Risk-Free Rate: {risk_free_rate * 100:.2f}%")
            print(f"Client's Weight in Risky Portfolio: {client_weight_in_risky * 100:.2f}%")
            print(f"Proportion of Stock A: {proportion_stock_A * 100:.2f}%",
                  "\nProportion of Stock B: {proportion_stock_B * 100:.2f}%",
                  "\nProportion of Stock C: {proportion_stock_C * 100:.2f}%")
            print(f"Total Proportion: {total_proportion * 100:.2f}%")

            if risky_portfolio_return is not None:
                # Calculate the expected return of the client's portfolio
                client_portfolio_return = risky_portfolio_return * client_weight_in_risky[0] + Portfolio.risk_free_rate * (
                            1 - Portfolio.client_weight_in_risky)


                res = calculate_optimal_portfolio(asset_A, asset_B, t_bills)
                print("Weight of A in the optimal portfolio: ", res[0])

                # Calculate the Sharpe ratio of the optimal portfolio
                standard_deviation = calculate_portfolio_risk([proportion_stock_A, proportion_stock_B, proportion_stock_C])
                correlation_coeficant_calculation(Portfolio.proportion_stock_A, Portfolio.proportion_stock_C)

                print("Weight of B in the optimal portfolio: ", res[1])
                print("Expected return of the optimal portfolio: ", res[2])
                print("Standard deviation of the optimal portfolio: ", res[3])

                print("The expected return of the client's portfolio is:", client_portfolio_return)
                print("The standard deviation of the client's portfolio is:", standard_deviation)
                print("The Sharpe ratio of the client's portfolio is:", sharp_ratio)


        except Exception as e:
                print(e)
                print(traceback.print_exc())

    if choice == '6':
        print("[QUESTION 13] You have chosen to calculate the Expected rate of return, given 1 asset and Risk free asset")
        try:
          #  asset_A, asset_B, t_bills = asset_input()
            asset_A = get_asset_input("Asset A")
            t_bills = get_asset_input("T-bills")
            weight = float(input("Enter the weight for the asset (as decimal, e.g., 0.6 for 60%): "))
            print("asset_A: ", asset_A, "weight: ", weight)

            expected_value = calculate_expected_value(t_bills, weight)
            std_deviation = calculate_standard_deviation(t_bills, asset_A, weight)

            print("\nThe expected value of the rate of return on his portfolio is:", expected_value)
            print("The standard deviation of the rate of return on his portfolio is:", std_deviation)

        except ValueError as e:
            print(e)
            print(traceback.print_exc())

    if choice == '7':
        print("\n\n[Question 14]You have chosen to calculate the investment proportions of your client's overall portfolio, including the position in T-bills")
        try:

            # asset_A = get_asset_input("Asset A")
            # asset_B = get_asset_input("Asset B")
            # t_bills = get_asset_input("T-bills")
            client_weight_in_risky = float(input("Enter the client's weight in the risky portfolio (e.g., 0.70 for 70%): "))
            proportion_stock_A = float(input("Enter the proportion of Stock A (e.g., 0.40 for 40%): "))
            proportion_stock_B = float(input("Enter the proportion of Stock B (e.g., 0.30 for 30%): "))
            proportion_stock_C = float(input("Enter the proportion of Stock C (e.g., 0.30 for 30%): "))

            client_investment_A = client_weight_in_risky * proportion_stock_A
            client_investment_B = client_weight_in_risky * proportion_stock_B
            client_investment_C = client_weight_in_risky * proportion_stock_C
            client_investment_t_bills = 1 - client_weight_in_risky
            total_proportion = proportion_stock_A + proportion_stock_B + proportion_stock_C

            print("\n\nProportion of Stock A: ", proportion_stock_A, "Proportion of Stock B: ", proportion_stock_B,
                  "Proportion of Stock C: ", proportion_stock_C, "Total Proportion: ", total_proportion)
            print("Proportion of Stock A: ", proportion_stock_A, "Proportion of Stock B: ", proportion_stock_B,
                  "Proportion of Stock C: ", proportion_stock_C, "Total Proportion: ", total_proportion)
            print("\nClient's Weight in Risky Portfolio: ", client_weight_in_risky)


            print("\n\n[ANSWER- Question 14]\nClient's Investment in Stock A: ", client_investment_A, "\nClient's Investment in Stock B: ", client_investment_B, "\nClient's Investment in Stock C: \n", client_investment_C,
                  "\nClient's Investment in T-bills: ", client_investment_t_bills)

            # Calculate the expected return of the client's portfolio

        except Exception as e:
            print(e)
            print(traceback.print_exc())



    if choice == '8':
        print("\n\n[Question 15] [Sharp Ratio]You have chosen to calculate the reward-to-volatility ratio (S) of the optimal risky portfolio")
        try:
            t_bills = get_asset_input("T-bills")
            asset_A = get_asset_input("Asset A")

            sharpe_ratio = calculate_sharpe_ratio(asset_A.expected_return, t_bills.expected_return, asset_A.std_deviation)
            print(f"Sharpe Ratio (Reward-to-Volatility Ratio): {sharpe_ratio:.4f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '9':
        print("\n\n[Question 17] You have chosen to find 'Y', the proportion of the risky portfolio given a specific rate of return to the complete portfolio")
        try:
            target_return = float(input("Enter the target rate of return: [this will be used to find the optimal portolio) "))
            asset_A = get_asset_input("Asset A")
            t_bills = get_asset_input("T-bills")

            proportion_y = calculate_proportion_y(target_return, asset_A.expected_return, t_bills.expected_return)
            print(f"The proportion of the risky portfolio to the complete portfolio is: {proportion_y:.4f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '10':
        print("\n\n[Question 18] You have chosen to calculate the investment proportion, expected return, and standard deviation of the complete portfolio")
        try:
            print("\n")
            max_std_dev_given = float(input("Enter the maximum standard deviation for the complete portfolio: "))
            asset_A = get_asset_input("Asset A")
            y = calculate_investment_proportion_y(max_std_dev_given, std_dev_risky) # Calculate the investment proportion y to ensure the complete portfolio's standard deviation does not exceed a specified maximum.

            print(f"\n\n[ANSWER- PROBLEM 18] \n The investment proportion 'Y' to ensure the complete portfolio's standard deviation does not exceed {max_std_dev_given} is: {y:.4f}")
            results = calculate_expected_return_complete(y, asset_A.expected_return, t_bills.expected_return) #    Calculate the expected return of the complete portfolio.
            print(f"The expected return of the complete portfolio is: {results:.4f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    ''' Question 28A Calculate the reward-to-volatility ratio (Sharpe ratio) of the optimal risky portfolio-- PASSIVE / ACTIVE '''
    if choice == '11':
        print("\n\n[Question 28A] You have chosen to calculate the reward-to-volatility ratio (Sharpe ratio) of the optimal risky portfolio")
        try:
            print("\n")
            asset_A = get_asset_input("Asset A")
            t_bills = get_asset_input("T-bills")

            passive_yield = float(input("Enter the expected return of the passive investment: "))
            passive_std = float(input("Enter the standard deviation of the passive investment: "))

            # Calculate Sharpe ratios [Passive and active
            sharpe_active = calculate_sharpe_ratio(asset_A.expected_return, t_bills.expected_return, asset_A.std_deviation)
            sharpe_passive = calculate_sharpe_ratio(passive_yield, t_bills.expected_return, passive_std)

            print(f"Sharpe Ratio of Active Portfolio: {sharpe_active:.4f}")
            print(f"Sharpe Ratio of Passive Portfolio: {sharpe_passive:.4f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())


    '''[Problem 28B] Calculate the maximum fee that can be charged such that the investor's Sharpe ratio is at least equal to that of the passive portfolio.'''
    if choice == '12':
        print("\n\n[Question 28B] You have chosen to calculate the maximum fee you could charge (as a percentage of the investment in your fund, deducted at the end of the year)")
        print("    Calculate the maximum fee that can be charged such that the investor's Sharpe ratio is at least equal to that of the passive portfolio.")
        try:
            print("\n")
            asset_A = get_asset_input("Asset A")
            t_bills = get_asset_input("T-bills")

            passive_return = float(input("Enter the expected return of the passive investment: "))
            passive_std_dev = float(input("Enter the standard deviation of the passive investment: "))

            print("\n inputs: \nasset_A: ", asset_A, "t_bills: ", t_bills, "passive_return: ", passive_return, "passive_std_dev: ", passive_std_dev)

            max_fee = calculate_max_fee(asset_A.expected_return, t_bills.expected_return, asset_A.std_deviation, passive_return, passive_std_dev)
            print(f"[QUESTION 28B] -- ANSWERThe maximum fee you could charge is: {max_fee:.4f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '13':
        print("\n\n[Question 4] You have chosen to calculate the present value (fair price) of the portfolio based on the required risk premium and the risk-free rate.")
        try:
            min_cash_flow = float(input("Enter the minimum expected cash flow: "))
            max_cash_flow = float(input("Enter the maximum expected cash flow: "))
            probability = float(input("Enter the probability of the maximum cash flow: "))
            risk_premium = float(input("Enter the required risk premium: "))
            t_bills = get_asset_input("T-bills")

            expected_cash_flow = (probability * cash_flow_low) + (probability * cash_flow_high)
            present_value = calculate_present_value(expected_cash_flow, risk_premium)
            expected_rate_of_return = calculate_expected_rate_of_return(expected_cash_flow, present_value)

            print(f"\n\n [QUESTION 4 ANSWER] \n The present value of the portfolio is: {present_value:.4f},\n "
                  f"The expected cash flow is: {expected_cash_flow:.4f}"
                  f"The expected rate of return is: {expected_rate_of_return:.4f}")




        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '14':
        print("\n\n[Question 5] calculate the maximum level of risk aversion (A) for which the risky portfolio is still preferred to T-bills.")
        try:
            asset_A = get_asset_input("Asset A")
            t_bills = get_asset_input("T-bills")

            print(f"\n\nThe maximum level of risk aversion (A) for which the risky portfolio is still preferred to T-bills is: {risk_aversion:.4f}")

            max_risk_aversion = calculate_max_risk_aversion(asset_A.expected_return, t_bills.expected_return, asset_A.std_deviation)
            print(f"Maximum Risk Aversion Coefficient (A): {max_risk_aversion:.2f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())


    if choice == '15':
        print("\n\n[Question 6] calculate expected return for given utility level and "
              "plot the indifference curve by calculating the expected return r_P for different values of σ_P (the standard deviation) and plot r_P against σ_P.")
        try:
            asset_A = get_asset_input("Asset A")
            t_bills = get_asset_input("T-bills")

            utility = float(input("Enter the utility value: "))
            risk_aversion_coefficient = float(input("Enter the risk aversion coefficient: "))
            deviation = float(input("Enter the deviation: "))
            std_devs = np.linspace(0, deviation, 100) # defines values for standard deviation risk

            ## function to calculate the expected return for a given utilyt level
            expected_return_indifference_curve = expected_return_for_indifference_curve(utility, risk_aversion_coefficient, std_devs)
            print("\n\n expected return for the indifference curve: ", expected_return_indifference_curve)

            # Plotting the indifference curve
            plt.figure(figsize=(10, 6))
            plt.plot(std_devs, expected_returns, label=f'Indifference Curve (U = {utility}, A = {risk_aversion_coefficient})', color='b')

            # Adding labels and title
            plt.xlabel('Standard Deviation (Risk)')
            plt.ylabel('Expected Return')
            plt.title(f'Indifference Curve for Utility Level {utility} and Risk Aversion Coefficient {risk_aversion_coefficient}')
            plt.grid(True)
            plt.legend()
            # Show the plot
            plt.show()

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '16':
        print("\n\n[CHAPTER 5- Question 1] Calculate the EAR, Quarterly APR, and monthly APR, when given a principal, time horizon and interest rate.")
        try:

            # Ask the user for input
            principal = float(input("Enter the initial investment amount (principal) in dollars: "))
            years = float(input("Enter the investment period in years: "))
            annual_rate = float(input("Enter the annual interest rate (as a percentage, e.g., 5 for 5%): ")) / 100

            print("\nCalculating future values...\n")

            # Calculate future values using the functions
            A_EAR = calculate_ear(principal, annual_rate, years)
            print(f"\n(a) Future Value with EAR: ${A_EAR:.2f}")

            A_quarterly = calculate_quarterly_apr(principal, annual_rate, years)
            print(f"(b)\n Future Value with Quarterly APR: ${A_quarterly:.2f}")

            A_monthly = calculate_monthly_apr(principal, annual_rate, years)
            print(f"(c)\n Future Value with Monthly APR: ${A_monthly:.2f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '17':

        try:
            print("\n\n[CHAPTER 5- Question 2] Calculate Effective Annual Rate (EAR) when given a FIXED APR.")

            apr_percentage = float(input("\nENTER THE apr: : "))
            apr_decimal = apr_percentage / 100  # Convert APR to decimal form

            '''COMPOUNDING RATESS: '''
            compounding_periods_annual = 1 # Annual compounding
            compounding_periods_monthly = 12 # Monthly compounding
            compounding_periods_weekly = 52 # Weekly compounding
            compounding_periods_daily = 365 # Daily compounding

            ''' Calculate EAR for different compounding frequencies '''
            ear_annual = calculate_ear(apr_decimal, compounding_periods_annual)
            ear_monthly = calculate_ear(apr_decimal, compounding_periods_monthly)
            ear_weekly = calculate_ear(apr_decimal, compounding_periods_weekly)
            ear_daily = calculate_ear(apr_decimal, compounding_periods_daily)

            ear_continuous = calculate_ear_continuous(apr_decimal)

            print(f"\n\n(a) EAR with Annual Compounding: {ear_annual * 100:.4f}")
            print(f"\n\n(b) EAR with Monthly Compounding: {ear_monthly * 100:.4f}")
            print(f"\n\n(c) EAR with Weekly Compounding: {ear_weekly * 100:.4f}")
            print(f"\n\n(d) EAR with Daily Compounding: {ear_daily *100:.4f}")
            print(f"\n\n(e) EAR with Continuous Compounding: {ear_continuous * 100:.4f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '18':
        print("\n\n[CHAPTER 5- Question 3] COMPARE TERMINAL VALUES OF TWO INVESTMENTS when Given: \n # Initial principal amount in dollars original_rate \n  # Original annual interest rate (5%)\
          n #\n Reduced annual interest rate due to early withdrawal compounding_periods_per_year  \n# Monthly compounding")
        try:
            # Get user inputs
            P = float(input("Enter the initial principal amount in dollars: "))
            original_rate = float(
                input("Enter the original annual interest rate (as a percentage, e.g., 5 for 5%): ")) / 100
            penalty_rate = float(input(
                "Enter the penalty annual interest rate due to early withdrawal (as a percentage, e.g., 4 for 4%): ")) / 100
            compounding_periods_per_year = int(
                input("Enter the number of compounding periods per year (e.g., 12 for monthly): "))
            total_years = float(input("Enter the total original investment period in years: "))
            years_elapsed = float(input("Enter the number of years already invested: "))

            # Calculate remaining years
            years_remaining = total_years - years_elapsed

            print("\nCalculating the future value if you withdraw now, and the future value with the penalty incured and the ratio between the two \n ")
            FV_original = future_value(P, original_rate, total_years, compounding_periods_per_year)
            FV_penalty = future_value(P, penalty_rate, years_remaining, compounding_periods_per_year)
            k = FV_original / FV_penalty
            # Number of compounding periods remaining
            n_remaining = years_remaining * compounding_periods_per_year
            # Avoid division by zero if years_remaining is zero
            if n_remaining == 0:
                print("No remaining investment period. The new interest rate cannot be calculated.")
            else:
                rate_per_period_new = k ** (1 / n_remaining) - 1
                R_new = rate_per_period_new * compounding_periods_per_year  # Annual rate
                R_new_percentage = R_new * 100

                print(f"\nFuture Value of the original CD after {total_years} years: ${FV_original:.2f}")
                print(f"Amount after early withdrawal with penalty rate: ${FV_penalty:.2f}")
                print(f"Minimum new annual interest rate needed: {R_new_percentage:.4f}%")

                # Verification: Calculate the future value with the new rate to confirm it matches FV_original
                FV_new = future_value(FV_penalty, R_new / 100, years_remaining, compounding_periods_per_year)
                print(f"Future Value after reinvesting at new rate: ${FV_new:.2f}")
                print(f"Difference between original and new future values: ${FV_new - FV_original:.2f}")


        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '19':
        print("\n\n[CHAPTER 5- Question 6] Find the Total return and determine the asset  Which is the safer investment?")

        try:
            principal = float(input("Enter the initial investment amount in dollars: "))
            conventional_cd_rate = float(input(
                "Enter the annual interest rate for the Conventional CD (as a percentage, e.g., 5 for 5%): ")) / 100
            inflation_plus_cd_rate = float(
                input("Enter the base rate for the Inflation-Plus CD (as a percentage, e.g., 1.5 for 1.5%): ")) / 100
            inflation_rate = float(input("Enter the expected inflation rate (as a percentage, e.g., 3 for 3%): ")) / 100
            compare_investments(principal, conventional_cd_rate, inflation_plus_cd_rate)

        except Exception as e:
            print(e)
            print(traceback.print_exc())
    if choice == '20':
        print("\n\n[CHAPTER 5- Question 9] Calculate the expected return and standard deviation given a set of probabilities of the complete portfolio.")
        try:
            # Get user inputs
            # Get user input for rates of return and probabilities
            values = []
            probabilities = []

            print("Enter the three possible rates of return and their corresponding probabilities.")

            # Input values of q
            for i in range(1, 4):
                try:
                    value = float(input(f"Enter value q{i}: "))
                    values.append(value)
                except ValueError:
                    print("Invalid input. Please enter a numerical value.")
                    exit()

            # Input probabilities
            for i in range(1, 4):
                try:
                    prob = float(input(f"Enter probability for q{i} (as a decimal between 0 and 1): "))
                    if 0 <= prob <= 1:
                        probabilities.append(prob)
                    else:
                        print("Probability must be between 0 and 1.")
                        exit()
                except ValueError:
                    print("Invalid input. Please enter a numerical value.")
                    exit()

            # Check if probabilities sum to 1
            if not math.isclose(sum(probabilities), 1.0):
                print("\nError: The probabilities must sum to 1.")
            else:
                # Calculate mean
                mean_q = calculate_mean01(values, probabilities)

                # Calculate variance
                variance_q = calculate_variance01(values, probabilities, mean_q)

                # Calculate standard deviation
                std_deviation_q = math.sqrt(variance_q)

                # Display results
                print(f"\nValues of q: {values}")
                print(f"\nProbabilities: {probabilities}\n")
                print(f"\nExpected Value (Mean) E(q): {mean_q:.4f}")
                print(f"\nVariance Var(q): {variance_q:.4f}")
                print(f"\nStandard Deviation σ(q): {std_deviation_q:.4f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '21':
        try:
            print("\n\n[CHAPTER 6- Question 4] Calculate the present value of the portfolio, given a risk premium.")
            # Get user inputs
            min_cash_flow = float(input("Enter the minimum expected cash flow: "))
            max_cash_flow = float(input("Enter the maximum expected cash flow: "))
            probability = float(input("Enter the probability of the maximum cash flow: "))
            risk_premium = float(input("Enter the required risk premium: "))
            risk_free_rate = float(input("Enter the Risk Free Rate "))

            #t_bills = get_asset_input("T-bills")

            required_return = (risk_free_rate + risk_premium) * 100
            expected_cash_flow = (probability * min_cash_flow) + (probability * max_cash_flow)
            expected_cash_flow2 = risk_free_rate + risk_premium
            combined_cash_flow= expected_cash_flow / (1 + expected_cash_flow2)

            present_value = expected_cash_flow / (1 + required_return)
            expected_rate_of_return = calculate_expected_rate_of_return(expected_cash_flow, present_value)

            print(f"\n[Answer 1A] The required return is: {required_return:.4f}")
            print(f"\n[Answer 1B] \nThe present value of the portfolio is: {present_value:.4f}")
            print(f"\n[Answer 1C] The expected cash flow is: {expected_cash_flow:.4f}")
            print(f"\n[ANSSWER 1D] The Combined cash flow is: {combined_cash_flow:.4f}")

            print(f"\nThe expected rate of return is: {expected_rate_of_return:.4f}")

        except Exception as e:
            print(e)
            print(traceback.print_exc())

    if choice == '22':
        print("\n\n[CHAPTER 6- Question 5] Calculate the utility of a risky asset and a risk-free asset, and compare the two.\n")

        try:
            expected_return_risky = float(
                input("Enter the expected return of the risky portfolio (as a percentage, e.g., 12 for 12%): ")) / 100
            standard_deviation_risky = float(input(
                "Enter the standard deviation of the risky portfolio (as a percentage, e.g., 18 for 18%): ")) / 100
            expected_return_risk_free = float(
                input("Enter the expected return of the risk-free asset (as a percentage, e.g., 7 for 7%): ")) / 100


            standard_deviation_risky_squard = standard_deviation_risky ** 2
            standard_deviation_risky_half = .05 * standard_deviation_risky_squard
            delta = expected_return_risk_free - expected_return_risky
            A = delta / (-standard_deviation_risky_half)
            print(f"Step 4: A = {delta} / (-{standard_deviation_risky_half}) = {A}")
            print(f"\nThe value of A is approximately: {A:.4f}")

            left_side = expected_return_risky - 0.5 * A * standard_deviation_risky_squard

            print(f"Step 3: delta = {delta}")

            risky_utility = expected_return_risky - 0.5 * standard_deviation_risky ** 2
            risk_free_utility = expected_return_risk_free - 0.5 * 0
            risk_free_utility01 = expected_return_risk_free - 0.5 * A

            print(f"\n[Answer 1A] The utility of the risky asset is: {risky_utility * 100 :.4f}")
            print(f"\n[Answer 1B] The utility of the risk-free asset is: {risk_free_utility * 100 :.4f}")
            print(f"\n[Answer 1C] The utility of the risk-free asset is: {risk_free_utility01 * 100 :.4f}")
            print(f"\n[Answer 1E] Leftside: {left_side:.4f}")
            print(f"\n[Answer 1D] The value of A is approximately:-- [THE MAXIMUM RISK AVERSION FOR RISK PORTFLIO] \n {A / 10:.4f}")



        except Exception as e:
            print(e)
            print(traceback.print_exc())

if __name__ == '__main__':
    main()





