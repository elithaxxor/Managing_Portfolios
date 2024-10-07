import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import traceback, time

from fontTools.misc.psLib import endofthingRE

'''
Define the Asset class, access the expected return and standard deviation of the asset
# I made this a class, so that I can easily access the expected return and standard deviation of the asset
'''

@dataclass
class Asset:
    expected_return: float
    std_deviation: float

def get_asset_input(asset_name):
    print("Enter the details for Expectd Return and Standard devation for ", asset_name)
    expected_return = float(input(f"Enter the expected return for {asset_name}: "))
    std_deviation = float(input(f"Enter the standard deviation for {asset_name}: "))
    return Asset(expected_return=expected_return, std_deviation=std_deviation)


# Calculate correlation coefficient

def correlation_coeficant_calculation(asset_A, asset_B):
    # Calculate the covariance of A and B
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
    """
    Calculate the Sharpe ratio of the risky portfolio.

    Args:
        expected_return (float): Expected return of the risky portfolio.
        risk_free_rate (float): Risk-free rate (T-bill return).
        std_dev (float): Standard deviation (risk) of the risky portfolio.

    Returns:
        float: Sharpe ratio.
    """
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

    print("Covariance of A and B: ", covariance_AB)
    print("Correlation Coefficient: ", correlation_AB)

    return correlation_coef


''' takes the expected returns and standard deviations of two assets, calculates the optimal portfolio weights for assets A and B, and provides the expected return and standard deviation of the optimal portfolio.'''
def calculate_optimal_portfolio(asset_A, asset_B, t_bills):
    # Calculating the covariance of A and B

    correlation_coef = correlation_coeficant(asset_A, asset_B)


    covariance_AB = correlation_coef * asset_A.std_deviation * asset_B.std_deviation

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

############################
''' TO CALCULATE EXPECTED RETURN AND STANDARD DEVIATION OF A PORTFOLIO WITH GIVEN WEIGHTS AND RETURNS OF ASSETS'''


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
        # Get user inputs
        proportion_stock_A = float(input("Enter the proportion of Stock A (e.g., 0.40 for 40%): "))
        proportion_stock_B = float(input("Enter the proportion of Stock B (e.g., 0.30 for 30%): "))
        proportion_stock_C = float(input("Enter the proportion of Stock C (e.g., 0.30 for 30%): "))

        # Ensure the proportions are valid
        total_proportion = proportion_stock_A + proportion_stock_B + proportion_stock_C
        if not (0 <= proportion_stock_A <= 1 and 0 <= proportion_stock_B <= 1 and 0 <= proportion_stock_C <= 1):
            raise ValueError("Proportions must be between 0 and 1.")
        if total_proportion != 1:
            raise ValueError("The total proportion of stocks A, B, and C must sum to 1.")

        # Ensure the weight is valid
        if client_weight_in_risky < 0 or client_weight_in_risky > 1:
            raise ValueError("Client's weight in the risky portfolio must be between 0 and 1.")

        return risky_portfolio_return, risky_portfolio_std_dev, risk_free_rate, client_weight_in_risky,  proportion_stock_A, proportion_stock_B, proportion_stock_C
    except ValueError as e:
        print(f"Invalid input: {e}\n", traceback.print_exc())


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
          "\n enter '2' if you like to calculate the Sharpe ratio of the optimal portfolio?"
          "\n enter '3' if you like to calculate the expected return of a portfolio with given weights and returns of assets"
          "\n enter '4' Calculate the expected return and standard deviation of a portfolio with given weights and returns of assets"
          "\n enter '5' to calculate e expected value and standard and deviation of the rate of return on his portfolio")


    choice = input("Enter your choice: ")
    if choice not in ['1', '2', '3', '4', '5']:
        print("\nInvalid choice. Please enter a valid choice.\n")
        main()


    if choice == '1':
        print("You have chosen to calculate the optimal portfolio weights for assets A and B, and the expected return and standard deviation of the optimal portfolio")
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
    elif choice == '2':
        print("You have chosen to calculate the Sharpe ratio of the optimal portfolio")
        try:
            # Calculate the optimal portfolio weights for assets A and B
            print("Calculate the optimal portfolio weights for assets A and B\nEnter the correlation coefeciant for the funds s:")
            global correlation_coef, user_expected_returns, user_probabilities
            global covariance_AB
            correlation_coef = float(input("Enter the correlation coefeciant for the funds s: "))
            covariance_AB = correlation_coef * asset_A.std_deviation * asset_B.std_deviation

            print("Covariance of A and B: ", covariance_AB)
            res = calculate_optimal_portfolio(asset_A, asset_B, t_bills)
            print("Weight of A in the optimal portfolio: ", res[0])
            print("Weight of B in the optimal portfolio: ", res[1])
            print("Expected return of the optimal portfolio: ", res[2])
            print("Standard deviation of the optimal portfolio: ", res[3])

            # Calculate the Sharpe ratio of the optimal portfolioc
            asset_A_optimal_weight = res[0]
            asset_B_optimal_weight = res[1]
            t_bill_return = t_bills.expected_return
            print("cal_slope: ", t_bill_return, "Optimal Expected Return", res[2], "optimal standard devaition", res[3])
            cal_slope = calculate_cal_slope(res[2], res[3], t_bill_return)

            print("Given the optimal portfolio \n", "Optimal Expected Return is: ", res[2],
                 "\n  Optimal Standard Deviation is: ", res[3], "\n", "cal slope: ", cal_slope)

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

            asset_A, asset_B, t_bills = asset_input()
           # user_probabilities = input_probabilities()

            coorelation = correlation_coeficant_calculation(asset_A, asset_B)

            (risky_portfolio_return, risky_portfolio_std_dev, risk_free_rate, client_weight_in_risky,
             proportion_stock_A, proportion_stock_B, proportion_stock_C) = get_portfolio_inputs()

            sharp_ratio = calculate_sharpe_ratio(risky_portfolio_return, t_bills.expected_return, risky_portfolio_std_dev)

            print("Correlation Coefficient: ", coorelation)
            print("Sharpe Ratio: ", sharp_ratio)
            print(f"\n\nRisky Portfolio Return: {risky_portfolio_return * 100:.2f}%")
            print(f"Risky Portfolio Standard Deviation: {risky_portfolio_std_dev * 100:.2f}%")
            print(f"Risk-Free Rate: {risk_free_rate * 100:.2f}%")
            print(f"Client's Weight in Risky Portfolio: {client_weight_in_risky * 100:.2f}%")
            print(f"Proportion of Stock A: {proportion_stock_A * 100:.2f}%",
                  "\nProportion of Stock B: {proportion_stock_B * 100:.2f}%",
                  "\nProportion of Stock C: {proportion_stock_C * 100:.2f}%")

            if risky_portfolio_return is not None:
                # Calculate the expected return of the client's portfolio
                client_portfolio_return = risky_portfolio_return * client_weight_in_risky + risk_free_rate * (
                            1 - client_weight_in_risky)

                res = calculate_optimal_portfolio(asset_A, asset_B, t_bills)
                print("Weight of A in the optimal portfolio: ", res[0])

                # Calculate the Sharpe ratio of the optimal portfolio
                standard_deviation = calculate_portfolio_risk([proportion_stock_A, proportion_stock_B, proportion_stock_C])
                correlation_coeficant_calculation(proportion_stock_A, proportion_stock_C)

                print("Weight of B in the optimal portfolio: ", res[1])
                print("Expected return of the optimal portfolio: ", res[2])
                print("Standard deviation of the optimal portfolio: ", res[3])

                print("The expected return of the client's portfolio is:", client_portfolio_return)
                print("The standard deviation of the client's portfolio is:", standard_deviation)
                print("The Sharpe ratio of the client's portfolio is:", sharp_ratio)


        except Exception as e:
                print(e)
                print(traceback.print_exc())


if __name__ == '__main__':
    main()





