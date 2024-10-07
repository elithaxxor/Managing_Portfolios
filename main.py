import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import traceback

# Define the Asset class, access the expected return and standard deviation of the asset
@dataclass
class Asset:
    expected_return: float
    std_deviation: float

def get_asset_input(asset_name):
    expected_return = float(input(f"Enter the expected return for {asset_name}: "))
    std_deviation = float(input(f"Enter the standard deviation for {asset_name}: "))
    return Asset(expected_return=expected_return, std_deviation=std_deviation)

def plot_oppurtunity_set(asset_A, asset_B, t_bills):
    #
    # Plotting the opportunity set
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

# takes the expected returns and standard deviations of two assets, calculates the optimal portfolio weights for assets A and B, and provides the expected return and standard deviation of the optimal portfolio.
def calculate_optimal_portfolio(asset_A, asset_B, t_bills):
    # Calculating the covariance of A and B
    covariance_AB = correlation_coef * asset_A.std_deviation * asset_B.std_deviation

    # Calculating the weights of A and B in the optimal portfolio
    weight_A = (asset_B.std_deviation ** 2 - covariance_AB) / ((asset_A.std_deviation ** 2) + (asset_B.std_deviation ** 2) - 2 * covariance_AB)
    weight_B = 1 - weight_A

    # Calculating the expected return and standard deviation of the optimal portfolio
    expected_return_optimal = weight_A * asset_A.expected_return + weight_B * asset_B.expected_return
    std_deviation_optimal = ((weight_A ** 2) * (asset_A.std_deviation ** 2) + (weight_B ** 2) * (asset_B.std_deviation ** 2) + 2 * weight_A * weight_B * covariance_AB) ** 0.5

    return weight_A, weight_B, expected_return_optimal, std_deviation_optimal

#The slope of the Capital Allocation Line (CAL) represents the Sharpe ratio of the optimal risky portfolio  P . The Sharpe ratio is a measure of the risk-adjusted return and is calculated using the following formula:
# Sharpe Ratio = (Expected Return of the Optimal Portfolio - Risk-Free Rate) / Standard Deviation of the Optimal Portfolio
# The Sharpe ratio is maximized when the optimal portfolio is chosen, as it represents the highest risk-adjusted return.


def calculate_cal_slope(optimal_return, optimal_std_dev, t_bill_return):
    cal_slope = (optimal_return - t_bill_return) / optimal_std_dev
    print("optimal_return: ", optimal_return, "optimal_std_dev: ", optimal_std_dev, "t_bill_return: ", t_bill_return, "\ncal_slope: ", cal_slope)
    return (optimal_return - t_bill_return) / optimal_std_dev


## gets input probabilites fron uyser to calculate the expected return
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


#The expected return of an asset or portfolio in finance is the weighted average of possible returns, with the weights being the probabilities of each return happening. The formula to calculate the expected return is:
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


# Calculate the expected return of a portfolio with given weights and returns of assets
def calculate_portfolio_return(weights, asset_A_return, asset_B_return):
    print("weights: ", weights, "asset_A_return: ", asset_A_return, "asset_B_return: ", asset_B_return)
    portfolio_return = weights * asset_A_return + (1 - weights) * asset_B_return
    print("portfolio return = ", portfolio_return)
    return weights * asset_A_return + (1 - weights) * asset_B_return

# Calculate the standard deviation of a portfolio with given weights and standard deviations of assets
def calculate_portfolio_risk(weights, asset_A_STD, asset_B_STD, correlation):
    #return (weights ** 2 * stocks_std_dev ** 2 + (1 - weights) ** 2 * gold_std_dev ** 2 + 2 * weights * (1 - weights) * stocks_std_dev * gold_std_dev * correlation) ** 0
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



def main():
    global correlation_coef, user_expected_returns, user_probabilities, covariance_AB,\
        user_probabilities, user_expected_returns, user_probabilities, user_expected_returns, asset_A, asset_B, t_bills
    print("Welcome to the Portfolio Optimization Tool!\n "
          "enter '1' if you like to calculate the optimal portfolio weights for assets A and B, and the expected return "
          "and standard deviation of the optimal portfolio?"
          "\n enter '2' if you like to calculate the Sharpe ratio of the optimal portfolio?"
          "\n enter '3' if you like to calculate the expected return of a portfolio with given weights and returns of assets"
          "\n enter '4' if you like to calculate the expected return of a portfolio with given weights and returns of assets")

    choice = input("Enter your choice: ")

    if choice == '1':
        print("You have chosen to calculate the optimal portfolio weights for assets A and B, and the expected return and standard deviation of the optimal portfolio")
        try:
            # Get input for Asset A, Asset B, and T-bills
            print("Enter the details for Expectd Return and Standard devation for  A:")
            asset_A = get_asset_input("Asset A")
            print("asset A input: " , asset_A)
            print("\nEnter the details for Expectd Return and Standard devation for  B:")
            asset_B = get_asset_input("Asset B")
            print("asset B input: " , asset_B)

            print("\nEnter the details for Expectd Return and Standard devation for T-bills:")

            t_bills = get_asset_input("T-bills")
            print("asset a input: " , t_bills)

        except Exception as e:
            print(e)
            print(traceback.print_exc())
            return
            # Plot the opportunity set
        plot_oppurtunity_set(asset_A, asset_B, t_bills)

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
                 "\n  Optimal Standard Deviation is: ", res[3], "\n", "the Sharpe ratio is: ", cal_slope)

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

    except exception as e:
        print(e)
        print(traceback.print_exc())

    # Option 4 Calculate the expected return and standard deviation of a portfolio with given weights and returns of assets
    if choice == '4':
        try:
            expected_return, standard_deviation = calculate_standard_deviation(user_probabilities, user_expected_returns)
            print("The expected return of the portfolio is:", expected_return)
            print("The standard deviation of the portfolio is:", standard_deviation)
        except ValueError as e:
            print(e)
            print(traceback.print_exc())




if __name__ == '__main__':
    main()





