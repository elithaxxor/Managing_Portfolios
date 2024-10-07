import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

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
    return (optimal_return - t_bill_return) / optimal_std_dev



def main():
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

    plot_oppurtunity_set(asset_A, asset_B, t_bills)


    print("\nEnter the correlation coefeciant for the funds s:")
    global correlation_coef
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




if __name__ == '__main__':
    main()





