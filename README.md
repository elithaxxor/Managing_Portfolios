Portfolio Optimization Tool

Welcome to the Portfolio Optimization Tool, a Python program designed to assist with various financial calculations related to portfolio management, asset allocation, and investment analysis. This tool allows users to perform calculations such as expected returns, standard deviations, Sharpe ratios, optimal portfolio weights, and more.

Table of Contents

	•	Overview
	•	Features
	•	Installation
	•	Usage
	•	Running the Program
	•	Menu Options
	•	Dependencies
	•	Examples
	•	Contributing
	•	License

Overview

The Portfolio Optimization Tool is an interactive console application that helps users perform various financial computations related to portfolio optimization. It provides a user-friendly interface to input data for different assets, calculate key financial metrics, and visualize results through plots.

Features

	•	Asset Data Input: Enter expected returns and standard deviations for multiple assets, including stocks and risk-free assets like T-bills.
	•	Correlation Calculation: Compute the correlation coefficient between two assets.
	•	Portfolio Optimization: Determine the optimal portfolio weights for assets to maximize returns or minimize risk.
	•	Expected Return and Risk: Calculate the expected return and standard deviation of a portfolio based on asset weights.
	•	Sharpe Ratio: Compute the reward-to-volatility ratio (Sharpe ratio) for portfolios.
	•	Capital Allocation Line (CAL): Plot the CAL to visualize the risk-return trade-off.
	•	Indifference Curves: Plot indifference curves for different utility levels and risk aversion coefficients.
	•	User Interaction: Interactive menu-driven program allowing users to select different financial calculations.

Installation

	1.	Clone the Repository:
        ``` git clone https://github.com/yourusername/portfolio-optimization-tool.git ```
    2.	Change Directory:
        ``` cd portfolio-optimization-tool ```
    3.	Install Dependencies:
        ``` pip install -r requirements.txt ```
    4.	Run the Program:
        ``` python portfolio_optimization_tool.py ```
    5.	Follow the on-screen instructions to use the tool.

Dependencies
    ```
    1. numpy
    2. matplotlib
    3. dataclasses
    ```

Menu Options
Upon running the program, you will be presented with a menu of options:
Welcome to the Portfolio Optimization Tool!
```
    Enter '1' to calculate the optimal portfolio weights for assets A and B, and the expected return and standard deviation of the optimal portfolio
    Enter '2' [Problem 16] to calculate CAL's slope and draw graphs
    Enter '3' to calculate the expected return of a portfolio with given weights and returns of assets
    Enter '4' to calculate the expected return and standard deviation of a portfolio with given weights and returns of assets
    Enter '5' to calculate the expected value and standard deviation of the rate of return on a portfolio
    Enter '6' [Problem 13] to calculate the expected rate of return, given one asset and a risk-free asset
    Enter '7' [Problem 14] to calculate the investment proportions of your client's overall portfolio, including the position in T-bills
    Enter '8' [Problem 15] to calculate the reward-to-volatility ratio (Sharpe Ratio) of the optimal risky portfolio
    Enter '9' [Problem 17] to find 'Y', the proportion of the risky portfolio given a specific rate of return to complete the portfolio
    Enter '10' [Problem 18] to calculate the investment proportion, expected return, and standard deviation of the complete portfolio
    Enter '11' [Problem 28A] to calculate the reward-to-volatility ratio (Sharpe Ratio) of the optimal risky portfolio
    Enter '12' [Problem 28B] to calculate the maximum fee you could charge that leaves the investor as well off as in a passive portfolio
    Enter '13' [Problem 4] to calculate the present value and expected rate of return for a risky portfolio
    Enter '14' [Problem 5] to calculate the maximum level of risk aversion (A) for which the risky portfolio is still preferred to T-bills
    Enter '15' [Problem 6] to plot the indifference curve for a given utility level and risk aversion coefficient
```


Examples

Example 1: Calculating Optimal Portfolio Weights

Option 1: Calculate the optimal portfolio weights for assets A and B, and the expected return and standard deviation of the optimal portfolio.

Steps:

	1.	Select Option 1 when prompted.
	2.	Enter Asset Details:
	•	Asset A:
	•	Expected Return (e.g., 0.15 for 15%)
	•	Standard Deviation (e.g., 0.2 for 20%)
	•	Asset B:
	•	Expected Return (e.g., 0.1 for 10%)
	•	Standard Deviation (e.g., 0.15 for 15%)
	•	T-bills:
	•	Expected Return (e.g., 0.05 for 5%)
	•	Standard Deviation (0, since it’s risk-free)
	3.	Calculate Correlation Coefficient:
The program will prompt you to enter the correlation coefficient between Asset A and Asset B (e.g., 0.3).
	4.	Results:
	•	Optimal weights for Asset A and B.
	•	Expected return and standard deviation of the optimal portfolio.

Example 2: Plotting the Capital Allocation Line (CAL)

Option 2: Calculate CAL’s slope and draw graphs.

Steps:

	1.	Select Option 2 when prompted.
	2.	Enter Asset Details as in Example 1.
	3.	Results:
	•	The program calculates the Sharpe ratio and the CAL slope.
	•	A plot is displayed showing the CAL and the positions of the assets.

Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

License

This project is licensed under the MIT License.

Note: The code provided in portfolio_optimization.py includes various financial calculations and plotting functions. Please ensure you understand financial concepts such as expected return, standard deviation, Sharpe ratio, and portfolio optimization before using this tool.

Disclaimer: This tool is intended for educational purposes only and should not be used for actual investment decisions without consulting a professional financial advisor.
