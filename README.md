
[Install]
2. Clone the repository or copy the provided code into a Python file.
	2.	Run the Python script:
        ```bash
        python portfolio_optimization.py
        ```
   3. install relevant libraries
        ```bash
        pip install numpy
        pip install matplotlib
        ```
      
************************************************************************************************************************************************************************************
[Introduction]
python portfolio_optimization.py
    3.	Follow the on-screen instructions to input the required parameters.
    4.	The script will output the optimal portfolio weights and the expected return and risk of the portfolio.
    5.	You can also visualize the efficient frontier and the optimal portfolio on a plot.
    6.	You can customize the script by changing the input parameters or modifying the code to suit your needs.
    7.	You can also use the script as a module in your own projects by importing the functions and classes from the script.

After running the script, you will be presented with the following menu:
``````
Welcome to the Portfolio Optimization Tool!
Enter '1' to calculate the optimal portfolio weights for assets A and B and the expected return and standard deviation of the optimal portfolio.
Enter '2' to calculate the Sharpe ratio of the optimal portfolio.
Enter '3' to calculate the expected return of a portfolio with given weights and returns of assets.
Enter '4' to calculate the expected return and standard deviation of a portfolio with given weights and returns of assets.
``````

************************************************************************************************************************************************************************************

[Features in Detail]

**1. Calculate Optimal Portfolio Weights (Option 1)**

	•	Input the expected return and standard deviation for Asset A, Asset B, and T-bills.
	•	The tool will calculate and display the optimal portfolio weights, expected return, and standard deviation.
	•	It will also plot the opportunity set showing the risk-return combinations for Asset A, Asset B, and T-bills.

**2. Sharpe Ratio Calculation (Option 2)**

	•	After calculating the optimal portfolio, this option computes the Sharpe ratio, which is a measure of the risk-adjusted return of the portfolio.

**3. Calculate Expected Return (Option 3)**

	•	This option allows the user to input the probabilities and expected returns for different outcomes and calculates the overall expected return of the portfolio.

**4. Expected Return and Standard Deviation (Option 4)**

	•	This option calculates both the expected return and standard deviation of the portfolio, based on the user-provided weights and returns.


************************************************************************************************************************************************************************************
[Functions]

**1. get_asset_input(asset_name)**

Takes user input for the expected return and standard deviation of an asset and returns an Asset object.

**2. plot_opportunity_set(asset_A, asset_B, t_bills)**

Plots the opportunity set for Asset A, Asset B, and T-bills, showing potential risk-return combinations.

**3. calculate_optimal_portfolio(asset_A, asset_B, t_bills)**

Calculates the optimal weights for Asset A and Asset B in the portfolio, as well as the expected return and standard deviation of the optimal portfolio.

**4. calculate_cal_slope(optimal_return, optimal_std_dev, t_bill_return)**

Calculates the slope of the Capital Allocation Line (CAL), which represents the Sharpe ratio of the portfolio.

**5. input_probabilities()**

Prompts the user to input probabilities that sum to 1, which are used for calculating expected returns.

**6. input_expected_returns()**

Prompts the user to input expected returns for each outcome, used in conjunction with probabilities to calculate the overall expected return.

**7. expected_return_finance(probabilities, returns)**

Calculates the expected return based on the input probabilities and returns.

**8. calculate_portfolio_return(weights, asset_A_return, asset_B_return)**

Calculates the expected return of the portfolio based on the weights of Asset A and Asset B.

**9. calculate_portfolio_risk(weights, asset_A_STD, asset_B_STD, correlation)**

Calculates the risk (standard deviation) of the portfolio based on the correlation between Asset A and Asset B.

**10. calculate_standard_deviation(probabilities, returns)**

Calculates both the expected return and the standard deviation based on the input probabilities and returns.

Example Usage: 

Choice(1).	Input Asset A, Asset B, and T-bills data:
```Enter the expected return for Asset A: 0.1
Enter the standard deviation for Asset A: 0.2
Enter the expected return for Asset B: 0.15
Enter the standard deviation for Asset B: 0.25
Enter the expected return for T-bills: 0.05
Enter the standard deviation for T-bills: 0.0
```

2.	Plot the opportunity set and calculate the optimal portfolio:
```Weight of A in the optimal portfolio: 0.6
Weight of B in the optimal portfolio: 0.4
Expected return of the optimal portfolio: 0.125
Standard deviation of the optimal portfolio: 0.22
```

3. Calculate the Sharpe ratio of the optimal portfolio:
```Sharpe ratio of the optimal portfolio: 0.22727272727272727
```

