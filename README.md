# Master-s-Thesis-
Economic Analysis of an Innovative Agrivoltaics Concept Regarding Future Electricity Market Conditions

<img width="724" height="378" alt="image" src="https://github.com/user-attachments/assets/e35d39ed-4483-455a-b15f-3a8ce126f36a" /> <img width="360" height="480" alt="image" src="https://github.com/user-attachments/assets/9d231b1b-f2a6-4b2c-a6b6-da33cb54af2b" />



1. Project Overview
This project presents a comprehensive workflow for time series analysis and forecasting of European electricity spot prices, followed by a detailed techno-economic evaluation of Agri-Photovoltaic (Agri-PV) systems. The analysis leverages historical data to build robust SARIMAX models, which are then used to forecast future prices under various economic scenarios. Finally, a Monte Carlo simulation and deterministic sensitivity analysis are performed to assess the financial viability (NPV, IRR, LCOE) of different Agri-PV plant configurations.
The entire process is structured into two main Python scripts: one for auxiliary data exploration and another that combines the main analysis pipeline.

2. Workflow & Methodology
The project follows a sequential, multi-stage methodology executed by two primary scripts.

Stage 1: Exploratory Data Analysis & Stationarity Testing (Auxiliary Code.py)

Data Loading & Cleaning: Initial datasets are loaded, cleaned, and prepared for analysis. Timestamps are standardized to an hourly frequency.
Exploratory Data Analysis (EDA): Correlation analysis (Pearson and Spearman) is performed to understand linear and monotonic relationships between variables.
Decomposition: Time series decomposition is used to identify and visualize trend, seasonal, and residual components, with a focus on the daily (m=24) cycle.
Stationarity Analysis: The Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests are used to check for stationarity. The script interactively applies differencing (seasonal and non-seasonal) and visualizes the results with ACF/PACF plots to guide the identification of appropriate ARIMA model orders (p, d, q) and seasonal orders (P, D, Q).

Stage 2: Main Analysis Pipeline (3 Parts Main Code.py)
This single script consolidates the core modeling, forecasting, and financial analysis workflow.

Part 1: Seasonal Model Identification & Cross-Validation:

Based on insights from the auxiliary script, the user provides a fixed non-seasonal ARIMA order (p, d, q) for each season.
The script performs a grid search over the seasonal parameters (P, D, Q) using TimeSeriesSplit cross-validation to find the order that yields the lowest average RMSE. The daily seasonality is fixed at m=24.
The final trained seasonal SARIMAX models and their corresponding StandardScaler objects are saved to disk (.joblib files).

Part 2: Scenario-Based Price Forecasting:
The script generates future hourly profiles for all exogenous variables from 2025 to 2045 based on "Optimistic", "Base Case", and "Pessimistic" scenarios.
It then loads the saved seasonal models and applies them to the generated data to create a continuous, long-term spot price forecast for each scenario.
Forecasts and generated variables are saved to Excel, and a summary report is plotted.

Part 3: Financial Viability Analysis:
A Monte Carlo simulation (700+ iterations) is run to model the financial performance (NPV, IRR, LCOE, Payback Period) of different Agri-PV plant sizes and operational variants.
A deterministic sensitivity analysis is performed to create "Tornado" and "Spider" plots, assessing the impact of key parameters on project NPV.
A final, comprehensive financial report is generated and saved to Excel.

3. File Structure
<img width="864" height="420" alt="image" src="https://github.com/user-attachments/assets/4bea1e25-6a64-4ebd-bf1d-36edf1e7be64" />



4. How to Run the Analysis
The scripts are designed to be run in a specific sequence. Ensure all required data files are present in the data/ directory.

Run Auxiliary Code.py:
This script is for interactive analysis. Review the generated plots (correlation, decomposition, ACF/PACF) to understand the data and determine appropriate differencing orders (d and D) for stationarity. The daily seasonality (m=24) is the focus.

Run 3 Parts Main Code.py:

Part 1 Section: You will be prompted to enter the fixed non-seasonal ARIMA order (p,d,q) for each of the four seasons. These choices should be informed by the analysis in the auxiliary script. The script will then train the models and save the necessary files.
Part 2 & 3 Sections: These parts will run automatically, loading the models from Part 1, generating forecasts, and performing the final financial analysis. All outputs will be saved to the output/ directory.

5. Required Libraries
Make sure you have the following Python libraries installed. You can install them using pip:

pip install pandas numpy statsmodels pmdarima scikit-learn joblib tqdm seaborn numpy-financial openpyxl
pandas: For data manipulation and analysis.
numpy: For numerical operations.
statsmodels: For time series models (SARIMAX, decomposition, stat tests).
pmdarima: For the auto_arima function (used in the auxiliary script for exploration).
scikit-learn: For StandardScaler and TimeSeriesSplit.
joblib: For saving and loading Python objects (models, scalers).
matplotlib & seaborn: For data visualization.
tqdm: For progress bars during long computations.
numpy-financial: For financial calculations (NPV, IRR).
openpyxl: Required by pandas to write to .xlsx files.

6. Declaration
The code in this repository was adapted and modified from various online resources and examples related to ARIMA, SARIMA, SARIMAX, and Monte Carlo simulations.

7. Author
Md Abu Nayeem/TH koln
