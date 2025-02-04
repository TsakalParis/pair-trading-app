import os
import subprocess

# Ensure dependencies are installed
def install_dependencies():
    try:
        import yfinance
    except ModuleNotFoundError:
        subprocess.run(["pip", "install", "yfinance", "sklearn", "matplotlib", "pykalman", "statsmodels"])

install_dependencies()

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

def get_stock_data(tickers, start_date, end_date):
    """
    Fetch and validate stock data from Yahoo Finance.
    Returns DataFrame and warnings instead of raising errors for date mismatches
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date,
                           interval="1d", group_by='ticker')

        if data.empty:
            raise ValueError(f"No data found for any tickers: {tickers}")

        # Check ticker existence
        missing_tickers = [t for t in tickers if t not in data.columns.levels[0]]
        if missing_tickers:
            raise ValueError(f"Tickers not found: {missing_tickers}")

        # Check Adjusted Close availability
        if 'Adj Close' not in data.columns.levels[1]:
            available_tickers = []
            for t in tickers:
                try:
                    if not data[t]['Close'].dropna().empty:
                        available_tickers.append(t)
                except KeyError:
                    continue
            msg = "Adjusted Close prices not available for requested tickers.\n"
            if available_tickers:
                msg += f"Close prices available for: {', '.join(available_tickers)}"
            raise ValueError(msg)

        # Process Adjusted Close data
        adj_close_data = data.xs('Adj Close', level=1, axis=1)

        # Collect date information and validate
        warnings = []
        actual_start_dates = {}

        for ticker in tickers:
            series = adj_close_data[ticker].dropna()
            if series.empty:
                raise ValueError(f"No Adjusted Close data for {ticker}")

            first_date = series.index.min()
            actual_start_dates[ticker] = first_date.strftime("%Y-%m-%d")

            # Check if actual start is after requested start
            if first_date > pd.to_datetime(start_date):
                warnings.append(
                    f"{ticker}: Available from {first_date.strftime('%Y-%m-%d')} "
                    f"(requested {start_date})"
                )

        return adj_close_data[tickers], warnings

    except Exception as e:
        raise RuntimeError(f"Data retrieval failed: {str(e)}") from e


def process_data(data, tickers, train_size):
    """
    Process data to get aligned log prices and training parameters
    """
    y1 = data[tickers[0]].dropna()
    y2 = data[tickers[1]].dropna()

    # Align on common dates
    if len(y1) != len(y2):
        common_dates = y1.index.intersection(y2.index)
        y1 = y1.loc[common_dates]
        y2 = y2.loc[common_dates]

    # Convert to log prices
    y1_log = np.log(y1)
    y2_log = np.log(y2)

    # Calculate time parameters
    T = len(y1_log)
    T_trn = int(T * train_size)

    return y1_log, y2_log, T, T_trn

# Cointegration test for log prices
def coint_test(y1_log, y2_log):
    score, p_value, _ = coint(y1_log, y2_log)
    return score, p_value

# Robust MLE for initialization
def initialize_robust_MLE(y1, y2, lookback):
    # Use the first `lookback` days to estimate initial mu and gamma
    y1_window = y1[:lookback]
    y2_window = y2[:lookback]

    # Robust regression using RANSAC
    ransac = RANSACRegressor()
    X = y2_window.values.reshape(-1, 1)
    y = y1_window.values
    ransac.fit(X, y)

    # Get robust estimates of gamma and mu
    gamma_hat = ransac.estimator_.coef_[0]
    mu_hat = ransac.estimator_.intercept_

    # Calculate residuals and their variance
    residuals = y1_window - (gamma_hat * y2_window + mu_hat)  # Spread
    var_eps = residuals.var()  # Variance of residuals (observation noise)
    var_y2 = y2_window.var()  # Variance of y2

    # Set initial state mean and covariance
    mu1 = mu_hat
    var_mu1 = (1 / lookback) * var_eps  # Initial variance for mu
    gamma1 = gamma_hat
    var_gamma1 = (1 / lookback) * var_eps / var_y2  # Initial variance for gamma

    return mu1, gamma1, var_mu1, var_gamma1, var_eps, var_y2

# Rolling LS for hedge ratio
def fit_rollingLS(y1, y2, lookback):
    gamma = np.zeros(len(y1))
    mu = np.zeros(len(y1))
    for t in range(lookback, len(y1)):
        y1_window = y1[t - lookback:t]
        y2_window = y2[t - lookback:t]
        # Perform linear regression: y1 = gamma * y2 + mu
        A = np.vstack([y2_window, np.ones(len(y2_window))]).T
        gamma[t], mu[t] = np.linalg.lstsq(A, y1_window, rcond=None)[0]
    return {"gamma": pd.Series(gamma, index=y1.index), "mu": pd.Series(mu, index=y1.index)}


def kalman_basic(y1, y2, alpha=1e-6, var_eps=1.0, var_y2=1.0, mu1=0, gamma1=1):
    """
    Implements a basic Kalman filter for time-varying regression.

    Parameters:
    - y1 : pd.Series Observed dependent variable.
    - y2 : pd.Series Observed independent variable.
    - alpha : float, optional Process noise scaling factor (default: 1e-6).
    - var_eps : float, optional Observation noise variance (default: 1.0).
    - var_y2 : float, optional Variance of y2 (default: 1.0).
    - mu1 : float, optional Initial state mean for the intercept (default: 0).
    - gamma1 : float, optional Initial state mean for the slope (default: 1).

    Returns:
    - Dictionary with keys 'mu' and 'gamma' containing the estimated state variables.
    """
    # Define time-varying observation matrix Zt
    Zt = np.zeros((len(y2), 1, 2))  # Shape: (T, 1, 2)
    Zt[:, 0, 0] = 1  # Constant term
    Zt[:, 0, 1] = y2.values  # y2 values (time-varying)

    # Define Kalman filter with time-varying observation matrix
    kf = KalmanFilter(
        transition_matrices=np.eye(2),  # State transition matrix (identity for constant state)
        observation_matrices=Zt,  # Time-varying observation matrix
        observation_covariance=np.array([[var_eps]]),  # Observation noise
        transition_covariance=alpha * np.diag([var_eps, var_eps / var_y2]),  # Process noise
        initial_state_mean=[mu1, gamma1],  # Initial state mean
        initial_state_covariance=np.diag([var_eps, var_eps / var_y2]),  # Initial state covariance
    )

    # Prepare observations (y1)
    observations = y1.values.reshape(-1, 1)  # Shape: (T, 1)

    # Apply Kalman filter (forward pass only)
    filtered_state_means, _ = kf.filter(observations)

    # Extract results
    return {
        "mu": pd.Series(filtered_state_means[:, 0], index=y1.index),
        "gamma": pd.Series(filtered_state_means[:, 1], index=y1.index),
    }

def momentum_kalman_filter(y1, y2, alpha=1e-6, alpha_speed=1e-6, var_eps=1.0, var_y2=1.0, mu1=0, gamma1=1):
    """
    Momentum Kalman Filter with two alpha parameters for pair trading.

    Parameters:
    - y1: pd.Series, time series of the first asset's prices.
    - y2: pd.Series, time series of the second asset's prices.
    - alpha: float, process noise scaling for the hedge ratio (gamma).
    - alpha_speed: float, process noise scaling for the rate of change of gamma.
    - var_eps: float, observation noise variance.
    - var_y2: float, variance of y2 (used to scale process noise).
    - mu1: float, initial state mean for the intercept (mu).
    - gamma1: float, initial state mean for the hedge ratio (gamma).

    Returns:
    - dict: Contains filtered state estimates for mu, gamma, and momentum.
    """
    # Ensure y1 and y2 have the same index
    if not y1.index.equals(y2.index):
        raise ValueError("Temporal indices of y1 and y2 are not aligned.")

    # Define time-varying observation matrix Zt for momentum model
    Zt_momentum = np.zeros((len(y2), 1, 3))  # Shape: (T, 1, 3)
    Zt_momentum[:, 0, 0] = 1  # Constant term (mu)
    Zt_momentum[:, 0, 1] = y2.values  # y2 values (time-varying, for gamma)
    Zt_momentum[:, 0, 2] = 0  # Momentum term (initialized to 0)

    # Define process noise covariance matrix
    transition_covariance = np.diag([
        alpha * var_eps,  # Process noise for mu
        alpha * var_eps / var_y2,  # Process noise for gamma
        alpha_speed * var_eps / var_y2  # Process noise for momentum (rate of change of gamma)
    ])

    # Define Kalman filter for momentum model
    kf_momentum = KalmanFilter(
        transition_matrices=np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]]),  # State transition matrix
        observation_matrices=Zt_momentum,  # Time-varying observation matrix
        observation_covariance=np.array([[var_eps]]),  # Observation noise
        transition_covariance=transition_covariance,  # Process noise
        initial_state_mean=[mu1, gamma1, 0],  # Initial state mean
        initial_state_covariance=np.diag([0, 0, 0]),  # Initial state covariance
    )

    # Prepare observations (y1) and inputs (y2)
    observations = y1.values.reshape(-1, 1)  # Shape: (T, 1)

    # Apply Kalman filter (forward pass only)
    filtered_state_means, _ = kf_momentum.filter(observations)

    # Extract results
    results = {
        "mu": pd.Series(filtered_state_means[:, 0], index=y1.index),  # Intercept (mu)
        "gamma": pd.Series(filtered_state_means[:, 1], index=y1.index),  # Hedge ratio (gamma)
        "momentum": pd.Series(filtered_state_means[:, 2], index=y1.index),  # Momentum term
    }

    return results


def plot_mu_gamma(rollingLS, kalman_basic, kalman_momentum, title):
    """Modified plotting function for Streamlit"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 9))

    # Common styling
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    def get_smart_ylim(data):
        """Calculate robust y-limits using IQR-based method"""
        # Calculate percentiles
        p5 = np.nanpercentile(data, 5)
        p95 = np.nanpercentile(data, 95)

        # Calculate IQR-based range
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        robust_lower = q1 - 2.5 * iqr
        robust_upper = q3 + 2.5 * iqr

        # Find tightest reasonable bounds
        lower = max(p5, robust_lower, np.nanmin(data))
        upper = min(p95, robust_upper, np.nanmax(data))

        # Add 5% padding
        padding = (upper - lower) * 0.05
        return lower - padding, upper + padding

    # Rolling LS
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(rollingLS[title], label="Rolling LS", color=colors[0])
    ax1.set_title(f"{title.capitalize()} Estimation: Rolling LS", color='white')
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#0E1117')
    ax1.legend()
    if title == "spread":
        ax1.set_ylim(get_smart_ylim(rollingLS[title]))

    # Kalman Basic
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(kalman_basic[title], label="Kalman Basic", color=colors[1])
    ax2.set_title(f"{title.capitalize()} Estimation: Kalman Basic", color='white')
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#0E1117')
    ax2.legend()
    if title == "spread":
        ax1.set_ylim(get_smart_ylim(kalman_basic[title]))

    # Kalman Momentum
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(kalman_momentum[title], label="Kalman Momentum", color=colors[2])
    ax3.set_title(f"{title.capitalize()} Estimation: Kalman Momentum", color='white')
    ax3.tick_params(colors='white')
    ax3.set_facecolor('#0E1117')
    ax3.legend()
    if title == "spread":
        ax1.set_ylim(get_smart_ylim(kalman_momentum[title]))

    plt.tight_layout()
    return fig

# Augmented Dickey-Fuller Test for stationarity
def adf_test(spread, significance_level=0.05):
    """
    Perform the Augmented Dickey-Fuller (ADF) test on a spread series.
    """
    result = adfuller(spread.dropna(), autolag='AIC')
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary (5%)": result[0] < result[4]['5%'],
        "Stationary (1%)": result[0] < result[4]['1%']
    }

# Generate Threshold Signals
def generate_BB_thresholded_signal(spread, entry_zscore=1, exit_zscore=0, lookback=20, start_signal_at=None):
    """
    Generate a thresholded signal based on Bollinger Bands Z-score without using ta-lib.

    Parameters:
    - spread: pd.Series, the spread time series.
    - entry_zscore: float, Z-score threshold for entering a position.
    - exit_zscore: float, Z-score threshold for exiting a position.
    - lookback: int, lookback period for Bollinger Bands.
    - start_signal_at: int, index to start generating signals (defaults to lookback).

    Returns:
    - dict: Contains 'signal' (pd.Series) and 'z_score' (pd.Series).
    """
    if start_signal_at is None:
        start_signal_at = lookback

    # Calculate rolling mean and standard deviation
    rolling_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = spread.rolling(window=lookback, min_periods=lookback).std()

    # Compute Z-score
    z_score = (spread - rolling_mean) / rolling_std

    # Initialize signal
    signal = pd.Series(0, index=spread.index)

    # Generate signal
    for t in range(start_signal_at, len(spread)):
        if z_score[t] < -entry_zscore:  # Buy signal
            signal[t] = 1
        elif z_score[t] > entry_zscore:  # Short-sell signal
            signal[t] = -1
        else:  # Maintain or close position
            if signal[t - 1] == 1 and z_score[t] < -exit_zscore:  # Maintain buy position
                signal[t] = signal[t - 1]
            elif signal[t - 1] == -1 and z_score[t] > exit_zscore:  # Maintain short-sell position
                signal[t] = signal[t - 1]

    return {'signal': signal, 'z_score': z_score}

# Compute Profits and Losses
def compute_cumPnL_spread_trading(spread, signal, T_trn, compounded=False, trading_fee=0.0005):
    """
    Compute cumulative PnL for a spread trading strategy with trading fees,
    and calculate additional metrics like % of profitable trades and max drawdown.

    Parameters:
    - spread: pd.Series, the spread time series.
    - signal: pd.Series, the trading signals (1 for long, -1 for short, 0 for no position).
    - compounded: bool, whether to compute compounded returns (default is False).
    - trading_fee: float, the trading fee as a proportion of the trade value (default is 0.5%).

    Returns:
    - pd.Series: Cumulative PnL of the strategy.
    - float: Percentage of profitable trades.
    - float: Maximum drawdown of the strategy.
    """
    # Ensure indices are aligned
    if not spread.index.equals(signal.index):
        raise ValueError("Temporal indices of spread and signal are not aligned.")

    # Convert to numpy arrays for faster computation
    spread_values = spread.values
    signal_values = signal.values

    # Compute delayed signal (shifted by one period)
    signal_delayed = np.roll(signal_values, 1)
    signal_delayed[0] = 0  # First value has no previous signal

    # Compute spread returns for log prices
    spread_ret = np.diff(spread_values, prepend=spread_values[0])  # prepend ensures same length

    # Compute portfolio returns
    portf_ret = signal_delayed * spread_ret

    # Compute trading fee impact
    signal_changes = np.diff(signal_values, prepend=0) != 0
    trading_fees = np.abs(signal_changes * trading_fee)

    # Adjust portfolio returns by subtracting trading fees
    portf_ret_net = portf_ret - trading_fees

    # Compute cumulative PnL
    if compounded:
        portf_cumret = np.cumprod(1 + portf_ret_net) - 1
    else:
        portf_cumret = np.cumsum(portf_ret_net)

    # Calculate percentage of profitable trades
    profitable_trades = portf_ret_net[T_trn:] > 0
    percent_profitable = np.mean(profitable_trades) * 100

    # Calculate maximum drawdown
    cumret = portf_cumret if compounded else np.cumsum(portf_ret_net)
    max_drawdown = np.min(cumret) if compounded else np.min(cumret)

    # Return cumulative PnL, % profitable trades, and max drawdown
    return pd.Series(portf_cumret, index=spread.index), percent_profitable, max_drawdown

# Streamlit UI Configuration
st.set_page_config(page_title="Pair Trading", layout="wide")
st.title("📈 Get Asset Data")

# App description
st.markdown("""
**A Quantitative Tool for Statistical Arbitrage Strategy Analysis**  
This application enables robust analysis of pair trading strategies between two cointegrated assets. Key features:

   **For a quick app tutorial** visit [this link](https://youtu.be/EZx_e9rwLEc)

🔍 **Core Functionality**
- Compare assets using three adaptive hedge ratio estimation methods:
  1. Rolling Window Linear Regression
  2. Kalman Filter (Static Coefficients)
  3. Kalman Filter with Momentum (Dynamic Coefficients)
- Automatically handles different market calendars (stocks vs crypto)
- Incorporates transaction costs in P&L calculations

📊 **Workflow**
1. Input asset pair and historical range
2. Define training period (25% default)
3. Check for cointegration
4. Analyze hedge ratio dynamics
5. Optimize entry/exit thresholds
6. Evaluate strategy performance metrics

⚙️ **Customization**
- Adjust model parameters via sidebar after initial analysis
- Modify trading strategy rules (Z-score thresholds, lookback periods)
- Tune risk parameters (max drawdown limits, transaction costs)

Built with `yfinance` for market data and `pykalman` for adaptive filtering.  
*Note: All computations update dynamically with parameter changes.*
""")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Input widgets
col1, col2 = st.columns(2)
with col1:
    ticker1 = st.text_input("First Ticker Symbol", value="EWA").strip()
with col2:
    ticker2 = st.text_input("Second Ticker Symbol", value="EWC").strip()

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2013-01-01"))
with col4:
    end_date = st.date_input("End Date", value=pd.to_datetime("2022-12-31"))

# Training size slider in its own centered column
train_col = st.columns([1])[0]  # Creates a centered column
with train_col:
    train_size = st.slider(
        "Training Data Size Ratio",
        min_value=0.25,
        max_value=0.80,
        value=0.25,
        step=0.01,
        format="%.2f",
        help="Proportion of data used for initial model training"
    )

# Data fetching and validation
if st.button("Get Data", type="primary"):
    # Input validation
    tickers = [ticker1, ticker2]

    if not all(tickers):
        st.error("❌ Please provide both ticker symbols")
    elif ticker1.lower() == ticker2.lower():
        st.error("❌ Ticker symbols must be different")
    elif start_date > end_date:
        st.error("❌ End date must be after start date")
    else:
        try:
            # Convert dates to string format
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Fetch data
            with st.spinner("Fetching stock data..."):
                data, warnings = get_stock_data(tickers, start_str, end_str)

            # Display results
            st.success("✅ Data successfully retrieved!")

            # Show date warnings if any
            if warnings:
                st.warning("⚠️ Note about data availability:")
                for warning in warnings:
                    st.write(f"- {warning}")

            # Process data
            with st.spinner("Processing data..."):
                y1, y2, T, T_trn = process_data(data, tickers, train_size)

            # Show log prices preview
            st.subheader("Processed Log Prices")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"{tickers[0]} Log Prices:")
                st.dataframe(y1.head(), use_container_width=True)
            with col2:
                st.write(f"{tickers[1]} Log Prices:")
                st.dataframe(y2.head(), use_container_width=True)

            # Plot log prices
            st.subheader("Log Price Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))

            # Dark theme styling
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')

            # Plot data
            ax.plot(y1.index, y1, label=tickers[0], color='#1f77b4', linewidth=2)
            ax.plot(y1.index, y2, label=tickers[1], color='#ff7f0e', linewidth=2)

            # Style adjustments
            ax.tick_params(colors='white')
            ax.set_xlabel('Date', color='white')
            ax.set_ylabel('Log Price', color='white')
            ax.legend(frameon=True)

            # Legend styling
            legend = ax.legend()
            frame = legend.get_frame()
            frame.set_facecolor('#0E1117')
            frame.set_edgecolor('white')
            plt.setp(legend.get_texts(), color='white')

            # Grid styling
            ax.grid(True, color='gray', linestyle='--', alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            # Show cointegration results
            score, p_value, _ =  coint(y1, y2)
            if p_value < 0.05:
                notice = "The series are cointegrated (reject null hypothesis)."
            else:
                notice = "The series are NOT cointegrated (fail to reject null)."

            st.subheader("Engle-Granger Cointegration Test for 5% significance")
            st.markdown(f"""
                                        - Test Statistic: {score}
                                        - p-value: {p_value}
                                        - {notice} 
                                        """)

            # Show training parameters
            st.subheader("Training Parameters")
            st.markdown(f"""
                            - Total observations: {T}
                            - Training period: {T_trn} observations (25%)
                            - Test period: {T - T_trn} observations
                            """)

            with st.spinner("Running Kalman Filter Initialization..."):
                mu1, gamma1, var_mu1, var_gamma1, var_eps, var_y2 = initialize_robust_MLE(y1, y2, T_trn)

                # Store processed data in session state
                st.session_state.processed_data = {
                    'y1': y1,
                    'y2': y2,
                    'T_trn': T_trn,
                    'train_size': train_size,
                    'mu1': mu1,
                    'gamma1': gamma1,
                    'var_eps': var_eps,
                    'var_y2': var_y2
                }

        except ValueError as ve:
            st.error(f"Validation Error: {str(ve)}")
        except RuntimeError as re:
            st.error(f"Runtime Error: {str(re)}")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")

# Show parameters ONLY after data is processed
if st.session_state.processed_data:
    with st.sidebar:
        st.header("Model Configuration")
        lookback = st.number_input(
            "Rolling Window Size (days)",
            min_value=1,
            value=60,
            step=1,
            format="%d",
            help="How frequently to re-estimate the hedge ratio and the Bollinger Bands"
        )
        alpha = st.number_input(
            "Process Noise (α)",
            min_value=1e-8,
            max_value=1.0,
            value=1e-6,
            format="%e",
            help="Kalman filter responsiveness to changes (1e-6 = 0.000001)"
        )
        alpha_speed = st.number_input(
            "Momentum Noise (α_speed)",
            min_value=1e-8,
            max_value=1.0,
            value=1e-6,
            format="%e",
            help="Responsiveness to momentum changes (1e-6 = 0.000001)"
        )
        trading_fee = st.number_input(
            "Trading Fee",
            min_value=0.0,
            value=0.0005,
            format="%f",
            help="Brokerage Fee per Trade"
        )
        significance_level = st.selectbox(
            "Significance Level",
            options=[0.01, 0.05],
            index=1,
            help="Threshold for stationarity testing"
        )
        st.header("Trading Strategy Parameters")
        entry_zscore = st.number_input(
            "Entry Z-score",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Z-score threshold for entering positions"
        )
        exit_zscore = st.number_input(
            "Exit Z-score",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Z-score threshold for exiting positions"
        )
        # Help text
        st.markdown("""
        **Parameter Guidance:**
        - Rolling Window: Larger values = smoother but slower adaptation (default: 60)
        - Process Noise (α): Higher = faster response to changes, Lower = smoother estimates
        - Momentum Noise: Controls momentum term adaptation speed
        - Trading Fee : Fee(%) you lose each trade, in decimal format (default: 0.5%)
        """)

# Analysis section (only after data AND parameters are available)
if st.session_state.processed_data and 'lookback' in locals():
    # Retrieve data from session state
    data = st.session_state.processed_data
    y1 = data['y1']
    y2 = data['y2']
    T_trn = data['T_trn']
    mu1 = data['mu1']
    gamma1 = data['gamma1']
    var_eps = data['var_eps']
    var_y2 = data['var_y2']

    # Analysis section
    st.header("Hedge Ratio Analysis")

    with st.spinner("Recalculating models..."):
        # Recalculate with current parameters
        rollingLS = fit_rollingLS(y1, y2, lookback)
        kalman_basic = kalman_basic(y1, y2, alpha, var_eps, var_y2, mu1, gamma1)
        kalman_momentum = momentum_kalman_filter(y1, y2, alpha, alpha_speed, var_eps, var_y2, mu1, gamma1)

        # Compute weights and spread for rolling LS
        rollingLS['w'] = pd.DataFrame(
            np.column_stack([np.ones(len(y1)), -rollingLS['gamma']]) / (1 + rollingLS['gamma']).values.reshape(-1, 1),
            index=y1.index
        )
        rollingLS['spread'] = (y1 * rollingLS['w'].iloc[:, 0] + y2 * rollingLS['w'].iloc[:, 1]) - (
                    rollingLS['mu'] / (1 + rollingLS['gamma']))

        # Kalman basic
        kalman_basic['w'] = pd.DataFrame(
            np.column_stack([np.ones(len(y1)), -kalman_basic['gamma']]) / (1 + kalman_basic['gamma']).values.reshape(-1,
                                                                                                                     1),
            index=y1.index
        )
        kalman_basic['spread'] = (y1 * kalman_basic['w'].iloc[:, 0] + y2 * kalman_basic['w'].iloc[:, 1]) - (
                    kalman_basic['mu'] / (1 + kalman_basic['gamma']))

        # Kalman momentum
        kalman_momentum['w'] = pd.DataFrame(
            np.column_stack([np.ones(len(y1)), -kalman_momentum['gamma']]) / (
                        1 + kalman_momentum['gamma']).values.reshape(-1, 1),
            index=y1.index
        )
        kalman_momentum['spread'] = (y1 * kalman_momentum['w'].iloc[:, 0] + y2 * kalman_momentum['w'].iloc[:, 1]) - (
                kalman_momentum['mu'] / (1 + kalman_momentum['gamma']))

        # Perform ADF tests
        spreads = {
            "Rolling LS Spread": rollingLS['spread'],
            "Kalman Basic Spread": kalman_basic['spread'],
            "Kalman Momentum Spread": kalman_momentum['spread']
        }
        adf_results = {}
        for name, spread in spreads.items():
            adf_results[name] = adf_test(spread, significance_level)

            # Generate trading signals and compute PnL
            start_signal_at = T_trn + 1

            # For Rolling LS
            rollingLS.update(generate_BB_thresholded_signal(
                rollingLS['spread'],
                entry_zscore=entry_zscore,
                exit_zscore=exit_zscore,
                lookback=lookback,
                start_signal_at=start_signal_at
            ))
            rollingLS['portf_cumret'], rolling_pct_profit, rolling_max_down = compute_cumPnL_spread_trading(
                rollingLS['spread'],
                rollingLS['signal'],
                T_trn,
                trading_fee=trading_fee
            )

            # For Kalman Basic
            kalman_basic.update(generate_BB_thresholded_signal(
                kalman_basic['spread'],
                entry_zscore=entry_zscore,
                exit_zscore=exit_zscore,
                lookback=lookback,
                start_signal_at=start_signal_at
            ))
            kalman_basic['portf_cumret'], klbasic_pct_profit, klbasic_max_down = compute_cumPnL_spread_trading(
                kalman_basic['spread'],
                kalman_basic['signal'],
                T_trn,
                trading_fee=trading_fee
            )

            # For Kalman Momentum
            kalman_momentum.update(generate_BB_thresholded_signal(
                kalman_momentum['spread'],
                entry_zscore=entry_zscore,
                exit_zscore=exit_zscore,
                lookback=lookback,
                start_signal_at=start_signal_at
            ))
            kalman_momentum['portf_cumret'], klm_pct_profit, klm_max_down = compute_cumPnL_spread_trading(
                kalman_momentum['spread'],
                kalman_momentum['signal'],
                T_trn,
                trading_fee=trading_fee
            )

    # Plot results
    st.subheader("Gamma Estimates Comparison")
    fig_gamma = plot_mu_gamma(rollingLS, kalman_basic, kalman_momentum, "gamma")
    st.pyplot(fig_gamma)
    plt.close(fig_gamma)

    st.subheader("Mu Estimates Comparison")
    fig_mu = plot_mu_gamma(rollingLS, kalman_basic, kalman_momentum, "mu")
    st.pyplot(fig_mu)
    plt.close(fig_mu)

    st.subheader("Spread Comparison")
    fig_spread = plot_mu_gamma(rollingLS, kalman_basic, kalman_momentum, "spread")
    st.pyplot(fig_spread)
    plt.close(fig_spread)

    # Display results table
    st.subheader("Stationarity Analysis")

    # Create DataFrame for display
    results_df = pd.DataFrame.from_dict(adf_results, orient='index')
    results_df["Significance Level"] = significance_level
    results_df["Stationary"] = results_df["p-value"] < results_df["Significance Level"]

    # Format table
    display_df = results_df[['ADF Statistic', 'p-value', 'Significance Level', 'Stationary']]
    st.dataframe(
        display_df.style.format({
            'ADF Statistic': '{:.4f}',
            'p-value': '{:.4f}',
            'Significance Level': '{:.2f}'
        }),
        use_container_width=True
    )

    # Plot cumulative returns
    st.subheader("Cumulative Returns Comparison")
    fig_returns = plot_mu_gamma(rollingLS, kalman_basic, kalman_momentum, 'portf_cumret')
    st.pyplot(fig_returns)
    plt.close(fig_returns)

    # Display performance metrics
    st.subheader("Strategy Performance Metrics")

    # Create metrics dataframe
    metrics_data = {
        'Strategy': ['Rolling LS', 'Kalman Basic', 'Kalman Momentum'],
        '% Profitable Trades': [rolling_pct_profit, klbasic_pct_profit, klm_pct_profit],
        'Max Drawdown': [rolling_max_down, klbasic_max_down, klm_max_down],
        'Final Return': [
            rollingLS['portf_cumret'].iloc[-1],
            kalman_basic['portf_cumret'].iloc[-1],
            kalman_momentum['portf_cumret'].iloc[-1]
        ]
    }
    metrics_df = pd.DataFrame(metrics_data).set_index('Strategy')

    # Format the dataframe display
    styled_df = metrics_df.style.format({
        '% Profitable Trades': '{:.1f}%',
        'Max Drawdown': '{:.2%}',
        'Final Return': '{:.2%}'
    })

    st.dataframe(styled_df, use_container_width=True)

# App description
st.markdown("""
**Attention!!! To Change the Assets or the Training Size you need to Refresh the Page**    
*Note: The program does not save user inputs.*
""")
