# Pair Trading Analysis

A quantitative analysis tool for statistical arbitrage and pair trading strategies with advanced adaptive estimation techniques.

## 📊 Overview

This application enables robust analysis of pair trading strategies between two cointegrated assets. It implements multiple adaptive hedge ratio estimation methods and provides comprehensive performance metrics to evaluate trading strategy effectiveness.

### 🔍 Core Functionality

- Compare assets using three adaptive hedge ratio estimation methods:
  1. Rolling Window Linear Regression
  2. Kalman Filter (Static Coefficients)
  3. Kalman Filter with Momentum (Dynamic Coefficients)
- Automatically handles different market calendars (stocks vs crypto)
- Incorporates transaction costs in P&L calculations
- Statistically tests for cointegration and spread stationarity
- Generates trading signals based on dynamic z-score thresholds
- Provides visual comparison of different estimation techniques

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/TsakalParis/pair-trading-app.git
cd pair-trading-app

# Install the required dependencies
pip install -r requirements.txt
```

## 📋 Requirements

The application requires the following Python libraries:
```
streamlit
yfinance==0.2.54
pandas
matplotlib
numpy
scikit-learn
pykalman
statsmodels
```

## 🖥️ Usage

To run the application:

```bash
streamlit run app.py
```

### 📊 Workflow

1. Input asset pair and historical date range
2. Define training period (25% default)
3. Check for cointegration
4. Analyze hedge ratio dynamics
5. Optimize entry/exit thresholds
6. Evaluate strategy performance metrics

### ⚙️ Customization

- Adjust model parameters via sidebar after initial analysis:
  - Rolling window size
  - Process noise parameters for Kalman filters
  - Trading fees
  - Significance levels for statistical tests
- Modify trading strategy rules:
  - Z-score thresholds for entry/exit
  - Lookback periods for Bollinger Bands

## 🔬 Technical Details

### Statistical Methods

- **Cointegration Testing**: Engle-Granger methodology to verify pair eligibility
- **Augmented Dickey-Fuller Test**: Assesses stationarity of the spread
- **Kalman Filtering**: State-space model for adaptive parameter estimation
- **Momentum Enhancement**: Second-order dynamics for trend capture

### Trading Signal Generation

Signals are generated using a z-score approach with Bollinger Bands:
- Long position when z-score < -entry_threshold
- Short position when z-score > entry_threshold
- Exit positions when z-score crosses exit_threshold

## 💡 Example Use Cases

- Identifying and exploiting market inefficiencies
- Portfolio hedging through statistical arbitrage
- Developing mean-reversion strategies for related assets
- Market-neutral trading for equity pairs, ETFs, or crypto

## ⚠️ Disclaimer

This application is provided for educational and research purposes only. Trading involves significant risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📬 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

*Note: The program's outputs are for entertainment and educational purposes only and should not be considered financial advice.*
