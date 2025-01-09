from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from main import run_hedge_fund
from tools.api import get_price_data
import seaborn as sns  # For better aesthetics
import json
import re

class Backtester:
    def __init__(self, agent, tickers, start_date, end_date, initial_capital):
        self.agent = agent
        self.tickers = tickers  # List of tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.portfolio = {
                            "cash": initial_capital,
                            "positions": {ticker: {"shares": 0, "value": 0} for ticker in tickers}
                        }
        self.portfolio_values = []
        self.trades = []  # List to store trade history

    def parse_action(self, agent_output):
        try:
            
            # Normalize the raw output by replacing percentages with decimal equivalents
            normalized_output = re.sub(r'(\d+)%', lambda m: str(float(m.group(1)) / 100), agent_output)

            # Parse the normalized JSON
            decision = json.loads(normalized_output)

            # Extract the action and quantity with default fallbacks
            action = decision.get("action", "hold")  # Default to "hold" if "action" is missing
            quantity = decision.get("quantity", 0)  # Default to 0 if "quantity" is missing
            
            # Ensure the quantity is an integer
            if isinstance(quantity, str) and quantity.isdigit():
                quantity = int(quantity)
            elif not isinstance(quantity, int):
                raise ValueError(f"Invalid quantity format: {quantity}")

            return action, quantity

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e.msg}. Agent output: {agent_output}")
        except ValueError as e:
            print(f"ValueError: {str(e)}. Agent output: {agent_output}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}. Agent output: {agent_output}")

        # Default fallback if parsing fails
        return "hold", 0





    def execute_trade(self, ticker, action, quantity, current_price, current_date):
        """Validate and execute trades for the specific ticker."""
        executed_quantity = 0  # Track how many shares were actually traded

        # Handle buy action
        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:  # Check if sufficient cash exists
                executed_quantity = quantity
                self.portfolio["positions"][ticker]["shares"] += executed_quantity
                self.portfolio["cash"] -= cost
            else:
                # If not enough cash, calculate maximum affordable quantity
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    executed_quantity = max_quantity
                    self.portfolio["positions"][ticker]["shares"] += executed_quantity
                    self.portfolio["cash"] -= executed_quantity * current_price

        # Handle sell action
        elif action == "sell" and quantity > 0:
            available_shares = self.portfolio["positions"][ticker]["shares"]
            executed_quantity = min(quantity, available_shares)  # Sell only available shares
            if executed_quantity > 0:
                self.portfolio["positions"][ticker]["shares"] -= executed_quantity
                self.portfolio["cash"] += executed_quantity * current_price

        # Log the trade if any shares were bought/sold
        if executed_quantity > 0:
            self.trades.append({
                "Date": current_date.strftime("%Y-%m-%d"),  # Record the current date
                "Ticker": ticker,
                "Action": action,
                "Quantity": executed_quantity,
                "Price": current_price,
            })

        return executed_quantity


    def run_backtest(self):
        dates = pd.date_range(self.start_date, self.end_date, freq="B").to_pydatetime()

        print("\nStarting backtest...")

        print(f"{'Date':<12} {'Ticker':<6} {'Action':<6} {'Quantity':>8} {'Price':>8} {'Cash':>12} {'Stock':>8} {'Total Value':>12}")
        print("-" * 100)

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            for ticker in self.tickers:
                agent_output = self.agent(
                    ticker=ticker,
                    start_date=lookback_start,
                    end_date=current_date_str,
                    portfolio=self.portfolio
                )

                print(f"Raw agent output: {agent_output}")  # Debugging

                action, quantity = self.parse_action(agent_output)
                # print(f"Parsed action: {action}, quantity: {quantity}")  # Debugging

                df = get_price_data(ticker, lookback_start, current_date_str)
                current_price = df.iloc[-1]['close']

                # Execute the trade with validation, passing the current_date
                executed_quantity = self.execute_trade(ticker, action, quantity, current_price, current_date)

                total_value = self.portfolio["cash"] + sum(
                    self.portfolio["positions"][t]["shares"] * df.iloc[-1]["close"] for t in self.tickers
                )

                # Save the total portfolio value
                self.portfolio["portfolio_value"] = total_value

                # Log the current state with executed quantity
                print(
                    f"{current_date.strftime('%Y-%m-%d'):<12} {ticker:<6} {action:<6} {executed_quantity:>8} {current_price:>8.2f} "
                    f"{self.portfolio['cash']:>12.2f} {self.portfolio['positions'][ticker]['shares']:>8} {total_value:>12.2f}"
                )

            # Record the portfolio value for the current date
            self.portfolio_values.append(
                {"Date": current_date, "Portfolio Value": self.portfolio["portfolio_value"]}
            )


    def analyze_performance(self):
        sns.set_theme(style="whitegrid")

        # Convert portfolio values to DataFrame
        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")

        # Ensure `Date` in `self.trades` is a timestamp for compatibility
        for trade in self.trades:
            trade["Date"] = pd.Timestamp(trade["Date"])

        # Calculate total return
        total_return = (
            self.portfolio["portfolio_value"] - self.initial_capital
        ) / self.initial_capital
        print(f"Total Return: {total_return * 100:.2f}%")

        # Compute daily returns
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change()

        # Calculate cumulative returns
        performance_df["Cumulative Return"] = (1 + performance_df["Daily Return"]).cumprod() - 1

        # Calculate Sharpe Ratio (assuming 252 trading days in a year)
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()

        if std_daily_return == 0 or pd.isna(std_daily_return):
            sharpe_ratio = "N/A"
        else:
            sharpe_ratio = (mean_daily_return / std_daily_return) * (252 ** 0.5)

        # Calculate Maximum Drawdown
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = performance_df["Portfolio Value"] / rolling_max - 1
        performance_df["Drawdown"] = drawdown
        max_drawdown = drawdown.min()
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

        # Visualize Portfolio Value with Buy/Sell Annotations
        plt.figure(figsize=(14, 7))
        plt.plot(performance_df.index, performance_df["Portfolio Value"], label="Portfolio Value", color="blue")
        plt.fill_between(performance_df.index, performance_df["Portfolio Value"], alpha=0.1, color="blue")
        plt.title("Portfolio Value Over Time", fontsize=16)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.grid(True)

        # Add Buy/Sell Annotations
        for trade in self.trades:
            date = trade["Date"]
            print (date)
            action = trade["Action"]
            price = trade["Price"]
            quantity = trade["Quantity"]

            # Find y-coordinate for annotation
            y_coord = performance_df.loc[date, "Portfolio Value"] if date in performance_df.index else None
            if y_coord is not None:
                if action == "buy":
                    plt.annotate(
                        f"Buy {quantity}",
                        (date, y_coord),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color="green")
                    )
                elif action == "sell":
                    plt.annotate(
                        f"Sell {quantity}",
                        (date, y_coord),
                        textcoords="offset points",
                        xytext=(0, -15),
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color="red")
                    )

        plt.legend()
        plt.show()

        # Visualize Daily Returns
        plt.figure(figsize=(14, 7))
        plt.plot(performance_df.index, performance_df["Daily Return"], label="Daily Return", color="orange")
        plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
        plt.title("Daily Returns Over Time", fontsize=16)
        plt.ylabel("Daily Return", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Visualize Cumulative Returns
        plt.figure(figsize=(14, 7))
        plt.plot(performance_df.index, performance_df["Cumulative Return"], label="Cumulative Return", color="purple")
        plt.fill_between(performance_df.index, performance_df["Cumulative Return"], alpha=0.1, color="purple")
        plt.title("Cumulative Returns Over Time", fontsize=16)
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Visualize Drawdown
        plt.figure(figsize=(14, 7))
        plt.plot(performance_df.index, performance_df["Drawdown"], label="Drawdown", color="red")
        plt.fill_between(performance_df.index, performance_df["Drawdown"], alpha=0.1, color="red")
        plt.title("Drawdown Over Time", fontsize=16)
        plt.ylabel("Drawdown", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Generate a Table of Trades
        trades_df = pd.DataFrame(self.trades)
        print("\nTrade History:")
        print(trades_df)

        return performance_df
    
### 4. Run the Backtest #####
if __name__ == "__main__":
    import argparse

    # Default agent configuration
    AGENT_CONFIG = {
        "market_data_agent": True,
        "technical_analyst_agent": True,
        "fundamentals_agent": False,
        "sentiment_agent": False,
        "risk_management_agent": True,
        "portfolio_management_agent": True,
        "valuation_agent": False,
    }

    print("\nWelcome to the Backtesting Simulation!")

    # Show default agent configuration
    print("\nDefault Agent Configuration:")
    for agent, is_enabled in AGENT_CONFIG.items():
        status = "Enabled" if is_enabled else "Disabled"
        print(f"  - {agent}: {status}")

    print("\nYou can enable or disable agents in the next steps.")

    # Suggest default values for user input
    default_tickers = "AAPL,GOOG"
    default_start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    default_initial_capital = 100000.0

    # Collect inputs interactively
    tickers = input(f"\nEnter stock tickers (comma-separated) [Default: {default_tickers}]: ") or default_tickers
    start_date = input(f"Enter start date (YYYY-MM-DD) [Default: {default_start_date}]: ") or default_start_date
    end_date = input(f"Enter end date (YYYY-MM-DD) [Default: {default_end_date}]: ") or default_end_date
    initial_capital = input(f"Enter initial capital amount [Default: {default_initial_capital}]: ") or default_initial_capital

    # Enable/disable agents interactively
    print("\nEnter agents to enable or disable (comma-separated, leave empty to keep defaults):")
    enable_agents = input("Agents to enable: ")
    disable_agents = input("Agents to disable: ")

    # Validate dates
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")

    # Parse the tickers
    tickers = tickers.split(",")

    # Update AGENT_CONFIG based on interactive input
    if enable_agents:
        for agent in enable_agents.split(","):
            if agent in AGENT_CONFIG:
                AGENT_CONFIG[agent] = True
            else:
                print(f"Warning: {agent} is not a recognized agent name.")

    if disable_agents:
        for agent in disable_agents.split(","):
            if agent in AGENT_CONFIG:
                AGENT_CONFIG[agent] = False
            else:
                print(f"Warning: {agent} is not a recognized agent name.")

    # Display the final configuration
    print("\nUpdated Agent Configuration:")
    for agent, is_enabled in AGENT_CONFIG.items():
        status = "Enabled" if is_enabled else "Disabled"
        print(f"  - {agent}: {status}")

    print("\nRunning backtest...\n")

    # Create an instance of Backtester
    backtester = Backtester(
        agent=lambda *args, **kwargs: run_hedge_fund(*args, agent_config=AGENT_CONFIG, **kwargs),
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
    )

    # Run the backtesting process
    backtester.run_backtest()
    performance_df = backtester.analyze_performance()
