from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from main import run_hedge_fund
from tools.api import get_price_data
import seaborn as sns  # For better aesthetics

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
            # Expect JSON output from agent
            import json
            decision = json.loads(agent_output)
            return decision["action"], decision["quantity"]
        except:
            print(f"Error parsing action: {agent_output}")
            return "hold", 0

    def execute_trade(self, ticker, action, quantity, current_price):
        """Validate and execute trades for the specific ticker."""
        executed_quantity = 0

        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                executed_quantity = quantity
                self.portfolio["positions"][ticker]["shares"] += executed_quantity
                self.portfolio["cash"] -= cost
                print(f"Buy executed: {executed_quantity} shares of {ticker} at {current_price}")
            else:
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    executed_quantity = max_quantity
                    self.portfolio["positions"][ticker]["shares"] += executed_quantity
                    self.portfolio["cash"] -= executed_quantity * current_price
                    print(f"Buy partial: {executed_quantity} shares of {ticker} at {current_price}")

        elif action == "sell" and quantity > 0:
            available_shares = self.portfolio["positions"][ticker]["shares"]
            executed_quantity = min(quantity, available_shares)
            if executed_quantity > 0:
                self.portfolio["positions"][ticker]["shares"] -= executed_quantity
                self.portfolio["cash"] += executed_quantity * current_price
                print(f"Sell executed: {executed_quantity} shares of {ticker} at {current_price}")

        if executed_quantity > 0:
            self.trades.append({
                "Date": datetime.now(),
                "Ticker": ticker,
                "Action": action,
                "Quantity": executed_quantity,
                "Price": current_price,
            })
            print(f"Trade logged: {self.trades[-1]}")

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

                action, quantity = self.parse_action(agent_output)
                df = get_price_data(ticker, lookback_start, current_date_str)
                current_price = df.iloc[-1]['close']

                # Execute the trade with validation
                executed_quantity = self.execute_trade(ticker, action, quantity, current_price)
                
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

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of stock tickers (e.g., AAPL,TSLA)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date in YYYY-MM-DD format")
    parser.add_argument("--start-date", type=str, default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"), help="Start date in YYYY-MM-DD format")
    parser.add_argument("--initial_capital", type=float, default=100000, help="Initial capital amount (default: 100000)")

    args = parser.parse_args()

    # Parse the tickers
    tickers = args.tickers.split(",")

    # Create an instance of Backtester
    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
    )

    # Run the backtesting process
    backtester.run_backtest()
    performance_df = backtester.analyze_performance()
