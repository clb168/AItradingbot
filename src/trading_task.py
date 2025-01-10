import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from main import run_hedge_fund, AGENT_CONFIG
from datetime import datetime, timedelta

# Alpaca imports
import alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest,LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.models import Quote
from alpaca.data.requests import StockQuotesRequest


import json

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")  # Use "https://paper-api.alpaca.markets" for paper trading

# Alpaca Client
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Clock  # For market status check
from datetime import datetime

class AlpacaClient:
    def __init__(self):
        self.api = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            url_override=BASE_URL
        )
        self.data_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )

    def get_portfolio_value(self):
        """
        Fetches the current portfolio value from Alpaca.
        """
        try:
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            return portfolio_value, cash
        except Exception as e:
            print(f"Error fetching portfolio value: {e}")
            return None, None

    def is_market_open(self):
        """
        Checks if the market is currently open.
        """
        try:
            clock: Clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"Error checking market status: {e}")
            return False

    def submit_market_order(self, symbol, qty, side, time_in_force="day", extended_hours=False, limit_price=None):
        """
        Submits a market order or a limit order using Alpaca's API.
        If extended_hours=True, the order must be a DAY limit order with a limit price.
        """
        try:
            # Convert time_in_force to a valid TimeInForce enum
            time_in_force_enum = TimeInForce(time_in_force.lower())

            if extended_hours:
                # Ensure time_in_force is set to 'day' and limit_price is provided
                if time_in_force_enum != TimeInForce.DAY:
                    raise ValueError("Extended hours orders must have time_in_force='day'.")
                if not limit_price:
                    raise ValueError("Extended hours orders must have a limit price.")

                # Create a limit order for extended hours
                print(f"Placing extended hours DAY limit order for {symbol} at ${limit_price:.2f}.")
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,  # Required for extended hours
                    limit_price=limit_price
                )
            else:
                # Create a standard market order for regular trading hours
                print(f"Placing regular market order for {symbol}.")
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                    time_in_force=time_in_force_enum,
                    extended_hours=False
                )

            # Submit the order using the Alpaca API
            order = self.api.submit_order(order_data=order_data)
            print(f"Order placed: {order}")
            return order

        except ValueError as e:
            print(f"Validation Error: {e}")
            return None

        except Exception as e:
            print(f"Error submitting order: {e}")
            return None


    def place_order(self, symbol, qty, side):
        """
        Places a market order during regular hours or an extended hours limit order.
        """
        market_open = self.is_market_open()
        if market_open:
            print("Market is open. Placing regular market order.")
            return self.submit_market_order(symbol, qty, side, time_in_force="day", extended_hours=False)
        else:
            print("Market is closed. Placing extended hours limit order.")
            # Fetch the ask price to use as the limit price
            limit_price = self.get_current_price(symbol)
            if not limit_price:
                print(f"Could not fetch ask price for {symbol}. Skipping order.")
                return None
            return self.submit_market_order(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force="day",  # Required for extended hours
                extended_hours=True,
                limit_price=limit_price
            )
    def get_current_price(self, symbol):
        """
        Fetches the latest ask price for the given symbol using the Alpaca Data API.
        """
        try:
            # Request the latest quote for the symbol
            quote_request = StockQuotesRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(minutes=1),  # Last minute
                end=datetime.now()
            )
            quotes = self.data_client.get_stock_quotes(quote_request)
            
            # Extract the latest quote
            if len(quotes[symbol]) > 0:
                latest_quote: Quote = quotes[symbol][-1]  # Get the most recent quote
                
                ask_price = latest_quote.ask_price
                print(f"Fetched ask price for {symbol}: ${ask_price}")
                return ask_price
            else:
                print(f"No quotes available for {symbol}.")
                return None
        except Exception as e:
            print(f"Error fetching quote for {symbol}: {e}")
            return None

    # Fetch trading decision and execute trade
def execute_trade(ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False):
    # Run the hedge fund workflow
    decision = run_hedge_fund(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        agent_config=AGENT_CONFIG,
        show_reasoning=show_reasoning,
    )
    print(f"Trading decision: {decision}")

    # Parse JSON decision
    try:
        decision_data = json.loads(decision)
    except json.JSONDecodeError as e:
        print(f"Error parsing decision JSON: {e}")
        return

    # Extract action, quantity, and reasoning
    action = decision_data.get("action", "hold")
    quantity = decision_data.get("quantity", 0)
    reasoning = decision_data.get("reasoning", "")

    print(f"Action: {action}")
    print(f"Quantity: {quantity}")
    print(f"Reasoning: {reasoning}")

    # Only proceed with a trade if action is "buy" or "sell"
    if action in ["buy", "sell"]:
        client = AlpacaClient()

        try:
            market_order = client.place_order(
                symbol=ticker,
                qty=quantity,
                side=action,
            )
            if market_order:
                print(f"Order successfully placed: {market_order}")
            else:
                print("Order submission failed.")
        except Exception as e:
            print(f"Error submitting order: {e}")
    else:
        print("No trade action required. Holding position.")


# Main Function for User Interaction
if __name__ == "__main__":
    print("\nStarting the AI Hedge Fund Trading Task on Alpaca...")

    # Suggest default values for user input
    default_tickers = "AAPL,GOOG"
    default_start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    default_end_date = datetime.now().strftime('%Y-%m-%d')

    # Initialize Alpaca client
    alpaca_client = AlpacaClient()

    # Fetch portfolio value
    portfolio_value, cash = alpaca_client.get_portfolio_value()
    if portfolio_value is None:
        print("Could not fetch portfolio value. Exiting...")
        exit(1)

    # Collect inputs interactively
    print(f"Current Portfolio Value: ${portfolio_value}")
    tickers = input(f"\nEnter stock tickers (comma-separated) [Default: {default_tickers}]: ") or default_tickers
    start_date = input(f"Enter start date (YYYY-MM-DD) [Default: {default_start_date}]: ") or default_start_date
    end_date = input(f"Enter end date (YYYY-MM-DD) [Default: {default_end_date}]: ") or default_end_date
    show_reasoning = input("Show reasoning for each decision? (yes/no) [Default: no]: ").strip().lower() == "yes"

    # Validate dates
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")

    # Parse the tickers
    tickers = tickers.split(",")

    # Initialize the portfolio using Alpaca data
    portfolio = {
        "cash": cash,
        "positions": {ticker: {"shares": 0, "value": 0} for ticker in tickers},
    }

    # Print the initial portfolio
    print("\nInitial Portfolio:")
    print(f"Total Portfolio Value: ${portfolio_value:.2f}")
    print(f"Cash Available: ${cash:.2f}")
    print("Positions:")
    for ticker, data in portfolio["positions"].items():
        print(f"{ticker}: Shares = {data['shares']}, Value = {data['value']:.2f}")

    # Process each ticker
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        # Check if the market is open
        market_open = alpaca_client.is_market_open()
        print(f"Market Open: {market_open}")

        # Fetch the current price of the ticker
        current_price = alpaca_client.get_current_price(ticker)
        if current_price is None:
            print(f"Could not fetch current price for {ticker}. Skipping...")
            continue
        print(f"Current price of {ticker}: ${current_price:.2f}")

        # Execute trade for the ticker
        execute_trade(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=show_reasoning,
        )

    # Print the final portfolio
    print("\nFinal Portfolio:")
    print(f"Total Portfolio Value: ${portfolio_value:.2f}")
    print(f"Remaining Cash: ${portfolio['cash']:.2f}")
    print("Positions:")
    for ticker, data in portfolio["positions"].items():
        print(f"{ticker}: Shares = {data['shares']}, Value = {data['value']:.2f}")
