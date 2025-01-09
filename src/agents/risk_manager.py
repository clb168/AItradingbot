import math

from langchain_core.messages import HumanMessage

from agents.state import AgentState, show_agent_reasoning
from tools.api import prices_to_df

import json
import ast

##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Evaluates portfolio risk and sets position limits for a specific ticker."""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    ticker = data["ticker"]  # Current ticker being processed

    prices_df = prices_to_df(data["prices"])
    current_price = prices_df['close'].iloc[-1]  # Get the latest closing price

    # Fetch messages from other agents with a default value if not found
    fundamentals_message = next((msg for msg in state["messages"] if msg.name == "fundamentals_agent"), None)
    technical_message = next((msg for msg in state["messages"] if msg.name == "technical_analyst_agent"), None)
    sentiment_message = next((msg for msg in state["messages"] if msg.name == "sentiment_agent"), None)
    valuation_message = next((msg for msg in state["messages"] if msg.name == "valuation_agent"), None)

    # Handle missing messages by assigning default values
    fundamental_signals = {}
    technical_signals = {}
    sentiment_signals = {}
    valuation_signals = {}

    try:
        if fundamentals_message and fundamentals_message.content.strip():
            fundamental_signals = json.loads(fundamentals_message.content)
        if technical_message and technical_message.content.strip():
            technical_signals = json.loads(technical_message.content)
        if sentiment_message and sentiment_message.content.strip():
            sentiment_signals = json.loads(sentiment_message.content)
        if valuation_message and valuation_message.content.strip():
            valuation_signals = json.loads(valuation_message.content)
    except Exception as e:
        # Fall back to `ast.literal_eval` if JSON parsing fails
        if fundamentals_message and fundamentals_message.content.strip():
            fundamental_signals = ast.literal_eval(fundamentals_message.content)
        if technical_message and technical_message.content.strip():
            technical_signals = ast.literal_eval(technical_message.content)
        if sentiment_message and sentiment_message.content.strip():
            sentiment_signals = ast.literal_eval(sentiment_message.content)
        if valuation_message and valuation_message.content.strip():
            valuation_signals = ast.literal_eval(valuation_message.content)

    # Combine agent signals
    agent_signals = {
        "fundamental": fundamental_signals,
        "technical": technical_signals,
        "sentiment": sentiment_signals,
        "valuation": valuation_signals
    }

    # 1. Calculate Risk Metrics for Current Ticker
    returns = prices_df['close'].pct_change().dropna()
    daily_vol = returns.std()
    volatility = daily_vol * (252 ** 0.5)  # Annualized volatility approximation
    var_95 = returns.quantile(0.05)         # Historical VaR at 95% confidence
    max_drawdown = (prices_df['close'] / prices_df['close'].cummax() - 1).min()

    # 2. Market Risk Assessment
    market_risk_score = 0
    if volatility > 0.30:
        market_risk_score += 2
    elif volatility > 0.20:
        market_risk_score += 1

    # VaR scoring
    # Note: var_95 is typically negative. The more negative, the worse.
    if var_95 < -0.03:
        market_risk_score += 2
    elif var_95 < -0.02:
        market_risk_score += 1

    if max_drawdown < -0.20:
        market_risk_score += 2
    elif max_drawdown < -0.10:
        market_risk_score += 1

    # 3. Position Size Limits for Current Ticker
   # Ensure positions is a dictionary
    if not isinstance(portfolio.get("positions"), dict):
        raise TypeError("Portfolio 'positions' must be a dictionary")

    # Safely get the ticker data
    ticker_data = portfolio["positions"].get(ticker, {})

    if not isinstance(ticker_data, dict):
        ticker_data = {"shares": 0, "value": 0}  # Fallback to default if ticker_data is not a dictionary

    # Get the number of shares, defaulting to 0 if not present
    shares = ticker_data.get("shares", 0)

    current_stock_value = shares * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio["cash"] + sum(
        pos["shares"] * pos["value"] for pos in portfolio["positions"].values()
    )

    base_position_size = total_portfolio_value * 0.25  # 25% max position size
    if market_risk_score >= 4:
        # Reduce position for high risk
        max_position_size = base_position_size * 0.5
    elif market_risk_score >= 2:
        # Slightly reduce for moderate risk
        max_position_size = base_position_size * 0.75
    else:
        # Keep base size for low risk
        max_position_size = base_position_size

    # 4. Stress Testing
    stress_test_scenarios = {
        "market_crash": -0.20,
        "moderate_decline": -0.10,
        "slight_decline": -0.05
    }

    stress_test_results = {}
    for scenario, decline in {"market_crash": -0.20, "moderate_decline": -0.10, "slight_decline": -0.05}.items():
        potential_loss = current_stock_value * decline
        portfolio_impact = potential_loss / total_portfolio_value if total_portfolio_value else 0
        stress_test_results[scenario] = {
            "potential_loss": potential_loss,
            "portfolio_impact": portfolio_impact
        }

    # 5. Risk-Adjusted Signals Analysis
    # Risk-Adjusted Signals Analysis
    low_confidence = any(
        float(signal['confidence'].strip('%')) < 30 
        for signal in agent_signals.values() 
        if isinstance(signal, dict) and 'confidence' in signal
    )

    # Extract unique signals, ensuring only valid dictionaries with a 'signal' key are considered
    unique_signals = set(
        signal['signal']
        for signal in agent_signals.values()
        if isinstance(signal, dict) and 'signal' in signal
    )

    signal_divergence = 2 if len(unique_signals) == 3 else 0

    # Calculate risk score, ensuring low_confidence and signal_divergence are integrated
    risk_score = min(
        round(market_risk_score * 2 + (4 if low_confidence else 0) + signal_divergence),
        10
    )
    # Generate Trading Action
    if risk_score < 6:
        # Low risk encourages buying
        trading_action = "buy"
    elif 6 <= risk_score < 8:
        # Moderate risk encourages reducing exposure
        trading_action = "reduce"
    elif risk_score >= 8:
        # High risk discourages taking action
        trading_action = "hold"
    else:
        # Indecisive outcome
        trading_action = "indecisive"

    current_price = prices_df['close'].iloc[-1]
    suggested_quantity = int(max_position_size / current_price)

    message_content = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "suggested_quantity": suggested_quantity,
        "risk_metrics": {
            "volatility": float(volatility),
            "value_at_risk_95": float(var_95),
            "max_drawdown": float(max_drawdown),
            "market_risk_score": market_risk_score,
            "stress_test_results": stress_test_results
        },
        "reasoning": f"Risk Score {risk_score}/10: Trading action '{trading_action}' selected based on risk level."
    }

    message = HumanMessage(content=json.dumps(message_content), name="risk_management_agent")

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Management Agent")

    return {"messages": state["messages"] + [message]}

