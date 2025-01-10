from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from agents.fundamentals import fundamentals_agent
from agents.market_data import market_data_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.state import AgentState
from agents.valuation import valuation_agent

import argparse
from datetime import datetime, timedelta

# Default agent configuration
AGENT_CONFIG = {
    "market_data_agent": True,
    "technical_analyst_agent": True,
    "fundamentals_agent": True,
    "sentiment_agent": False,
    "risk_management_agent": True,
    "portfolio_management_agent": True,
    "valuation_agent": True,
}

##### Run the Hedge Fund #####
def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, agent_config: dict, show_reasoning: bool = False) -> str:
    """
    Runs the hedge fund workflow and returns the final trading decision.

    Parameters
    ----------
    ticker : str
        The ticker symbol for the stock to be analyzed.
    start_date : str
        The start date of the analysis period (YYYY-MM-DD).
    end_date : str
        The end date of the analysis period (YYYY-MM-DD).
    portfolio : dict
        The current state of the portfolio.
    agent_config : dict
        Configuration to toggle agents on or off.
    show_reasoning : bool, optional
        Whether to show the reasoning behind the trading decision (default is False).

    Returns
    -------
    str
        The final trading decision.
    """
    # Define the workflow
    workflow = StateGraph(AgentState)

    # Add nodes dynamically based on the configuration
    for agent, is_enabled in agent_config.items():
        if is_enabled:
            # Add the agent to the workflow
            workflow.add_node(agent, globals()[agent])

    # Set the entry point of the workflow
    if agent_config.get("market_data_agent"):
        workflow.set_entry_point("market_data_agent")

    # Add edges dynamically based on the specified workflow logic
    if agent_config.get("market_data_agent"):
        # Market data agent sends data to all analysis agents
        for analysis_agent in ["technical_analyst_agent", "fundamentals_agent", "sentiment_agent", "valuation_agent"]:
            if agent_config.get(analysis_agent):
                # Add an edge from the market data agent to the analysis agent
                workflow.add_edge("market_data_agent", analysis_agent)

    # All analysis agents send their results to the risk management agent
    if agent_config.get("risk_management_agent"):
        for analysis_agent in ["technical_analyst_agent", "fundamentals_agent", "sentiment_agent", "valuation_agent"]:
            if agent_config.get(analysis_agent):
                # Add an edge from the analysis agent to the risk management agent
                workflow.add_edge(analysis_agent, "risk_management_agent")

    # Risk management agent sends its analysis to the portfolio management agent
    if agent_config.get("portfolio_management_agent"):
        # Add an edge from the risk management agent to the portfolio management agent
        workflow.add_edge("risk_management_agent", "portfolio_management_agent")

    # End with the portfolio management agent
    if agent_config.get("portfolio_management_agent"):
        # Add an edge from the portfolio management agent to the end node
        workflow.add_edge("portfolio_management_agent", END)

    # Compile and run the workflow
    app = workflow.compile()
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    # Return the final trading decision
    return final_state["messages"][-1].content

if __name__ == "__main__":
    print("\nStarting the ai agent hedge fund...")

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
    default_initial_cash = 100000.0

    # Collect inputs interactively
    tickers = input(f"\nEnter stock tickers (comma-separated) [Default: {default_tickers}]: ") or default_tickers
    start_date = input(f"Enter start date (YYYY-MM-DD) [Default: {default_start_date}]: ") or default_start_date
    end_date = input(f"Enter end date (YYYY-MM-DD) [Default: {default_end_date}]: ") or default_end_date
    initial_cash = input(f"Enter initial cash amount [Default: {default_initial_cash}]: ") or default_initial_cash

    # Enable/disable agents interactively
    print("\nEnter agents to enable or disable (comma-separated, leave empty to keep defaults):")
    enable_agents = input("Agents to enable: ")
    disable_agents = input("Agents to disable: ")
    show_reasoning = input("Show reasoning for each agent? (yes/no) [Default: no]: ").strip().lower() == "yes"

    # Validate dates
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")

    # Parse the tickers
    tickers = tickers.split(",")

    # Initialize the portfolio
    portfolio = {
        "cash": float(initial_cash),
        "positions": {ticker: {"shares": 0, "value": 0} for ticker in tickers},
    }

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

    # Process each ticker
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        result = run_hedge_fund(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            agent_config=AGENT_CONFIG,
            show_reasoning=show_reasoning
        )
        print(f"Result for {ticker}:")
        print(result)

    print("\nFinal Portfolio:")
    print(portfolio)
