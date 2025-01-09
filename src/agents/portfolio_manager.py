from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from agents.state import AgentState, show_agent_reasoning


from tools.api import prices_to_df

from dotenv import load_dotenv, find_dotenv
import os
import ast
import json

# # Check if .env file is found
# dotenv_path = find_dotenv()
# print(f".env file found at: {dotenv_path}")  # Debugging

# # Load the environment variables
# load_dotenv(dotenv_path=dotenv_path)

# # Check if the API key is loaded
# api_key = os.getenv("OPENAI_API_KEY")
# print(f"Loaded API Key: {api_key[:4]}...")  # Prints the first 4 characters
# print(f"Current working directory: {os.getcwd()}")

##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Get the technical analyst, fundamentals agent, sentiment agent, valuation agent, and risk management agent messages
    technical_message = next((msg for msg in state["messages"] if msg.name == "technical_analyst_agent"), None)
    fundamentals_message = next((msg for msg in state["messages"] if msg.name == "fundamentals_agent"), None)
    sentiment_message = next((msg for msg in state["messages"] if msg.name == "sentiment_agent"), None)
    valuation_message = next((msg for msg in state["messages"] if msg.name == "valuation_agent"), None)
    risk_message = next((msg for msg in state["messages"] if msg.name == "risk_management_agent"), None)

    # Safely process each message
    technical_signals = {}
    fundamental_signals = {}
    sentiment_signals = {}
    valuation_signals = {}
    risk_signals = {}

    # Parse each message content if it exists
    try:
        if technical_message and technical_message.content.strip():
            technical_signals = json.loads(technical_message.content)
        if fundamentals_message and fundamentals_message.content.strip():
            fundamental_signals = json.loads(fundamentals_message.content)
        if sentiment_message and sentiment_message.content.strip():
            sentiment_signals = json.loads(sentiment_message.content)
        if valuation_message and valuation_message.content.strip():
            valuation_signals = json.loads(valuation_message.content)
        if risk_message and risk_message.content.strip():
            risk_signals = json.loads(risk_message.content)

            # Extract risk management constraints
            max_position_size = risk_signals.get("max_position_size", 0)
            trading_action = risk_signals.get("trading_action", "hold")

            # Calculate feasible quantity based on risk constraints
            prices_df = prices_to_df(state["data"]["prices"])
            current_price = prices_df["close"].iloc[-1]
            ticker = state["data"]["ticker"]

            # Ensure the ticker exists in the portfolio
            current_position = portfolio["positions"].get(ticker, {})
            shares_owned = current_position.get("shares", 0)

            if trading_action == "buy":
                feasible_quantity = min(int(max_position_size / current_price), int(portfolio["cash"] / current_price))
            elif trading_action == "sell":
                feasible_quantity = min(int(max_position_size / current_price), shares_owned)
            else:
                feasible_quantity = 0  # Default for "hold"

    except Exception as e:
        # Handle fallback parsing with `ast.literal_eval` in case of JSON errors
        if technical_message and technical_message.content.strip():
            technical_signals = ast.literal_eval(technical_message.content)
        if fundamentals_message and fundamentals_message.content.strip():
            fundamental_signals = ast.literal_eval(fundamentals_message.content)
        if sentiment_message and sentiment_message.content.strip():
            sentiment_signals = ast.literal_eval(sentiment_message.content)
        if valuation_message and valuation_message.content.strip():
            valuation_signals = ast.literal_eval(valuation_message.content)
        if risk_message and risk_message.content.strip():
            risk_signals = ast.literal_eval(risk_message.content)

            # Extract risk management constraints in fallback
            max_position_size = risk_signals.get("max_position_size", 0)
            trading_action = risk_signals.get("trading_action", "hold")
            feasible_quantity = 0  # Default in fallback scenario


    # Combine signals into agent signals dictionary for further processing if needed
    agent_signals = {
        "technical": technical_signals,
        "fundamental": fundamental_signals,
        "sentiment": sentiment_signals,
        "valuation": valuation_signals,
        "risk": risk_signals,
    }

    # Optional: Print the combined signals for debugging
    # print(agent_signals)



    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions. Your job is to make a trading decision based on the team's analysis while adhering to risk management constraints, with these guidelines:

                    ### RISK MANAGEMENT CONSTRAINTS:
                    - You MUST NOT exceed the max_position_size specified by the risk manager.
                    - If the risk manager specifies a desired_position_size, aim to achieve that position, even if other signals recommend "hold." This applies as long as you have enough cash (for buys) or enough shares (for sells).
                    - If no explicit desired_position_size is provided, follow the trading_action (buy/sell/hold) recommended by the risk manager.

                    ### WEIGHING THE SIGNALS:
                    When weighing signals for direction and timing:
                    1. Valuation Analysis (35% weight)
                    - Primary driver for fair value assessment.
                    - Determines if the price offers a good entry/exit point.

                    2. Fundamental Analysis (30% weight)
                    - Business quality and growth assessment.
                    - Determines conviction in long-term potential.

                    3. Technical Analysis (25% weight)
                    - Secondary confirmation.
                    - Helps with entry/exit timing.

                    4. Sentiment Analysis (10% weight)
                    - Final consideration.
                    - Can influence sizing within risk limits.

                    ### HANDLING MISSING SIGNALS:
                    If one or more signals are missing:
                    - Adjust weights proportionally among the remaining signals.
                    - Use indirect clues or data from other signals to hypothesize plausible market scenarios.
                    - Incorporate hypothetical or unconventional ideas if they align with market trends or historical patterns.

                    ### DECISION-MAKING PRIORITIES:
                    1. **Risk Management First**: Ensure compliance with max_position_size and desired_position_size.
                    2. Evaluate the remaining signals, adjusted for weight changes due to missing data.
                    3. Use technical analysis for timing and sentiment for final adjustments.
                    4. Hypothesize potential market scenarios based on available signals and trends to fill gaps in analysis.

                    ### OUTPUT FORMAT:
                    Provide your output strictly in the following JSON format:
                    {
                    "action": "buy" | "sell" | "hold",
                    "quantity": <positive integer>,
                    "confidence": <float between 0 and 1>,
                    "agent_signals": [
                        {"agent": <string>, "signal": "bullish" | "bearish" | "neutral", "confidence": <float>}
                    ],
                    "reasoning": <concise explanation of the decision, including how you weighted the signals, handled missing data, and aligned with risk manager constraints>
                    }

                    IMPORTANT: 
                    - Do not include additional text or formatting like "```json".
                    - Only include the pure JSON object in your output.
                    - Any deviation from this structure will result in rejection of the output.

                    ### TRADING RULES:
                    - Never exceed risk manager’s max_position_size.
                    - Aim to achieve the desired_position_size set by the risk manager if cash/shares allow.
                    - Only buy if you have available cash.
                    - Only sell if you have shares in the portfolio to sell.
                    - Quantity for sells must be ≤ current position.
                    - Quantity for buys must be ≤ available cash and max_position_size.
                    - For missing signals, hypothesize plausible scenarios to guide decisions within the given constraints.
                    """
            ),
            (
                "human",
                """Based on the team's analysis below, make your trading decision.

                Technical Analysis Trading Signal: {technical_message}
                Fundamental Analysis Trading Signal: {fundamentals_message}
                Sentiment Analysis Trading Signal: {sentiment_message}
                Valuation Analysis Trading Signal: {valuation_message}
                Risk Management Trading Signal: {risk_message}

                Here is the current portfolio:
                Portfolio:
                Cash: {portfolio_cash}
                Current Position: {portfolio_stock} shares

                Notes:
                - If any signal is missing, adjust the weights proportionally and use plausible hypotheses or extrapolations to fill gaps.
                - Hypothesize potential market scenarios based on indirect clues, broader trends, or unconventional insights.
                - Clearly explain how these scenarios influenced your decision-making.

                Only include the action, quantity, reasoning, confidence, and agent_signals in your output as JSON. Do not include any JSON markdown.

                Remember, the action must be either buy, sell, or hold.
                You can only buy if you have available cash.
                You can only sell if you have shares in the portfolio to sell.
                """
            ),
        ]
    )

        # Current ticker being processed
    ticker = state["data"]["ticker"]

    # Ensure the ticker exists in the portfolio
    if ticker not in portfolio["positions"]:
        raise KeyError(f"Ticker {ticker} not found in portfolio!")

    # Get the current position and calculate stock value
    current_position = portfolio["positions"][ticker]
    shares = current_position.get("shares", 0)

    # Fetch the most recent closing price
    prices_df = prices_to_df(state["data"]["prices"])
    current_price = prices_df["close"].iloc[-1]
    current_stock_value = shares * current_price

    # Safely fetch message content or default to an empty string if the message is None
    technical_content = technical_message.content if technical_message else ""
    fundamentals_content = fundamentals_message.content if fundamentals_message else ""
    sentiment_content = sentiment_message.content if sentiment_message else ""
    valuation_content = valuation_message.content if valuation_message else ""
    risk_content = risk_message.content if risk_message else ""

    # Generate the prompt
    prompt = template.invoke(
        {
            "technical_message": technical_content,
            "fundamentals_message": fundamentals_content,
            "sentiment_message": sentiment_content,
            "valuation_message": valuation_content,
            "risk_message": risk_content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": f"{current_stock_value:.2f}"  # Use current stock value for the ticker
        }
    )

    # Invoke the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    result = llm.invoke(prompt)

    # Create the portfolio management message
    message = HumanMessage(
        content=result.content,
        name="portfolio_management"
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}