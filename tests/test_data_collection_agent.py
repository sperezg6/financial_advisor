from portfolio_advisor.agents.data_collection import DataCollectorAgent
from portfolio_advisor.models.base_model import FinancialAdvisorLLM
import pandas as pd
import os

def test_individual_stocks_retrieval():
    """
    Test retrieval of individual stocks data using the updated DataCollectionAgent
    """
    print("Testing Individual Stocks Retrieval")
    print("-" * 50)
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Initialize the agent
    agent = DataCollectorAgent(llm=FinancialAdvisorLLM,data_dir="./data")
    
    # 1. Test getting stocks from specific sectors
    print("\n1. Testing sector-specific stock retrieval...")
    tech_stocks = agent.get_sector_stocks(sectors=["technology"], count=3)
    print(f"Technology stocks: {', '.join(tech_stocks)}")
    
    health_energy_stocks = agent.get_sector_stocks(sectors=["healthcare", "energy"], count=2)
    print(f"Healthcare & Energy stocks: {', '.join(health_energy_stocks)}")
    
    emerging_market_stocks = agent.get_sector_stocks(sectors=["emerging_markets_stocks"], count=4)
    print(f"Emerging markets stocks: {', '.join(emerging_market_stocks)}")
    
    # 2. Test retrieving market data for a selection of stocks
    print("\n2. Testing market data retrieval for selected stocks...")
    test_tickers = tech_stocks[:2] + health_energy_stocks[:2] + ["SPY"]  # Add SPY for reference
    print(f"Retrieving data for: {', '.join(test_tickers)}")
    
    # Get 30 days of daily data
    market_data = agent.get_market_data(test_tickers, period="30d", interval="1d")
    
    # Display the last 3 days of closing prices
    if not market_data.empty:
        print("\nLast 3 days of closing prices:")
        if len(test_tickers) > 1:
            # Multiple tickers - get Close prices
            closes = pd.DataFrame({ticker: market_data[ticker]["Close"] for ticker in test_tickers})
            print(closes.tail(3))
        else:
            # Single ticker
            print(market_data["Close"].tail(3))
    
    # 3. Test getting detailed information about stocks
    print("\n3. Testing detailed stock information retrieval...")
    stock_info = agent.get_stock_info(test_tickers[:2])  # Just get info for first 2 tickers
    
    # Display basic info for each stock
    for ticker, info in stock_info.items():
        print(f"\n{ticker} - {info.get('name', 'N/A')}:")
        print(f"  Sector: {info.get('sector', 'N/A')}")
        print(f"  Industry: {info.get('industry', 'N/A')}")
        print(f"  Market Cap: {info.get('market_cap', 'N/A')}")
        print(f"  P/E Ratio: {info.get('pe_ratio', 'N/A')}")
        print(f"  Dividend Yield: {info.get('dividend_yield', 'N/A')}")
    
    print("\nTest completed! Check the ./data directory for saved files.")

if __name__ == "__main__":
    test_individual_stocks_retrieval()