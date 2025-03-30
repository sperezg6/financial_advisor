import yfinance as yf
import pandas as pd
from datetime import datetime

def test_stock_data_retrieval():
    """
    Test basic stock data retrieval using yfinance
    """
    print("Testing YFinance Stock Data Retrieval")
    print("-" * 50)
    
    # List of tickers to test
    tickers = [ "AMZN"]
    
    # Get current date for display
    current_date = datetime.now().strftime("%Y-%m-%d")
    print(f"Current Date: {current_date}")
    print(f"Testing data retrieval for: {', '.join(tickers)}")
    print("-" * 50)
    
    try:
        # Retrieve historical data (last 10 days)
        print("\nRetrieving historical price data (last 10 days)...")
        hist_data = yf.download(tickers, period="10d")
        
        # Show the first few rows of closing prices
        print("\nClose Prices (first 3 days):")
        print(hist_data["Close"].head(3))
        
        # Get some basic statistics
        print("\nBasic Statistics (Close Prices):")
        print(hist_data["Close"].describe())
        
        # Test getting information for a single stock
        print("\nRetrieving detailed info for Apple (AAPL)...")
        apple = yf.Ticker("AAPL")
        info = apple.info
        
        # Display key information
        print("\nKey information for AAPL:")
        key_info = {
            "Name": info.get("shortName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Website": info.get("website", "N/A")
        }
        
        for key, value in key_info.items():
            print(f"{key}: {value}")
        
        # Get recent news
        print("\nRecent news headlines for AAPL:")
        news = apple.news
        for i, article in enumerate(news[:3]):  # Show first 3 news items
            print(f"{i+1}. {article.get('title', 'No title')} ({article.get('publisher', 'Unknown')})")
        
        # Test ETF data
        print("\nRetrieving ETF data for SPY...")
        spy = yf.Ticker("SPY")
        spy_info = spy.info
        
        print("\nKey information for SPY:")
        key_etf_info = {
            "Name": spy_info.get("shortName", "N/A"),
            "Category": spy_info.get("category", "N/A"),
            "Total Assets": spy_info.get("totalAssets", "N/A"),
            "Expense Ratio": spy_info.get("expenseRatio", "N/A"),
            "YTD Return": spy_info.get("ytdReturn", "N/A"),
            "Three Year Return": spy_info.get("threeYearAverageReturn", "N/A"),
        }
        
        for key, value in key_etf_info.items():
            print(f"{key}: {value}")
        
        print("\nYFinance test completed successfully!")
        
    except Exception as e:
        print(f"\nError during YFinance test: {e}")
        
if __name__ == "__main__":
    test_stock_data_retrieval()