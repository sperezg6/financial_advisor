import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import json




class DataCollectorAgent:
    """
        Agent responsible for collecting stock data from Yahoo Finance
    """

    def __init__(self, llm:None, data_dir="./data"):
        # Initializing the agent

        self.data_dir = data_dir
        self.llm = llm if llm else FinancialAdvisorLLM()

        # Pre-defined ETF and index categories
        self.market_segments = {
            "us_broad_market": ["SPY", "VTI", "IVV"],  # S&P 500, Total Market, S&P 500
            "us_tech": ["QQQ", "XLK", "VGT"],  # Nasdaq 100, Tech Sector, Tech Sector
            "emerging_markets": ["VWO", "IEMG", "EEM"],  # Emerging Markets ETFs
            "dividend_focus": ["VYM", "SCHD", "HDV"],  # High Dividend ETFs
            "fixed_income": ["AGG", "BND", "VCIT"],  # Bond ETFs
            "real_estate": ["VNQ", "IYR", "USRT"]  # Real Estate ETFs
        }

        # Individual stocks by sector
        self.individual_stocks = {
            "technology": [
                "AAPL",  # Apple
                "MSFT",  # Microsoft
                "NVDA",  # NVIDIA
                "GOOGL", # Alphabet (Google)
                "AMZN",  # Amazon
                "META",  # Meta Platforms (Facebook)
                "TSLA",  # Tesla
                "TSM",   # Taiwan Semiconductor
                "AVGO",  # Broadcom
                "ADBE",  # Adobe
                "CRM",   # Salesforce
                "ORCL",  # Oracle
                "AMD",   # Advanced Micro Devices
                "INTC",  # Intel
                "IBM"    # IBM
            ],
            "healthcare": [
                "JNJ",   # Johnson & Johnson
                "UNH",   # UnitedHealth Group
                "LLY",   # Eli Lilly
                "PFE",   # Pfizer
                "ABBV",  # AbbVie
                "MRK",   # Merck
                "TMO",   # Thermo Fisher Scientific
                "ABT",   # Abbott Laboratories
                "DHR",   # Danaher
                "BMY"    # Bristol Myers Squibb
            ],
            "financials": [
                "JPM",   # JPMorgan Chase
                "BAC",   # Bank of America
                "V",     # Visa
                "MA",    # Mastercard
                "WFC",   # Wells Fargo
                "MS",    # Morgan Stanley
                "GS",    # Goldman Sachs
                "BLK",   # BlackRock
                "C",     # Citigroup
                "AXP"    # American Express
            ],
            "consumer": [
                "PG",    # Procter & Gamble
                "KO",    # Coca-Cola
                "PEP",   # PepsiCo
                "COST",  # Costco
                "WMT",   # Walmart
                "HD",    # Home Depot
                "MCD",   # McDonald's
                "NKE",   # Nike
                "SBUX",  # Starbucks
                "TGT"    # Target
            ],
            "industrials": [
                "CAT",   # Caterpillar
                "DE",    # Deere & Company
                "HON",   # Honeywell
                "UPS",   # United Parcel Service
                "BA",    # Boeing
                "RTX",   # Raytheon Technologies
                "GE",    # General Electric
                "LMT",   # Lockheed Martin
                "MMM",   # 3M
                "UNP"    # Union Pacific
            ],
            "energy": [
                "XOM",   # Exxon Mobil
                "CVX",   # Chevron
                "COP",   # ConocoPhillips
                "SLB",   # Schlumberger
                "EOG",   # EOG Resources
                "MPC",   # Marathon Petroleum
                "PSX",   # Phillips 66
                "OXY",   # Occidental Petroleum
                "PXD",   # Pioneer Natural Resources
                "VLO"    # Valero Energy
            ],
            "utilities": [
                "NEE",   # NextEra Energy
                "DUK",   # Duke Energy
                "SO",    # Southern Company
                "D",     # Dominion Energy
                "AEP"    # American Electric Power
            ],
            "telecommunications": [
                "VZ",    # Verizon
                "T",     # AT&T
                "TMUS",  # T-Mobile
                "CMCSA", # Comcast
                "CHTR"   # Charter Communications
            ],
            "materials": [
                "LIN",   # Linde
                "APD",   # Air Products & Chemicals
                "ECL",   # Ecolab
                "FCX",   # Freeport-McMoRan
                "NEM"    # Newmont
            ],
            "emerging_markets_stocks": [
                "BABA",  # Alibaba (China)
                "JD",    # JD.com (China)
                "PDD",   # PDD Holdings (China)
                "TCEHY", # Tencent (China)
                "BIDU",  # Baidu (China)
                "NTES",  # NetEase (China)
                "MELI",  # MercadoLibre (Latin America)
                "GRAB",  # Grab (Southeast Asia)
                "SE",    # Sea Limited (Southeast Asia)
                "RELIANCE.NS", # Reliance Industries (India)
                "INFY",  # Infosys (India)
                "9988.HK", # Alibaba Hong Kong
                "0700.HK"  # Tencent Hong Kong
            ]
        }


    def get_market_data(self, tickers=None, period="1y", interval="1d"):
        """
        Retrive historical market data for specified tickers
            
        Args:
            tickers (list): List of ticker symbols
            period (str): Time period (e.g., "1y", "5y")
            interval (str): Data interval (e.g., "1d", "1wk", "1mo")
            
        Returns:
            pd.DataFrame: Historical price data
        
        
        """

        if tickers is None:
            tickers =[]

            # Add ETFs

            for category, symbols in self.market_segments.items():
                tickers.extend(symbols[:1])

            for sector, stocks in self.individual_stocks.items():
                tickers.extend(stocks[:2])

            
        # Download historical data
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker")

        # Save to CSV
        file_name = os.path.join(self.data_dir, f"market_data_{interval}_{period}.csv")
        data.to_csv(file_name)

        print(f"Downloaded data for {len(tickers)} tickers: {', '.join(tickers)}")

        return data 
    

    def get_sector_stocks(self, sectors=None, count=5):
        """
        Get a selection of individual stocks from specified sectors.
        Args:
            sectors (list): List of sector names
            count (int): Number of stocks to retrieve per sector
        Returns:
            list: List of selected stocks
        """

        if sectors is None:
            sectors = list(self.individual_stocks.keys())

        
        # Filter to only include valid sectors
        valid_sectors = [s for s in sectors if s in self.individual_stocks]

        # Get stocks from each sector
        stocks = []
        for sector in valid_sectors:
            sector_stocks = self.individual_stocks.get(sector, [])
            stocks.extend(sector_stocks[:count])

        return stocks
    
    def get_etf_info(self, etfs=None):
        """
        Get detailed information about ETFs
        Args:
            etfs (list): List of ETF ticker symbols
        
        Returns:
            dict: Dictionary containing ETF information
        """

        if etfs is None:
            etfs = []
            for category, symbols in self.market_segments.items():
                etfs.extend(symbols[:1])
        
        etf_data = {}

        for etf in etfs:
            try:
                ticker = yf.Ticker(etf)
                info = ticker.info

                etf_data[etf] = {
                    "name": info.get("shortName", "N/A"),
                    "asset_class": info.get("assetClass", "N/A"),
                    "category": info.get("category", "N/A"),
                    "expense_ratio": info.get("expenseRatio", "N/A"),
                    "description": info.get("longBusinessSummary", "N/A"),
                    "returns": {
                        "1m": info.get("oneMonthReturn", "N/A"),
                        "3m": info.get("threeMonthReturn", "N/A"),
                        "ytd": info.get("ytdReturn", "N/A"),
                        "1y": info.get("oneYearReturn", "N/A"),
                        "3y": info.get("threeYearReturn", "N/A"),
                        "5y": info.get("fiveYearReturn", "N/A")
                    }
                }

            except Exception as e:
                print(f"Error retrieving data for {etf}: {e}")
                etf_data[etf] ={"error": str(e)}

        # Save to json
        file_name = os.path.join(self.data_dir, "etf_info.json")
        with open(file_name, "w") as f:
            json.dump(etf_data, f, indent=4)


        return etf_data            


    def get_stock_info(self, tickers=None):
        """
        Get detailed information about individual stocks
        """

        if tickers is None:
            # If no tickers provided, use a selection from each sector
            tickers = self.get_sector_stocks(count=2)
        
        stock_data = {}

        for ticker_symbol in tickers:
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info

                stock_data[ticker_symbol] = {
                    "name": info.get("shortName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "market_cap": info.get("marketCap", "N/A"),
                    "pe_ratio": info.get("trailingPE", "N/A"),
                    "forward_pe": info.get("forwardPE", "N/A"),
                    "dividend_yield": info.get("dividendYield", "N/A"),
                    "beta": info.get("beta", "N/A"),
                    "52wk_high": info.get("fiftyTwoWeekHigh", "N/A"),
                    "52wk_low": info.get("fiftyTwoWeekLow", "N/A"),
                    "description": info.get("longBusinessSummary", "N/A")
                }
            except Exception as e:
                print(f"Error retrieving data for {ticker_symbol}: {e}")
                stock_data[ticker_symbol] = {"error": str(e)}
            
        # Save to json
        filename = os.path.join(self.data_dir, "stock_info.json")
        with open(filename, "w") as f:
            json.dump(stock_data, f, indent=4)
        
        return stock_data



    def get_recommended_assets(self, risk_tolerance, time_horizon, focus_sectors=None):
        """
            Ask the LLM for recommended assets based on user preferences.
        
        """

        sectors_text = "any promising sectors" if not focus_sectors else f"the following sectors: {', '.join(focus_sectors)}"

        system_prompt = """You are a financial advisor with expertise in stocks, ETFs, and portfolio construction.
        Your recommendations should be evidence-based, consider macroeconomic factors, and align with the client's risk profile.
        Always provide real, currently existing assets with accurate ticker symbols."""
        

        prompt = f"""Recommend investment assets (both stocks and ETFs) for a client with:
        - Risk tolerance: {risk_tolerance}/10 (where 1 is very conservative and 10 is very aggressive)
        - Investment time horizon: {time_horizon} years
        - Interested in: {sectors_text}
        - Market focus: US markets and emerging markets
        
        Provide 5-8 stocks and 3-5 ETFs that would be suitable.
        
        For each asset, include:
        1. Ticker symbol
        2. Full name
        3. Asset type (stock/ETF)
        4. Brief justification (1-2 sentences)
        
        Format your response as a JSON list of objects with keys: ticker, name, type, and justification.

        """

        # Format instructions
        format_instruction = """
         Format your response as valid JSON like this:
        ```json
        [
          {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "type": "stock",
            "justification": "Strong balance sheet and consistent returns."
          },
          ...
        ]
        ```
        """

        response = self.llm.get_structured_response(
            prompt, 
            system_prompt=system_prompt,
            format_instructions=format_instruction
        )

        # Extract JSON from response
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                recommended_assets = json.loads(json_str)
            else:
                # Try to find JSON if it's wrapped in code blocks
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    recommended_assets = json.loads(json_str)
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    recommended_assets = json.loads(json_str)
                else:
                    raise ValueError("Could not find JSON in response")
            
            # Save to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.data_dir, f"recommended_assets_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(recommended_assets, f, indent=4)
            
            return recommended_assets

        except Exception as e:
                print(f"Error parsing LLM response: {e}")
                print(f"Raw response: {response}")
                return []