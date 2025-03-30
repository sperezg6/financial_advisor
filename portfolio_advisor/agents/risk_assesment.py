import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import yfinance as yf
from portfolio_advisor.models.base_model import FinancialAdvisorLLM

class RiskAssessmentAgent:
    """Agent responsible for assessing risk of recommended assets"""
    
    def __init__(self, llm=None, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize LLM
        self.llm = llm if llm else FinancialAdvisorLLM()
    
    def calculate_risk_metrics(self, tickers, period="1y", interval="1d"):
        """
        Calculate key risk metrics for a list of tickers
        
        Args:
            tickers (list): List of ticker symbols
            period (str): Time period for historical data
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: Risk metrics for each asset
        """
        # Download historical data
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker")
        
        # Initialize results DataFrame
        risk_metrics = pd.DataFrame(index=tickers)
        
        # Calculate metrics for each ticker
        for ticker in tickers:
            try:
                # Get price data for this ticker
                if len(tickers) > 1:
                    # For multiple tickers, data is structured with tickers as the first level
                    if 'Adj Close' in data[ticker].columns:
                        prices = data[ticker]['Adj Close']
                    else:
                        # If Adj Close is not available, use Close
                        prices = data[ticker]['Close']
                else:
                    # For a single ticker, data doesn't have ticker as the first level
                    if 'Adj Close' in data.columns:
                        prices = data['Adj Close']
                    else:
                        prices = data['Close']
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                # Annualization factor
                if interval == "1d":
                    annualize_factor = np.sqrt(252)  # Trading days in a year
                elif interval == "1wk":
                    annualize_factor = np.sqrt(52)   # Weeks in a year
                elif interval == "1mo":
                    annualize_factor = np.sqrt(12)   # Months in a year
                else:
                    annualize_factor = 1
                
                # Calculate metrics
                risk_metrics.at[ticker, "volatility"] = returns.std() * annualize_factor
                risk_metrics.at[ticker, "max_drawdown"] = (prices / prices.cummax() - 1).min()
                risk_metrics.at[ticker, "avg_return"] = returns.mean() * annualize_factor
                risk_metrics.at[ticker, "sharpe_ratio"] = (returns.mean() / returns.std()) * annualize_factor if returns.std() > 0 else 0
                
                # Calculate downside deviation (only negative returns)
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    risk_metrics.at[ticker, "downside_deviation"] = negative_returns.std() * annualize_factor
                else:
                    risk_metrics.at[ticker, "downside_deviation"] = 0
                
            except Exception as e:
                print(f"Error calculating metrics for {ticker}: {e}")
                risk_metrics.loc[ticker] = np.nan
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"risk_metrics_{timestamp}.csv")
        risk_metrics.to_csv(filename)
        
        return risk_metrics
    
    def calculate_correlation_matrix(self, tickers, period="1y", interval="1d"):
        """
        Calculate correlation matrix between assets
        
        Args:
            tickers (list): List of ticker symbols
            period (str): Time period for historical data
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Download historical data
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker")
        
        # Initialize returns DataFrame
        returns = pd.DataFrame(index=data.index)
        
        # Extract returns for each ticker
        for ticker in tickers:
            try:
                # Get price data for this ticker
                if 'Adj Close' in data[ticker].columns:
                    price_series = data[ticker]['Adj Close']
                else:
                    price_series = data[ticker]['Close']
                    
                # Calculate returns and add to the DataFrame
                returns[ticker] = price_series.pct_change()
            except Exception as e:
                print(f"Error processing {ticker} for correlation: {e}")
        
        # Drop NA values (first row will have NaN due to pct_change)
        returns = returns.dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"correlation_matrix_{timestamp}.csv")
        correlation_matrix.to_csv(filename)
        
        return correlation_matrix
    
    def assess_portfolio_risk(self, risk_metrics, correlation_matrix, asset_allocations):
        """
        Assess risk of the entire portfolio
        
        Args:
            risk_metrics (pd.DataFrame): Risk metrics for individual assets
            correlation_matrix (pd.DataFrame): Correlation matrix between assets
            asset_allocations (dict): Allocation percentages for each asset
            
        Returns:
            dict: Portfolio risk metrics
        """
        # Get tickers that exist in all datasets
        tickers = [t for t in asset_allocations.keys() 
                  if t in risk_metrics.index and t in correlation_matrix.index]
        
        # If no valid tickers, return empty metrics
        if not tickers:
            return {
                "volatility": 0,
                "expected_return": 0,
                "sharpe_ratio": 0,
                "diversification_score": 0
            }
        
        # Get weights for valid tickers
        weights = np.array([asset_allocations[ticker] for ticker in tickers])
        
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Filter metrics and correlation for our tickers
        filtered_metrics = risk_metrics.loc[tickers].copy()
        filtered_corr = correlation_matrix.loc[tickers, tickers].copy()
        
        # Replace NaN with 0 for calculations
        filtered_metrics = filtered_metrics.fillna(0)
        filtered_corr = filtered_corr.fillna(0)
        
        # Calculate portfolio volatility
        try:
            asset_vols = filtered_metrics["volatility"].values
            port_variance = np.dot(weights.T, np.dot(filtered_corr.values * np.outer(asset_vols, asset_vols), weights))
            port_volatility = np.sqrt(port_variance)
        except Exception as e:
            print(f"Error calculating portfolio volatility: {e}")
            port_volatility = np.nan
        
        # Calculate portfolio return
        try:
            port_return = np.sum(filtered_metrics["avg_return"].values * weights)
        except Exception as e:
            print(f"Error calculating portfolio return: {e}")
            port_return = np.nan
        
        # Calculate portfolio Sharpe ratio
        try:
            port_sharpe = port_return / port_volatility if port_volatility > 0 else 0
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            port_sharpe = np.nan
        
        # Calculate diversification score
        try:
            diversification_score = 1 - float(np.sqrt(np.sum(weights**2)))
        except Exception as e:
            print(f"Error calculating diversification score: {e}")
            diversification_score = np.nan
        
        # Portfolio metrics
        portfolio_metrics = {
            "volatility": float(port_volatility) if not np.isnan(port_volatility) else 0,
            "expected_return": float(port_return) if not np.isnan(port_return) else 0,
            "sharpe_ratio": float(port_sharpe) if not np.isnan(port_sharpe) else 0,
            "diversification_score": float(diversification_score) if not np.isnan(diversification_score) else 0
        }
        
        return portfolio_metrics
    
    def assess_risk_tolerance_match(self, portfolio_metrics, user_risk_tolerance):
        """
        Assess whether portfolio risk matches user's risk tolerance
        
        Args:
            portfolio_metrics (dict): Portfolio risk metrics
            user_risk_tolerance (int): User's risk tolerance (1-10)
            
        Returns:
            dict: Risk assessment
        """
        # Convert risk tolerance to expected volatility range
        # Higher risk tolerance = higher acceptable volatility
        volatility_ranges = {
            1: (0.00, 0.05),  # Very conservative
            2: (0.03, 0.07),
            3: (0.05, 0.09),
            4: (0.07, 0.11),
            5: (0.09, 0.13),  # Moderate
            6: (0.11, 0.15),
            7: (0.13, 0.17),
            8: (0.15, 0.20),
            9: (0.18, 0.25),
            10: (0.22, 0.35)  # Very aggressive
        }
        
        # Get volatility range for user's risk tolerance
        min_vol, max_vol = volatility_ranges.get(user_risk_tolerance, (0.09, 0.13))
        portfolio_vol = portfolio_metrics["volatility"]
        
        # Check if portfolio volatility is within acceptable range
        if portfolio_vol < min_vol:
            risk_match = "too_conservative"
            recommendation = "Consider adding more growth-oriented assets to match your risk tolerance."
        elif portfolio_vol > max_vol:
            risk_match = "too_aggressive"
            recommendation = "Consider reducing exposure to volatile assets to better match your risk tolerance."
        else:
            risk_match = "appropriate"
            recommendation = "The portfolio risk level is well-aligned with your risk tolerance."
        
        assessment = {
            "portfolio_volatility": portfolio_vol,
            "acceptable_range": {"min": min_vol, "max": max_vol},
            "risk_match": risk_match,
            "recommendation": recommendation
        }
        
        return assessment
    
    def get_risk_analysis(self, risk_metrics, user_risk_tolerance):
        """
        Get LLM-generated risk analysis of assets
        
        Args:
            risk_metrics (pd.DataFrame): Risk metrics for assets
            user_risk_tolerance (int): User's risk tolerance (1-10)
            
        Returns:
            dict: Risk analysis
        """
        # Convert risk metrics to a string for the prompt
        metrics_str = risk_metrics.to_string()
        
        system_prompt = """You are a financial risk analyst specializing in portfolio risk assessment.
        Your analysis should be evidence-based and tailored to the client's risk tolerance.
        Focus on providing actionable insights about which assets align with the client's risk profile."""
        
        prompt = f"""Analyze the following risk metrics for a set of investment assets:
        
        {metrics_str}
        
        The client has indicated a risk tolerance of {user_risk_tolerance} on a scale of 1-10 (where 1 is very conservative and 10 is very aggressive).
        
        Provide a risk analysis with:
        1. Which assets are most appropriate for this risk tolerance
        2. Which assets may be too risky 
        3. Which assets may be too conservative
        4. How these assets might be combined to create a suitable portfolio
        
        Format your response as a JSON object with the following structure:
        ```json
        {{
          "appropriate_assets": ["List of tickers"],
          "too_risky_assets": ["List of tickers"],
          "too_conservative_assets": ["List of tickers"],
          "portfolio_strategy": "Description of strategy",
          "reasoning": "Your explanation"
        }}
        ```
        """
        
        response = self.llm.get_structured_response(prompt, system_prompt=system_prompt)
        
        # Parse JSON from response
        try:
            if "```json" in response:
                json_start = response.find("```json") + 8
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                
            risk_analysis = json.loads(json_str)
            
            # Save to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.data_dir, f"risk_analysis_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(risk_analysis, f, indent=4)
                
            return risk_analysis
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response}")
            return {
                "appropriate_assets": [],
                "too_risky_assets": [],
                "too_conservative_assets": [],
                "portfolio_strategy": "Error in processing response",
                "reasoning": "Could not parse structured data from the LLM response."
            }