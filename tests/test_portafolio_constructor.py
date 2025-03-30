import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from portfolio_advisor.agents.portofolio_constructor import PortfolioConstructorAgent
from portfolio_advisor.agents.risk_assesment import RiskAssessmentAgent

def test_portfolio_constructor_agent():
    """
    Test the functionality of the Portfolio Constructor Agent
    """
    print("Testing Portfolio Constructor Agent")
    print("=" * 50)
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Initialize agent
    portfolio_agent = PortfolioConstructorAgent(data_dir="./data")
    
    # 1. Test allocation template creation
    print("\n1. Testing allocation template creation...")
    
    # Test across different risk levels
    risk_levels = [1, 3, 5, 7, 10]
    
    for risk_level in risk_levels:
        template = portfolio_agent.create_allocation_template([], risk_level, time_horizon=5)
        print(f"\nRisk Level {risk_level}/10 Template:")
        for asset_class, allocation in template.items():
            print(f"  {asset_class}: {allocation * 100:.1f}%")
    
    # 2. Test with sample assets and metrics
    print("\n2. Testing portfolio allocation generation...")
    
    # Sample assets and their information
    sample_assets = ["AAPL", "MSFT", "JNJ", "VTI", "BND", "GLD", "VWO"]
    
    # Sample asset information
    asset_info = {
        "AAPL": {"name": "Apple Inc.", "type": "stock", "sector": "Technology"},
        "MSFT": {"name": "Microsoft Corp.", "type": "stock", "sector": "Technology"},
        "JNJ": {"name": "Johnson & Johnson", "type": "stock", "sector": "Healthcare"},
        "VTI": {"name": "Vanguard Total Stock Market ETF", "type": "etf", "category": "Equity"},
        "BND": {"name": "Vanguard Total Bond Market ETF", "type": "etf", "category": "Bond"},
        "GLD": {"name": "SPDR Gold Shares", "type": "etf", "category": "Commodity"},
        "VWO": {"name": "Vanguard Emerging Markets ETF", "type": "etf", "category": "Equity"}
    }
    
    # Create sample risk metrics
    # Note: We'll generate synthetic metrics for testing
    risk_metrics = pd.DataFrame(index=sample_assets)
    
    # Simulated risk metrics (just for testing)
    risk_metrics["volatility"] = [0.25, 0.28, 0.15, 0.18, 0.05, 0.20, 0.30]
    risk_metrics["max_drawdown"] = [-0.35, -0.30, -0.20, -0.25, -0.10, -0.25, -0.40]
    risk_metrics["avg_return"] = [0.15, 0.18, 0.10, 0.12, 0.03, 0.08, 0.14]
    risk_metrics["sharpe_ratio"] = [0.60, 0.64, 0.67, 0.67, 0.60, 0.40, 0.47]
    
    # Test allocations with different risk levels
    test_risk_levels = [3, 5, 8]
    test_horizons = [3, 10, 20]
    
    for risk in test_risk_levels:
        for horizon in test_horizons:
            print(f"\nPortfolio for Risk Level {risk}/10, Time Horizon {horizon} years:")
            
            allocations = portfolio_agent.generate_allocation(
                sample_assets, 
                asset_info,
                risk_metrics,
                risk_level=risk,
                time_horizon=horizon
            )
            
            # Print allocations in percentage format
            for asset, weight in allocations.items():
                print(f"  {asset} ({asset_info[asset]['type']}): {weight * 100:.2f}%")
            
            # Calculate asset class totals
            asset_class_totals = {
                "stocks": sum(allocations[a] for a in allocations if asset_info[a]['type'] == 'stock'),
                "bonds": sum(allocations[a] for a in allocations 
                           if asset_info[a]['type'] == 'etf' and 'bond' in asset_info[a]['category'].lower()),
                "other_etfs": sum(allocations[a] for a in allocations 
                                if asset_info[a]['type'] == 'etf' and 'bond' not in asset_info[a]['category'].lower())
            }
            
            print("\n  Asset Class Totals:")
            for asset_class, total in asset_class_totals.items():
                print(f"    {asset_class}: {total * 100:.2f}%")
            
            # Sanity check that allocations sum to approximately 1
            allocation_sum = sum(allocations.values())
            print(f"\n  Total Allocation: {allocation_sum * 100:.2f}%")
            
            # Create a simple allocation pie chart
            try:
                plt.figure(figsize=(10, 6))
                plt.pie(
                    allocations.values(), 
                    labels=[f"{k} ({asset_info[k]['type']})" for k in allocations.keys()],
                    autopct='%1.1f%%'
                )
                plt.title(f"Portfolio Allocation - Risk {risk}/10, Horizon {horizon} years")
                plt.axis('equal')
                
                # Save the chart
                chart_file = f"./data/portfolio_risk{risk}_horizon{horizon}.png"
                plt.savefig(chart_file)
                print(f"\n  Allocation chart saved to: {chart_file}")
                plt.close()
            except Exception as e:
                print(f"  Error creating chart: {e}")
    
    # 3. Test with real data if risk assessment agent is available
    print("\n3. Testing with real market data...")
    try:
        # Initialize risk agent
        risk_agent = RiskAssessmentAgent(data_dir="./data")
        
        # Define test tickers
        real_tickers = ["AAPL", "MSFT", "VTI", "BND", "GLD"]
        
        # Get real risk metrics
        print("  Calculating risk metrics from market data...")
        real_risk_metrics = risk_agent.calculate_risk_metrics(real_tickers, period="1y", interval="1d")
        
        # Calculate correlation matrix
        real_correlation = risk_agent.calculate_correlation_matrix(real_tickers, period="1y", interval="1d")
        
        # Create a sample portfolio allocation
        print("  Generating portfolio allocation...")
        real_asset_info = {ticker: {"type": "stock" if ticker in ["AAPL", "MSFT"] else "etf", 
                                   "category": "Bond" if ticker == "BND" else "Equity"} 
                          for ticker in real_tickers}
        
        real_allocations = portfolio_agent.generate_allocation(
            real_tickers,
            real_asset_info,
            real_risk_metrics,
            risk_level=5,  # Moderate risk
            time_horizon=10
        )
        
        print("\n  Real Market Data Portfolio (Risk 5/10, 10-year horizon):")
        for ticker, allocation in real_allocations.items():
            print(f"    {ticker}: {allocation * 100:.2f}%")
        
        # Calculate portfolio risk
        port_metrics = risk_agent.assess_portfolio_risk(
            real_risk_metrics,
            real_correlation,
            real_allocations
        )
        
        print("\n  Portfolio Risk Metrics:")
        for metric, value in port_metrics.items():
            print(f"    {metric}: {value:.4f}")
        
        # Check if it matches risk tolerance
        risk_match = risk_agent.assess_risk_tolerance_match(port_metrics, 5)
        print(f"\n  Risk Match: {risk_match['risk_match']}")
        print(f"  Recommendation: {risk_match['recommendation']}")
        
    except Exception as e:
        print(f"  Error testing with real data: {e}")
        print("  Skipping real data test.")
    
    print("\nPortfolio Constructor Agent test completed!")

if __name__ == "__main__":
    test_portfolio_constructor_agent()