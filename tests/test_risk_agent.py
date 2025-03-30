import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolio_advisor.agents.risk_assesment import RiskAssessmentAgent
from portfolio_advisor.models.base_model import FinancialAdvisorLLM

def test_risk_assessment_agent():
    """
    Test the functionality of the Risk Assessment Agent
    """
    print("Testing Risk Assessment Agent")
    print("=" * 50)
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Initialize agent
    risk_agent = RiskAssessmentAgent(data_dir="./data")
    
    # Test tickers representing different asset classes
    tickers = [
        "AAPL",    # Tech stock
        "JNJ",     # Healthcare stock
        "XOM",     # Energy stock
        "SPY",     # S&P 500 ETF
        "QQQ",     # Nasdaq ETF
        "VWO",     # Emerging Markets ETF
        "BND",     # Bond ETF
        "GLD"      # Gold ETF
    ]
    
    # Test risk metrics calculation
    print("\n1. Testing risk metrics calculation...")
    try:
        risk_metrics = risk_agent.calculate_risk_metrics(tickers, period="1y", interval="1d")
        
        # Display risk metrics
        print("\nRisk Metrics:")
        print(risk_metrics)
        print("\nRisk Metrics calculation successful!")
    except Exception as e:
        print(f"Error in risk metrics calculation: {e}")
    
    # Test correlation matrix calculation
    print("\n2. Testing correlation matrix calculation...")
    try:
        correlation_matrix = risk_agent.calculate_correlation_matrix(tickers, period="1y", interval="1d")
        
        # Display correlation matrix
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        print("\nCorrelation matrix calculation successful!")
        
        # Visualize correlation matrix if matplotlib is available
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation Coefficient')
            plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
            plt.yticks(range(len(correlation_matrix)), correlation_matrix.index)
            plt.title('Asset Correlation Matrix')
            plt.tight_layout()
            plt.savefig('./data/correlation_matrix.png')
            print("\nCorrelation matrix visualization saved to ./data/correlation_matrix.png")
        except Exception as e:
            print(f"Could not create visualization: {e}")
    except Exception as e:
        print(f"Error in correlation matrix calculation: {e}")
    
    # Test portfolio risk assessment
    print("\n3. Testing portfolio risk assessment...")
    try:
        # Create sample portfolio allocations (equal weight for simplicity)
        sample_allocations = {ticker: 1/len(tickers) for ticker in tickers}
        
        # Calculate portfolio risk
        portfolio_metrics = risk_agent.assess_portfolio_risk(
            risk_metrics, 
            correlation_matrix, 
            sample_allocations
        )
        
        # Display portfolio metrics
        print("\nPortfolio Metrics:")
        for metric, value in portfolio_metrics.items():
            print(f"{metric}: {value:.4f}")
        print("\nPortfolio risk assessment successful!")
    except Exception as e:
        print(f"Error in portfolio risk assessment: {e}")
    
    # Test different risk tolerance levels
    print("\n4. Testing risk tolerance matching...")
    try:
        for risk_level in [2, 5, 8]:  # Low, Medium, High
            print(f"\nRisk Tolerance Level: {risk_level}/10")
            risk_match = risk_agent.assess_risk_tolerance_match(portfolio_metrics, risk_level)
            
            print(f"Portfolio Volatility: {risk_match['portfolio_volatility']:.4f}")
            print(f"Acceptable Range: {risk_match['acceptable_range']['min']:.4f} - {risk_match['acceptable_range']['max']:.4f}")
            print(f"Risk Match: {risk_match['risk_match']}")
            print(f"Recommendation: {risk_match['recommendation']}")
        print("\nRisk tolerance matching successful!")
    except Exception as e:
        print(f"Error in risk tolerance matching: {e}")
    
    # Optional: Test LLM-based risk analysis if Ollama is running
    print("\n5. Testing LLM-based risk analysis...")
    print("Note: This test will be skipped if Ollama is not running")
    print("Set up the ollama_client.py file and run Ollama to enable this test")
    
    try:
        # Check if we should skip LLM test by trying a simple connection
        try:
            advisor = FinancialAdvisorLLM()
            test_response = advisor.test_connection()
            llm_available = True
        except Exception:
            llm_available = False
            print("Skipping LLM test as Ollama connection failed")
            
        if llm_available:
            risk_analysis = risk_agent.get_risk_analysis(risk_metrics, 5)  # Moderate risk tolerance
            
            print("\nLLM Risk Analysis Results:")
            for key, value in risk_analysis.items():
                if isinstance(value, list):
                    print(f"{key}:")
                    for item in value:
                        print(f"  - {item}")
                else:
                    print(f"{key}: {value}")
            print("\nLLM risk analysis successful!")
    except Exception as e:
        print(f"\nError in LLM risk analysis: {e}")
    
    print("\nRisk Assessment Agent test completed!")

if __name__ == "__main__":
    test_risk_assessment_agent()