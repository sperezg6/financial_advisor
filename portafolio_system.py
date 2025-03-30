import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from portfolio_advisor.models.base_model import FinancialAdvisorLLM
from portfolio_advisor.agents.data_collection import DataCollectorAgent
from portfolio_advisor.agents.risk_assesment import RiskAssessmentAgent
from portfolio_advisor.agents.portofolio_constructor import PortfolioConstructorAgent

class PortfolioAdvisorSystem:
    """Main system that coordinates all the specialized agents"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize shared LLM
        self.llm = FinancialAdvisorLLM()
        
        # Initialize agents
        self.data_agent = DataCollectorAgent(llm=self.llm, data_dir=self.data_dir)
        self.risk_agent = RiskAssessmentAgent(llm=self.llm, data_dir=self.data_dir)
        self.portfolio_agent = PortfolioConstructorAgent(llm=self.llm, data_dir=self.data_dir)
        
        print("Portfolio Advisor System initialized!")
    
    def create_portfolio_recommendation(self, risk_tolerance, time_horizon, focus_sectors=None, investment_amount=None):
        """
        Create a complete portfolio recommendation
        
        Args:
            risk_tolerance (int): User's risk tolerance (1-10)
            time_horizon (int): Investment time horizon in years
            focus_sectors (list): Optional list of sectors to focus on
            investment_amount (float): Optional investment amount
            
        Returns:
            dict: Complete portfolio recommendation
        """
        print("\n--- Starting Portfolio Recommendation Process ---")
        
        # Step 1: Get recommended assets from the Data Agent
        print("\nStep 1: Finding suitable assets based on your profile...")
        recommended_assets = self.data_agent.get_recommended_assets(
            risk_tolerance, 
            time_horizon, 
            focus_sectors
        )
        
        if not recommended_assets:
            return {"error": "Could not generate asset recommendations"}
        
        # Extract ticker list and create asset info dict
        asset_tickers = [asset["ticker"] for asset in recommended_assets]
        asset_info = {
            asset["ticker"]: {
                "name": asset["name"], 
                "type": asset["type"].lower(),
                "justification": asset["justification"]
            } 
            for asset in recommended_assets
        }
        
        print(f"Found {len(asset_tickers)} suitable assets:")
        for asset in recommended_assets:
            print(f"  {asset['ticker']} ({asset['name']}) - {asset['type']}")
        
        # Step 2: Calculate risk metrics using the Risk Agent
        print("\nStep 2: Analyzing risk characteristics...")
        risk_metrics = self.risk_agent.calculate_risk_metrics(asset_tickers)
        
        print("Risk analysis completed. Key metrics (volatility, max drawdown, returns):")
        for ticker in asset_tickers[:3]:  # Show first 3 for brevity
            risk_row = risk_metrics.loc[ticker]
            print(f"  {ticker}: Vol={risk_row['volatility']:.2f}, MaxDD={risk_row['max_drawdown']:.2f}, Return={risk_row['avg_return']:.2f}")
        if len(asset_tickers) > 3:
            print(f"  ... and {len(asset_tickers) - 3} more assets")
        
        # Step 3: Calculate correlations
        print("\nStep 3: Analyzing diversification potential...")
        correlation_matrix = self.risk_agent.calculate_correlation_matrix(asset_tickers)
        
        # Calculate average correlation for each asset
        avg_correlations = correlation_matrix.mean().sort_values()
        
        print("Diversification analysis completed. Assets with lowest correlations:")
        for ticker, corr in avg_correlations.iloc[:3].items():
            print(f"  {ticker}: Avg correlation = {corr:.2f}")
        
        # Step 4: Get risk analysis from the Risk Agent
        print("\nStep 4: Evaluating asset suitability based on risk tolerance...")
        risk_analysis = self.risk_agent.get_risk_analysis(risk_metrics, risk_tolerance)
        
        print("Risk suitability analysis completed:")
        print(f"  Appropriate assets: {', '.join(risk_analysis.get('appropriate_assets', []))}")
        print(f"  Too risky assets: {', '.join(risk_analysis.get('too_risky_assets', []))}")
        print(f"  Too conservative assets: {', '.join(risk_analysis.get('too_conservative_assets', []))}")
        
        # Step 5: Generate portfolio allocations
        print("\nStep 5: Generating optimal portfolio allocations...")
        allocations = self.portfolio_agent.generate_allocation(
            asset_tickers,
            asset_info,
            risk_metrics,
            risk_level=risk_tolerance,
            time_horizon=time_horizon
        )
        
        # Step 6: Assess portfolio risk
        print("\nStep 6: Assessing overall portfolio risk...")
        portfolio_metrics = self.risk_agent.assess_portfolio_risk(
            risk_metrics,
            correlation_matrix,
            allocations
        )
        
        risk_match = self.risk_agent.assess_risk_tolerance_match(
            portfolio_metrics,
            risk_tolerance
        )
        
        print(f"Portfolio risk assessment: {risk_match['risk_match']}")
        print(f"Volatility: {portfolio_metrics['volatility']:.2f}")
        print(f"Expected return: {portfolio_metrics['expected_return']:.2f}")
        print(f"Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
        print(f"Diversification score: {portfolio_metrics['diversification_score']:.2f}")
        
        # Step 7: Create final recommendation object
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate dollar amounts if investment amount provided
        dollar_allocations = None
        if investment_amount:
            dollar_allocations = {
                ticker: round(pct * investment_amount, 2) 
                for ticker, pct in allocations.items()
            }
        
        # Compile final recommendation
        recommendation = {
            "timestamp": timestamp,
            "user_profile": {
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "focus_sectors": focus_sectors,
                "investment_amount": investment_amount
            },
            "recommended_assets": recommended_assets,
            "portfolio_allocations": {
                ticker: round(pct * 100, 2) for ticker, pct in allocations.items()
            },
            "dollar_allocations": dollar_allocations,
            "portfolio_metrics": {
                "volatility": round(portfolio_metrics["volatility"], 4),
                "expected_return": round(portfolio_metrics["expected_return"], 4),
                "sharpe_ratio": round(portfolio_metrics["sharpe_ratio"], 4),
                "diversification_score": round(portfolio_metrics["diversification_score"], 4)
            },
            "risk_assessment": {
                "risk_match": risk_match["risk_match"],
                "recommendation": risk_match["recommendation"]
            },
            "strategy": risk_analysis.get("portfolio_strategy", ""),
            "reasoning": risk_analysis.get("reasoning", "")
        }
        
        # Save recommendation to file
        filename = os.path.join(self.data_dir, f"portfolio_recommendation_{timestamp}.json")
        with open(filename, 'w') as f:
            json.dump(recommendation, f, indent=4)
        
        # Generate visualization
        try:
            self._generate_allocation_chart(
                allocations, 
                asset_info, 
                risk_tolerance, 
                time_horizon,
                filename=os.path.join(self.data_dir, f"portfolio_allocation_{timestamp}.png")
            )
        except Exception as e:
            print(f"Error generating chart: {e}")
        
        print(f"\nPortfolio recommendation completed and saved to {filename}")
        return recommendation
    
    def _generate_allocation_chart(self, allocations, asset_info, risk_tolerance, time_horizon, filename=None):
        """Generate a pie chart of portfolio allocations"""
        # Group allocations by asset type
        asset_types = {}
        for ticker, allocation in allocations.items():
            asset_type = asset_info[ticker]["type"]
            if asset_type in asset_types:
                asset_types[asset_type] += allocation
            else:
                asset_types[asset_type] = allocation
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Overall asset type allocation
        ax1.pie(
            asset_types.values(),
            labels=[f"{k.capitalize()}: {v*100:.1f}%" for k, v in asset_types.items()],
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title(f'Asset Type Allocation\nRisk Level: {risk_tolerance}/10, Time Horizon: {time_horizon} years')
        
        # Plot 2: Individual asset allocation
        sorted_allocations = dict(sorted(allocations.items(), key=lambda x: x[1], reverse=True))
        ax2.pie(
            sorted_allocations.values(),
            labels=[f"{k}: {v*100:.1f}%" for k, v in sorted_allocations.items()],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Individual Asset Allocation')
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename)
            print(f"Portfolio allocation chart saved to {filename}")
        
        plt.close()

def interactive_portfolio_advisor():
    """Run an interactive portfolio advisor session"""
    print("=" * 60)
    print("Welcome to the AI Portfolio Advisor System!")
    print("=" * 60)
    print("\nThis system will help you create a personalized investment portfolio.")
    
    # Initialize the system
    print("\nInitializing system...")
    advisor = PortfolioAdvisorSystem()
    
    while True:
        print("\n" + "=" * 60)
        print("NEW PORTFOLIO RECOMMENDATION SESSION")
        print("=" * 60)
        
        # Get user risk tolerance
        while True:
            try:
                risk_input = input("\nOn a scale of 1-10, what is your risk tolerance? (1=very conservative, 10=very aggressive): ")
                risk_tolerance = int(risk_input)
                if 1 <= risk_tolerance <= 10:
                    break
                else:
                    print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get investment time horizon
        while True:
            try:
                horizon_input = input("\nWhat is your investment time horizon in years? (e.g., 5, 10, 20): ")
                time_horizon = int(horizon_input)
                if time_horizon > 0:
                    break
                else:
                    print("Please enter a positive number of years.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get focus sectors (optional)
        sectors_input = input("\nAre there any specific sectors you want to focus on? (comma-separated, or press Enter for none): ")
        focus_sectors = [s.strip() for s in sectors_input.split(",")] if sectors_input.strip() else None
        
        # Get investment amount (optional)
        amount_input = input("\nWhat is your investment amount? (optional, press Enter to skip): $")
        investment_amount = float(amount_input) if amount_input.strip() else None
        
        # Generate portfolio recommendation
        print("\nGenerating your personalized portfolio recommendation...")
        recommendation = advisor.create_portfolio_recommendation(
            risk_tolerance,
            time_horizon,
            focus_sectors,
            investment_amount
        )
        
        if "error" in recommendation:
            print(f"\nError: {recommendation['error']}")
        else:
            # Show results
            print("\n" + "=" * 60)
            print("YOUR PERSONALIZED PORTFOLIO RECOMMENDATION")
            print("=" * 60)
            
            # Portfolio allocation
            print("\nRECOMMENDED ASSET ALLOCATION:")
            sorted_allocations = sorted(
                recommendation["portfolio_allocations"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for ticker, percentage in sorted_allocations:
                # Find the asset info
                asset_info = next((a for a in recommendation["recommended_assets"] if a["ticker"] == ticker), None)
                if asset_info:
                    print(f"  {ticker} ({asset_info['name']}) - {percentage}%")
                else:
                    print(f"  {ticker} - {percentage}%")
            
            # Dollar allocations if provided
            if recommendation["dollar_allocations"]:
                print("\nDOLLAR ALLOCATION:")
                sorted_dollars = sorted(
                    recommendation["dollar_allocations"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for ticker, amount in sorted_dollars:
                    print(f"  {ticker}: ${amount:,.2f}")
            
            # Portfolio metrics
            print("\nPORTFOLIO METRICS:")
            metrics = recommendation["portfolio_metrics"]
            print(f"  Expected Annual Volatility: {metrics['volatility']*100:.2f}%")
            print(f"  Expected Annual Return: {metrics['expected_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Diversification Score: {metrics['diversification_score']:.2f}")
            
            # Risk assessment
            print("\nRISK ASSESSMENT:")
            print(f"  Match with your risk profile: {recommendation['risk_assessment']['risk_match'].replace('_', ' ').title()}")
            print(f"  Recommendation: {recommendation['risk_assessment']['recommendation']}")
            
            # Strategy
            if recommendation["strategy"]:
                print("\nPORTFOLIO STRATEGY:")
                print(f"  {recommendation['strategy']}")
            
            # Chart location
            print(f"\nA visual representation of your portfolio has been saved to:")
            print(f"  {os.path.join(os.path.abspath(advisor.data_dir), 'portfolio_allocation_' + recommendation['timestamp'] + '.png')}")
        
        # Ask if user wants another recommendation
        another = input("\nWould you like to create another portfolio recommendation? (y/n): ")
        if another.lower() not in ['y', 'yes']:
            break
    
    print("\nThank you for using the AI Portfolio Advisor System!")

if __name__ == "__main__":
    interactive_portfolio_advisor()