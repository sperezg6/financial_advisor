import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from portfolio_advisor.models.base_model import FinancialAdvisorLLM

class PortfolioConstructorAgent:
    """Agent responsible for creating the final portfolio recommendations"""
    
    def __init__(self, llm=None, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize LLM
        self.llm = llm if llm else FinancialAdvisorLLM()
    
    def create_allocation_template(self, assets, risk_level, time_horizon):
        """
        Create a template for asset allocation based on risk level
        
        Args:
            assets (list): List of asset tickers
            risk_level (int): Risk tolerance (1-10)
            time_horizon (int): Investment time horizon in years
            
        Returns:
            dict: Template allocations for different asset classes
        """
        # Simple templates based on risk level (1-10)
        templates = {
            # Conservative: More fixed income, less equities
            1: {"stocks": 0.20, "bonds": 0.70, "cash": 0.10},
            2: {"stocks": 0.30, "bonds": 0.60, "cash": 0.10},
            3: {"stocks": 0.40, "bonds": 0.55, "cash": 0.05},
            4: {"stocks": 0.45, "bonds": 0.50, "cash": 0.05},
            # Moderate: Balanced approach
            5: {"stocks": 0.50, "bonds": 0.45, "cash": 0.05},
            6: {"stocks": 0.60, "bonds": 0.35, "cash": 0.05},
            7: {"stocks": 0.70, "bonds": 0.25, "cash": 0.05},
            # Aggressive: More equities, less fixed income
            8: {"stocks": 0.80, "bonds": 0.15, "cash": 0.05},
            9: {"stocks": 0.90, "bonds": 0.10, "cash": 0.00},
            10: {"stocks": 0.95, "bonds": 0.05, "cash": 0.00}
        }
        
        # Get template based on risk level
        allocation_template = templates.get(risk_level, templates[5])  # Default to moderate
        
        # Adjust for time horizon
        if time_horizon > 10:
            # Longer time horizon can support more stocks
            allocation_template["stocks"] = min(allocation_template["stocks"] + 0.05, 0.95)
            allocation_template["bonds"] = max(allocation_template["bonds"] - 0.05, 0.05)
        elif time_horizon < 3:
            # Shorter time horizon needs more conservative approach
            allocation_template["stocks"] = max(allocation_template["stocks"] - 0.10, 0.20)
            allocation_template["bonds"] = min(allocation_template["bonds"] + 0.05, 0.70)
            allocation_template["cash"] = min(allocation_template["cash"] + 0.05, 0.10)
            
        return allocation_template
    
    def generate_allocation(self, assets, asset_info, risk_metrics, risk_level, time_horizon):
        """
        Generate specific allocation percentages for the given assets
        
        Args:
            assets (list): List of asset tickers
            asset_info (dict): Information about each asset
            risk_metrics (pd.DataFrame): Risk metrics for assets
            risk_level (int): Risk tolerance (1-10)
            time_horizon (int): Investment time horizon in years
            
        Returns:
            dict: Allocation percentages for each asset
        """
        # Get allocation template
        template = self.create_allocation_template(assets, risk_level, time_horizon)
        
        # Group assets by type
        stocks = []
        bonds = []
        others = []
        
        for ticker in assets:
            asset_type = asset_info.get(ticker, {}).get("type", "").lower()
            if asset_type == "stock":
                stocks.append(ticker)
            elif asset_type == "bond" or "bond" in asset_type:
                bonds.append(ticker)
            else:
                # Check if it's an ETF and determine its category
                if asset_type == "etf":
                    category = asset_info.get(ticker, {}).get("category", "").lower()
                    if "bond" in category or "fixed income" in category:
                        bonds.append(ticker)
                    elif "equity" in category or "stock" in category:
                        stocks.append(ticker)
                    else:
                        others.append(ticker)
                else:
                    others.append(ticker)
        
        # Allocate based on template
        allocations = {}
        
        # Allocate to stocks
        stock_weight = template["stocks"]
        if stocks:
            weight_per_stock = stock_weight / len(stocks)
            for stock in stocks:
                allocations[stock] = weight_per_stock
        
        # Allocate to bonds
        bond_weight = template["bonds"]
        if bonds:
            weight_per_bond = bond_weight / len(bonds)
            for bond in bonds:
                allocations[bond] = weight_per_bond
        
        # Allocate to others (cash equivalents, commodities, etc.)
        other_weight = template["cash"]
        if others:
            weight_per_other = other_weight / len(others)
            for other in others:
                allocations[other] = weight_per_other
        
        # Handle case where we don't have certain asset classes
        if not stocks:
            # Redistribute stock allocation to bonds and others
            if bonds:
                bond_adj = bond_weight + (stock_weight * bond_weight / (bond_weight + other_weight))
                for bond in bonds:
                    allocations[bond] = bond_adj / len(bonds)
            if others:
                other_adj = other_weight + (stock_weight * other_weight / (bond_weight + other_weight))
                for other in others:
                    allocations[other] = other_adj / len(others)
        
        if not bonds:
            # Redistribute bond allocation to stocks and others
            if stocks:
                stock_adj = stock_weight + (bond_weight * stock_weight / (stock_weight + other_weight))
                for stock in stocks:
                    allocations[stock] = stock_adj / len(stocks)
            if others:
                other_adj = other_weight + (bond_weight * other_weight / (stock_weight + other_weight))
                for other in others:
                    allocations[other] = other_adj / len(others)
        
        # Ensure allocations sum to 1
        total = sum(allocations.values())
        if total > 0:
            for ticker in allocations:
                allocations[ticker] = allocations[ticker] / total
        
        return allocations