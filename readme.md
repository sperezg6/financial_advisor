# AI Portfolio Advisor System

The **AI Portfolio Advisor System** is a Python-based application designed to help users create personalized investment portfolios. It leverages advanced financial models and AI agents to recommend assets, analyze risk, and generate optimal portfolio allocations tailored to the user's risk tolerance, investment time horizon, and sector preferences.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Dependencies](#dependencies)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features

- **Personalized Recommendations**: Suggests assets based on user-defined risk tolerance, time horizon, and sector preferences.
- **Risk Analysis**: Calculates key risk metrics such as volatility, maximum drawdown, and average returns.
- **Diversification Analysis**: Evaluates asset correlations to ensure portfolio diversification.
- **Optimal Allocations**: Generates portfolio allocations that align with the user's risk profile.
- **Visualization**: Creates pie charts to visualize asset type and individual asset allocations.
- **Interactive Mode**: Allows users to interactively input their preferences and receive recommendations.

---

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/your-username/ai-portfolio-advisor.git
   cd ai-portfolio-advisor
```

2. Create a virtual environment and activate it:
```bash 
    python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash 
pip install -r requirements.txt
````

## Usage
Interactive Mode
Run the application in interactive mode to generate portfolio recommendations:
```bash
 python [portafolio_system.py](http://_vscodecontentref_/1)
```

Follow the prompts to input your:

Risk tolerance (1-10)
Investment time horizon (in years)
Focus sectors (optional)
Investment amount (optional)
The system will generate a personalized portfolio recommendation and save the results in the data/ directory.

Example Output
Portfolio Recommendation: Saved as a JSON file (e.g., portfolio_recommendation_YYYYMMDD_HHMMSS.json).
Visualization: Saved as a PNG file (e.g., portfolio_allocation_YYYYMMDD_HHMMSS.png).



## How It Works
1. Data Collection: The DataCollectorAgent gathers market data and identifies suitable assets based on the user's profile.
2. Risk Assessment: The RiskAssessmentAgent calculates risk metrics and evaluates asset correlations.
3. Portfolio Construction: The PortfolioConstructorAgent generates optimal allocations and assesses portfolio risk.
4. Visualization: A pie chart is created to visualize the portfolio allocations.
5. Recommendation: The final recommendation is compiled and saved as a JSON file.