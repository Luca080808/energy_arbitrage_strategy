# Energy Arbitrage Strategy Using Machine Learning

## Key Results

**Best model**: LightGBM

**Backtest period**: July 2024 - December 2024

**Cumulative PnL**: 20,463,070	RON

## Overview

This project develops a **machine learning-based trading strategy** designed to exploit arbitrage opportunities between the **intraday electricity market** and the **imbalance settlement mechanism**.

Using historical power data at **15-minute time intervals**, the goal is to:

* Predict imbalance price spreads
* Determine optimal trading actions (**long, short, or no trade**)
* Evaluate the profitability of the resulting strategy


The repository contains both:

* A **Jupyter Notebook** with the full analysis and modeling workflow
* A **Streamlit interactive application** used to present the results and methodology in a structured format.

---

# Quick Start (Recommended)

The easiest way to review the project is through the **Streamlit interactive dashboard**.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Launch the dashboard
```bash
streamlit run app.py
```
### 3. Open the interface

Once launched, copy the local URL and open it via Google Chrome for the best UI compatibility.

# Running the Notebook

If you prefer reviewing the full workflow directly:

jupyter notebook

Open:

Jupyter_Notebook/Project.ipynb

The notebook contains:

* data preprocessing
* feature engineering
* model training
* cross-validation
* trading strategy implementation
* backtesting results


# Modeling Approach

Three regression models were trained to predict imbalance spreads:

* Random Forest
* HistGradientBoosting
* LightGBM

Each model predicts two spreads:

**Long spread**:

imbalance_price_positive − intraday_price

**Short spread**: 

intraday_price − imbalance_price_negative

Two evaluation metrics are used:

**Mean Absolute Error (MAE)**

MAE measures how close predicted spreads are to realized spreads.

**Mean Directional Accuracy (MDA)**

MDA measures how often the predicted spread has the correct sign.

---

# Trading Strategy

The strategy uses the predicted spreads to generate trading decisions.

For each 15-minute interval:

1. Predict the long and short spreads
2. Apply optimized buy and sell thresholds
3. Decide whether to:

* go long
* go short
* not trade at all

To provide a performance benchmark, a **Perfect PnL** series is computed.

This represents the theoretical maximum profit achievable with perfect foresight.

---

# Results

All three models produce **positive cumulative PnL**, with the gradient boosting models outperforming Random Forest.

**Best performing model:**

LightGBM

Cumulative PnL:

69,870,924 RON

Additional performance metrics reported in the dashboard include:

* cumulative PnL curves
* maximum drawdown
* statistical significance (PnL t-statistic)

---
# Project Structure
```
energy_arbitrage_ml/
├── received_documents/
│   ├── dataset_engie_studycase.pkl
│   └── Case_Study_Instructions.pdf
├── Jupyter_Notebook/
│   └── Project.ipynb
├── results/ 
│   ├── tables/ - Model evaluation csv tables
│   ├── numpy_objects/ - PnL vectors and some other numpy objects
├── app/
│   └── app.py  - The Streamlit app
├── requirements.txt
└── README.md
```
