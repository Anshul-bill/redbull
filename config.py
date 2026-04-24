"""Central configuration: feature list, paths, and shared constants.

Importing this module creates the output directories so every script
can write to ``outputs/`` and ``models/`` without guarding for existence.
"""
import os

FEATURES = [
    "P_E_Ratio", "P_S_Ratio", "EV_EBITDA", "P_B_Ratio", "ROE", "ROCE",
    "Profit_Margin", "EBITDA_Margin", "Debt_to_Equity", "Interest_Coverage",
    "Current_Ratio", "Quick_Ratio", "Net_Debt_to_EBITDA", "Cash_Ratio",
    "Revenue_CAGR_3Y", "Stock_Return_3Y", "YoY_Revenue_Growth", "Asset_Turnover",
    "Inventory_Turnover", "PEG_Ratio",
]

TARGET = "Total_Label_Score"
SECTOR_COL = "Sector"

# 3-class risk bucket derived from Total_Label_Score (0-10) for the DNN's
# auxiliary head. Edges are inclusive on both sides.
RISK_BUCKETS = {"High": (0, 3), "Medium": (4, 6), "Low": (7, 10)}


def score_to_bucket(score: int) -> int:
    """Map a 0-10 Total_Label_Score to 0 (High risk) / 1 (Medium) / 2 (Low)."""
    if score <= 3:
        return 0
    if score <= 6:
        return 1
    return 2

DATA_PATH = os.path.join("data", "norm-data.xlsx")
SHEET = "Train_Normalized"

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

RANDOM_STATE = 42
CV_FOLDS = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
