import pandas as pd
import os

# 1. Load Data
file_path = os.path.join('data', 'norm-data.xlsx')
df = pd.read_excel(file_path, sheet_name='Train_Normalized')

# 2. Recalibrate logic using Percentile Ranks (Market Relative)
print("Recalibrating M-Scores for all companies...")
ranked_df = pd.DataFrame(index=df.index)

# Higher is better
for col in ['Interest_Coverage', 'Current_Ratio', 'ROE', 'ROCE', 
            'Revenue_CAGR_3Y', 'YoY_Revenue_Growth', 'Asset_Turnover', 'Inventory_Turnover']:
    ranked_df[col + '_Rank'] = df[col].rank(pct=True) * 100

# Lower is better (Invert Rank)
ranked_df['DE_Rank'] = (1 - df['Debt_to_Equity'].rank(pct=True)) * 100
ranked_df['PE_Rank'] = (1 - df['P_E_Ratio'].rank(pct=True)) * 100
ranked_df['PB_Rank'] = (1 - df['P_B_Ratio'].rank(pct=True)) * 100

def calculate_m_score(row):
    q = (row['Interest_Coverage_Rank'] + row['Current_Ratio_Rank'] + row['DE_Rank']) / 3
    g = (row['Revenue_CAGR_3Y_Rank'] + row['YoY_Revenue_Growth_Rank'] + row['ROE_Rank'] + row['ROCE_Rank']) / 4
    v = (row['PE_Rank'] + row['PB_Rank']) / 2
    e = (row['Asset_Turnover_Rank'] + row['Inventory_Turnover_Rank']) / 2
    return round((q * 0.3) + (g * 0.3) + (v * 0.2) + (e * 0.2), 2)

# 3. Save the NEW Scores
df['M_Score'] = ranked_df.apply(calculate_m_score, axis=1)

def get_verdict(score):
    if score >= 70: return "✅ LOW RISK (Strong Performer)"
    elif score >= 45: return "⚖️ MEDIUM RISK (Neutral/Hold)"
    else: return "🚨 HIGH RISK (Financial Weakness)"

df['Risk_Verdict'] = df['M_Score'].apply(get_verdict)

# OVERWRITE the Excel file with corrected scores
output_path = os.path.join('data', 'final_stock_rankings.xlsx')
df.to_excel(output_path, index=False)
print(f"✅ Success! Updated scores for {len(df)} companies saved to {output_path}")

# 4. Search Check
query = input("\nEnter a company name to verify (e.g. Anant Raj): ")
res = df[df['Company_Name'].str.contains(query, case=False, na=False)]
if not res.empty:
    print(f"M-Score: {res.iloc[0]['M_Score']} | Verdict: {res.iloc[0]['Risk_Verdict']}")