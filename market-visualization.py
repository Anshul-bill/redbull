import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
file_path = os.path.join('data', 'norm-data.xlsx')
df = pd.read_excel(file_path, sheet_name='Train_Normalized')

# 2. Recalculate Ranks & M-Score
print("Calculating market distribution and scores...")
ranked_df = pd.DataFrame(index=df.index)

# Define Pillars (Percentiles)
cols_up = ['Interest_Coverage', 'Current_Ratio', 'ROE', 'ROCE', 
           'Revenue_CAGR_3Y', 'YoY_Revenue_Growth', 'Asset_Turnover', 'Inventory_Turnover']
for col in cols_up:
    ranked_df[col + '_Rank'] = df[col].rank(pct=True) * 100

ranked_df['DE_Rank'] = (1 - df['Debt_to_Equity'].rank(pct=True)) * 100
ranked_df['PE_Rank'] = (1 - df['P_E_Ratio'].rank(pct=True)) * 100
ranked_df['PB_Rank'] = (1 - df['P_B_Ratio'].rank(pct=True)) * 100

# Sub-Scores for Plotting
df['Growth_Score'] = ranked_df[['Revenue_CAGR_3Y_Rank', 'YoY_Revenue_Growth_Rank', 'ROE_Rank', 'ROCE_Rank']].mean(axis=1)
df['Value_Score'] = ranked_df[['PE_Rank', 'PB_Rank']].mean(axis=1)

q_score = ranked_df[['Interest_Coverage_Rank', 'Current_Ratio_Rank', 'DE_Rank']].mean(axis=1)
e_score = ranked_df[['Asset_Turnover_Rank', 'Inventory_Turnover_Rank']].mean(axis=1)

# Final M-Score
df['M_Score'] = (q_score * 0.3) + (df['Growth_Score'] * 0.3) + (df['Value_Score'] * 0.2) + (e_score * 0.2)

# Assign Categories
def get_verdict(score):
    if score >= 70: return "Low Risk (Elite)"
    elif score >= 45: return "Medium Risk (Neutral)"
    else: return "High Risk (Avoid)"

df['Risk_Status'] = df['M_Score'].apply(get_verdict)

# 3. Create the Dashboard (2 Plots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
palette = {"Low Risk (Elite)": "#2ecc71", "Medium Risk (Neutral)": "#f39c12", "High Risk (Avoid)": "#e74c3c"}

# --- PLOT 1: SCATTER PLOT ---
sns.scatterplot(data=df, x='Value_Score', y='Growth_Score', hue='Risk_Status', 
                palette=palette, alpha=0.6, s=100, ax=ax1, edgecolor='w')

# Annotate some Top stocks
top_5 = df.sort_values(by='M_Score', ascending=False).head(5)
for i, row in top_5.iterrows():
    ax1.text(row['Value_Score']+1, row['Growth_Score']+1, row['Company_Name'], fontsize=9, weight='bold')

ax1.set_title('Market Map: Growth vs Valuation', fontsize=16)
ax1.set_xlabel('Value Score (Higher = "Cheaper")')
ax1.set_ylabel('Growth Score (Higher = Better Performance)')
ax1.grid(True, linestyle='--', alpha=0.5)

# --- PLOT 2: PERCENTAGE PIE CHART ---
counts = df['Risk_Status'].value_counts()
# Ensure the colors match the legend
pie_colors = [palette[category] for category in counts.index]

ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, 
        colors=pie_colors, explode=(0.05, 0, 0), shadow=True)
ax2.set_title('Market Risk Distribution (%)', fontsize=16)

# 4. Save and Show
plt.tight_layout()
plt.savefig('market_dashboard.png', dpi=300)
print(f"✅ Success! Dashboard saved as 'market_dashboard.png'")
print("\nMarket Summary:")
print(counts)
# plt.show()