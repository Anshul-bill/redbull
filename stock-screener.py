import pandas as pd
import os

# 1. Load the ranked data created by your scoring engine
# Note: Ensure you have run your scoring-engine.py first to create this file
file_path = os.path.join('data', 'final_stock_rankings.xlsx')

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please run your scoring-engine.py script first.")
    exit()

df = pd.read_excel(file_path)

def run_screener():
    print("\n" + "="*50)
    print("       🚀 AI STOCK SCREENER: FILTER BY SCORE")
    print("="*50)
    
    try:
        min_score = float(input("Enter minimum M-Score (e.g., 70): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    # 2. Filter companies
    results = df[df['M_Score'] >= min_score].copy()
    
    # 3. Sort results by score (highest first)
    results = results.sort_values(by='M_Score', ascending=False)

    # 4. Display Results
    if not results.empty:
        print(f"\n✅ Found {len(results)} companies with a score of {min_score} or higher:")
        print("-" * 75)
        print(f"{'Company Name':<45} | {'Sector':<20} | {'Score':<10}")
        print("-" * 75)
        
        for _, row in results.iterrows():
            print(f"{row['Company_Name'][:43]:<45} | {row['Sector'][:18]:<20} | {row['M_Score']:<10.2f}")
            
        print("-" * 75)
        print(f"End of list. Total: {len(results)} companies.")
    else:
        print(f"\n❌ No companies found with a score of {min_score} or higher.")

if __name__ == "__main__":
    while True:
        run_screener()
        cont = input("\nRun another search? (y/n): ").lower()
        if cont != 'y':
            break