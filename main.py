#                             CONSUMER BEHAVIOR ANALYSIS: INCOME VS. PRODUCT SPENDING(WINE)



# We first of all import the numpy 
import numpy as np 

# 1. THE DATA IMPORT
# We grab: Income (4), Recency (8), and Wine Spending (9)
# Use 'delimiter' because this Kaggle file uses Tabs (\t), not Commas
data = np.genfromtxt('marketing_campaign.csv',
                    delimiter='\t',
                    skip_header=1,
                    usecols = (4,8,9))
#print(data)

# 2. THE CLEANING OF THE DATA AND LODING DATA IN THE COLUMNS
#We are using a dataset that has Clean data already so we will not be going to clear the data this time
income_raw = data[:,0]
#print(income_raw)
recency = data[:,1]
wine_spent = data[:,2]

# 3. FINDING OF THE AVERAGE INCOME
avg_income_raw = np.nanmean(income_raw).round()
#print(f'The average income is: {avg_income_raw}')

# 4. REPLACING THE MISSING VALUES WITH THE AVERAGE
# We are using this method here btw np.where(CONDITION, VALUE_IF_TRUE, VALUE_IF_FALSE)
income = np.where(np.isnan(income_raw), avg_income_raw, income_raw)

# 5. ANALYSING THE DATA NOW
# Find the 90th percentile: Who are the top 10% of wine spenders?
wine_threhsold = np.nanpercentile(wine_spent, 90)
#print(wine_threhsold)
big_spenders = data[wine_spent > wine_threhsold]

# 6.FINDING THE PERCENT OF INCOME GOING TO WINE
spend_ratio = (wine_spent/ income)*100
avg_ratio = np.nanmean(spend_ratio)
#print(avg_ratio)

# 7. ADVANCE SEGMENTATION LOGIC
# Let's define "Recent" as within the last 30 days
# and "High Spend" as above the 75th percentile
recent_limit = 30
spend_limit = np.nanpercentile(wine_spent,75)
# A: THE WHALES (Spent > Limit AND Recency <= 30)
whales = data[(wine_spent >= spend_limit) & (recency <= recent_limit)]
# B: THE LAPSED VIPs (Spent > Limit AND Recency > 60)
lapsed_vips = data[(wine_spent >= spend_limit) & (recency > 60)]

# 8. CORRELATION CHECK
correlation = np.corrcoef(income, wine_spent)[0,1] 
#print(correlation)

# FINALLY WE PRINT EVERYTHING
print("-" * 45)
print("       MARKETING INTELLIGENCE REPORT")
print("-" * 45)
print(f"Average Customer Income:     ${avg_income_raw:,.2f}")
print(f"Average Income-Spend Ratio:  {avg_ratio:.2f}%")
print(f"Top 10% Spend Threshold:     ${wine_threhsold:.2f}")
print("-" * 45)
print(f"ACTIVE WHALES FOUND:         {len(whales)}")
print(f"LAPSED VIPs (AT RISK):       {len(lapsed_vips)}")
print("-" * 45)
print(f"CORRELATION SCORE:           {correlation:.2f}")

# Interpreting the Correlation for the user
if correlation > 0.7:
    print("INSIGHT: High Income strongly predicts higher spending.")
elif correlation > 0.3:
    print("INSIGHT: There is a moderate link between Income and spending.")
else:
    print("INSIGHT: Income is NOT the main driver for these sales.")
print("-" * 45)