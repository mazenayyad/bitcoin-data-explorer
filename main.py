import pandas as pd

def clean_volume(value):
    if 'K' in value:
        return float(value.replace('K','').replace(',','')) * 1000
    if 'M' in value:
        return float(value.replace('M','').replace(',','')) * 1000000
    if 'B' in value:
        return float(value.replace('B','').replace(',','')) * 1000000000
    else:
        return float(value.replace(',',''))
    
df = pd.read_csv('data/bitcoin_data.csv')

# Renaming columns to be more descriptive
df = df.rename(columns={'Price' : 'Close', 'Vol.' : 'Volume', 'Change %' : 'Change_Percentage'})

# Convering columns to correct data types
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.date # Removes the time component from datetime (ex: 00:00)
df['Change_Percentage'] = df['Change_Percentage'].str.replace('%','').astype(float)
for col in ['Close', 'Open', 'High', 'Low']:
    df[col] = df[col].str.replace(',','').astype(float)
df['Volume'] = df['Volume'].apply(clean_volume)

df = df.sort_values(by='Date', ascending=True)

# Dataset Summary
maxPriceIndex = df['High'].idxmax()
maxPrice = df.loc[maxPriceIndex,'High']
maxPriceDate = df.loc[maxPriceIndex,'Date']

minPriceIndex = df['Low'].idxmin()
minPrice = df.loc[minPriceIndex, 'Low']
minPriceDate = df.loc[minPriceIndex, 'Date']

# Note: The 'Volume' column represents trading activity on a specific exchange 
# and may not accurately reflect the total trading volume of Bitcoin across 
# all markets, especially given the decline in activity on this particular exchange.

# maxVolumeIndex = df['Volume'].idxmax()
# maxVolume = df.loc[maxVolumeIndex, 'Volume']
# maxVolumeDate = df.loc[maxVolumeIndex, 'Date']

# minVolumeIndex = df['Volume'].idxmin()
# minVolume = df.loc[minVolumeIndex,'Volume']
# minVolumeDate = df.loc[minVolumeIndex,'Date']

maxChangeIndex = df['Change_Percentage'].idxmax()
maxChange = df.loc[maxChangeIndex, 'Change_Percentage']
maxChangeDate = df.loc[maxChangeIndex, 'Date']

minChangeIndex = df['Change_Percentage'].idxmin()
minChange = df.loc[minChangeIndex, 'Change_Percentage']
minChangeDate = df.loc[minChangeIndex, 'Date']

print(maxPrice,maxPriceDate,sep=' = ')
print(minPrice,minPriceDate,sep=' = ')
print(maxChange,maxChangeDate,sep=' = ')
print(minChange,minChangeDate,sep=' = ')
