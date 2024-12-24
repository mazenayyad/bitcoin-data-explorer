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
df['Change_Percentage'] = df['Change_Percentage'].str.replace('%','').astype(float)
for col in ['Close', 'Open', 'High', 'Low']:
    df[col] = df[col].str.replace(',','').astype(float)
df['Volume'] = df['Volume'].apply(clean_volume)

df = df.sort_values(by='Date', ascending=False)
print(df.head(5))