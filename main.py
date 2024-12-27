import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def streak_sign(value):
    if value > 0:
        return 1 # Winning day
    elif value < 0:
        return -1 # Losing day
    else:
        return 0 # Neutral day

def clean_volume(value):
    if 'K' in value:
        return float(value.replace('K','').replace(',','')) * 1000
    if 'M' in value:
        return float(value.replace('M','').replace(',','')) * 1000000
    if 'B' in value:
        return float(value.replace('B','').replace(',','')) * 1000000000
    else:
        return float(value.replace(',',''))

def main():
    df = pd.read_csv('data/bitcoin_data.csv')

    # -----Renaming columns to be more descriptive-----
    df = df.rename(columns={'Price' : 'Close', 'Vol.' : 'Volume', 'Change %' : 'Change_Percentage'})

    # -----Convering columns to correct data types-----
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date # Removes the time component from datetime (ex: 00:00)
    df['Change_Percentage'] = df['Change_Percentage'].str.replace('%','').astype(float)
    for col in ['Close', 'Open', 'High', 'Low']:
        df[col] = df[col].str.replace(',','').astype(float)
    df['Volume'] = df['Volume'].apply(clean_volume)

    df = df.sort_values(by='Date', ascending=True)
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month_Name'] = pd.to_datetime(df['Date']).dt.strftime('%B')

    # -----Dataset summary-----
    maxPriceIndex = df['High'].idxmax()
    maxPrice = df.loc[maxPriceIndex,'High']
    maxPriceDate = df.loc[maxPriceIndex,'Date']

    minPriceIndex = df['Low'].idxmin()
    minPrice = df.loc[minPriceIndex, 'Low']
    minPriceDate = df.loc[minPriceIndex, 'Date']

    maxChangeIndex = df['Change_Percentage'].idxmax()
    maxChange = df.loc[maxChangeIndex, 'Change_Percentage']
    maxChangeDate = df.loc[maxChangeIndex, 'Date']

    minChangeIndex = df['Change_Percentage'].idxmin()
    minChange = df.loc[minChangeIndex, 'Change_Percentage']
    minChangeDate = df.loc[minChangeIndex, 'Date']

    # -----Moving Averages-----
    rolling_7_day = df['Close'].rolling(window=7)
    rolling_30_day = df['Close'].rolling(window=30)
    rolling_21_day = df['Close'].rolling(window=21)
    df['7_Day_MA'] = rolling_7_day.mean()
    df['30_Day_MA'] = rolling_30_day.mean()
    df['21_Day_MA'] = rolling_21_day.mean()

    # -----Longest winning/losing streaks-----
    df['Change_Sign'] = df['Change_Percentage'].apply(streak_sign)
    # Identify where streaks reset by comparing the current row's Change_Sign with the previous row's value.
    # If the sign changes (e.g., from positive to negative or vice versa), mark it as True (reset).
    df['Streak_Reset'] = (df['Change_Sign'] != df['Change_Sign'].shift(1))

    # Each time a streak resets (i.e., Streak_Reset is True), the cumulative sum increments by 1
    # This gives a unique group ID for every streak.
    df['Streak_Group'] = df['Streak_Reset'].cumsum()
    grouped = df.groupby('Streak_Group')
    df['Streak_Length'] = grouped.cumcount()+1

    winningStreaks = df[df['Change_Sign'] == 1]
    losingStreaks = df[df['Change_Sign'] == -1]

    maxWinningStreak = winningStreaks['Streak_Length'].max()
    maxLosingStreak = losingStreaks['Streak_Length'].max()

    # Filter rows matching the longest streak length
    longestWinningStreak = winningStreaks[winningStreaks['Streak_Length'] == maxWinningStreak]
    longestLosingStreak = losingStreaks[losingStreaks['Streak_Length'] == maxLosingStreak]

    # Winning streak calculations
    winningStreakEnd = longestWinningStreak['Date'].iloc[-1] # Last occurrence of the streak
    endIndex = longestWinningStreak.index[-1]
    startIndex = endIndex - (maxWinningStreak - 1)
    winningStreakStart = df.loc[startIndex, 'Date']

    # Ensure the start date is earlier than the end date
    if winningStreakStart > winningStreakEnd:
        winningStreakStart, winningStreakEnd = winningStreakEnd, winningStreakStart

    # Losing streak calculations
    losingStreakEnd = longestLosingStreak['Date'].iloc[-1] # Last occurrence of the streak
    endIndex = longestLosingStreak.index[-1]
    startIndex = endIndex - (maxLosingStreak - 1)
    losingStreakStart = df.loc[startIndex, 'Date']

    # Ensure the start date is earlier than the end date
    if losingStreakStart > losingStreakEnd:
        losingStreakStart, losingStreakEnd = losingStreakEnd, losingStreakStart

    # -----RSI-----
    # The difference in closing prices between consecutive days
    df['Price_Change'] = df['Close'].diff()
    df['Gain'] = 0.0
    df['Loss'] = 0.0
    df.loc[df['Price_Change'] > 0, 'Gain'] = df['Price_Change'] # Positive changes are gains
    df.loc[df['Price_Change'] < 0, 'Loss'] = -df['Price_Change'] # Convert negative changes to positive for losses
    window = 14
    df['Avg_Gain'] = df['Gain'].rolling(window=window, min_periods=1).mean()
    df['Avg_Loss'] = df['Loss'].rolling(window=window, min_periods=1).mean()
    df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
    # RSI formula/calculation
    df['RSI'] = 100 - (100 / (1 + df['RS']))

    # -----Visualizations-----
    st.title('Bitcoin Price')

    # Checkboxes for each Moving Average
    show_7day  = st.checkbox("Show 7-Day MA",  value=False)
    show_21day = st.checkbox("Show 21-Day MA", value=False)
    show_30day = st.checkbox("Show 30-Day MA", value=False)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='BTC Price (USD)',
        line={'color': '#F5C518', 'width': 2}
    ))
    if show_7day:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['7_Day_MA'],
            mode='lines',
            name='7-Day MA',
            line={'color': '#1DA1F2', 'width': 1}
        ))
    if show_21day:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['21_Day_MA'],
            mode='lines',
            name='21-Day MA',
            line={'color': '#2ECC71', 'width': 1}
        ))
    if show_30day:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['30_Day_MA'],
            mode='lines',
            name='30-Day MA',
            line={'color': '#FF5E79', 'width': 1}
        ))
    fig.update_layout(
        title="Bitcoin Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()