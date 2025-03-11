import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots # For secondary y-axis on graph
import yfinance as yf
import datetime
from pytrends.request import TrendReq

st.set_page_config(layout='wide')


def get_updated_df(csv_path='data/bitcoin_data.csv'):
    # load old csv
    df_old = pd.read_csv(csv_path)

    # make sure Date column is datetime
    df_old['Date'] = pd.to_datetime(df_old['Date'])

    # convert old's Price/Open/High/Low to numeric in case there's commas or string
    df_old['Price'] = df_old['Price'].replace({',':''}, regex=True).astype(float)
    df_old['Open']  = df_old['Open'].replace({',':''},  regex=True).astype(float)
    df_old['High']  = df_old['High'].replace({',':''},  regex=True).astype(float)
    df_old['Low']   = df_old['Low'].replace({',':''},   regex=True).astype(float)

    # remove % from change %, in case
    if 'Change %' in df_old.columns:
        df_old['Change %'] = df_old['Change %'].replace({'%':''}, regex=True).astype(float)
        df_old.rename(columns={'Change %': 'Change_Percentage'})

    # figure out last date prior to updating
    last_date = df_old['Date'].max()
    start_date = last_date + pd.Timedelta(days=1)

    # download new data from yfinance
    today = datetime.datetime.today().date()
    if start_date.date() > today:
        return df_old  # Nothing new

    data_new = yf.download(
        'BTC-USD',
        start=start_date,
        end=today,
        group_by='column',    # flatten columns ex: Open_BTC-USD
        auto_adjust=False,
        progress=False
    )

    if data_new.empty:
        return df_old

    # This next step ensures we drop the second level if it’s still MultiIndex
    if isinstance(data_new.columns, pd.MultiIndex):
        data_new.columns = data_new.columns.droplevel(1)

    # --- 3) data_new cleanup & rename ---
    data_new.reset_index(inplace=True)  # Move DatetimeIndex -> 'Date'
    data_new.rename(
        columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Price',  # We match the old CSV's "Price" column name
        },
        inplace=True
    )

    # Keep only the columns we actually use
    data_new = data_new[['Date', 'Price', 'Open', 'High', 'Low']]

    # Convert to numeric (yfinance is usually already numeric, but let’s be sure)
    for col in ['Price','Open','High','Low']:
        data_new[col] = data_new[col].astype(float)

    # Create a daily % change so it matches the old CSV’s concept of "Change %"
    # or name it "Change_Percentage"—whichever you prefer. For example:
    data_new['Change %'] = data_new['Price'].pct_change() * 100

    # Make sure Date is datetime
    data_new['Date'] = pd.to_datetime(data_new['Date'])

    # --- 4) Combine old + new ---
    df_combined = pd.concat([df_old, data_new], ignore_index=True)
    df_combined.sort_values(by='Date', inplace=True)
    df_combined.drop_duplicates(subset='Date', keep='first', inplace=True)

    # --- 5) Save updated file & return ---
    df_combined.to_csv(csv_path, index=False)
    return df_combined

def get_updated_sp(csv_path='data/snp500.csv'):
    df_old = pd.read_csv(csv_path)
    df_old['Date'] = pd.to_datetime(df_old['Date'])

    if 'Change_Percentage' not in df_old.columns and 'Change %' in df_old.columns:
        df_old.rename(columns={'Change %': 'Change_Percentage'}, inplace=True)
    if 'Change_Percentage' in df_old.columns:
        df_old['Change_Percentage'] = (df_old['Change_Percentage']
        .replace({'%':''}, regex=True) # in case it has "%" sign
        .replace({',':''}, regex=True) # in case it has commas
        .astype(float)
        )
    for col in ['Price', 'Open', 'High','Low']:
        if col in df_old.columns:
            df_old[col] = (df_old[col].replace({',':''}, regex=True).astype(float))

    last_date = df_old['Date'].max()
    start_date = last_date + pd.Timedelta(days=1)

    today = datetime.datetime.today().date()
    if start_date.date() > today:
        return df_old

    # fetch data from yfinance for S&P500 index
    data_new = yf.download(
        '^GSPC', # S&P500 index ticker
        start=start_date,
        end=today,
        group_by='column', # flatten multi-index
        auto_adjust=False,
        progress=False
    )
    
    if data_new.empty:
        return df_old

    # flatten if still multi level
    if isinstance(data_new.columns, pd.MultiIndex):
        data_new.columns = data_new.columns.droplevel(1)

    # reset index to get 'Date' as column
    data_new.reset_index(inplace=True)

    data_new.rename(
        columns={
            'Date':'Date',
            'Open':'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Price'
        },
        inplace=True
    )

    # keep only columns used
    data_new = data_new[['Date', 'Price', 'Open', 'High', 'Low']]

    # convert them to float
    for col in ['Price', 'Open', 'High', 'Low']:
        data_new[col] = data_new[col].astype(float)
    
    # compute daily % change
    data_new['Change_Percentage'] = data_new['Price'].pct_change() * 100

    # make sure Date is datetime
    data_new['Date'] = pd.to_datetime(data_new['Date'])

    # concat old + new
    df_combined = pd.concat([df_old, data_new], ignore_index=True)
    df_combined.sort_values(by='Date', inplace=True)
    df_combined.drop_duplicates(subset='Date', keep='first', inplace=True)

    # save and return
    df_combined.to_csv(csv_path, index=False)
    return df_combined

def streak_sign(value):
    if value > 0:
        return 1 # Winning day
    elif value < 0:
        return -1 # Losing day
    else:
        return 0 # Neutral day

def get_updated_trends(csv_path='data/btcgoogle.csv', search_term='Bitcoin'):
    df_old = pd.read_csv(csv_path)
    df_old.rename(
        columns={'Month': 'Trends_Date', 'bitcoin: (Worldwide)': 'Trend_Score'},
        inplace=True
    )

    df_old['Trends_Date'] = pd.to_datetime(df_old['Trends_Date'])
    df_old.sort_values(by='Trends_Date', inplace=True)
    df_old.reset_index(drop=True, inplace=True)

    # connect to pytrends
    pytrends = TrendReq(hl='en-US', tz=360)
    try:
        pytrends.build_payload(
            kw_list=['Bitcoin'],
            timeframe='today 5-y', # 5y window
            geo='', # '' = worldwide
            gprop=''
        )

        df_trends=pytrends.interest_over_time()
    except Exception as e:
        print(f'Pytrends fetch failed: {e}')
        return df_old
    if df_trends.empty:
        return df_old
    
    # reset index so 'date' is a normal column
    df_trends.reset_index(inplace=True) # 'date' becomes a column

    df_trends.rename(columns={
        'date': 'Trends_Date',
        'Bitcoin': 'Trend_Score'
    }, inplace=True
    )

    # drop the 'isPartial' column that indicates whether a particular
    # data point is partial or incomplete
    if 'isPartial' in df_trends.columns:
        df_trends.drop(columns=['isPartial'], inplace=True)
    
    # convert trends_date to date. for weekly data, each row is the start of that week
    df_trends['Trends_Date'] = pd.to_datetime(df_trends['Trends_Date'])

    df_trends.sort_values(by='Trends_Date', inplace=True)
    df_final = df_trends.copy()
    df_final.to_csv(csv_path, index=False)
    return df_final

def main():
    df_trends = get_updated_trends('data/btcgoogle.csv')

    df_trends['Trends_Date'] = pd.to_datetime(df_trends['Trends_Date'])
    df_trends['Trends_Date'] = df_trends['Trends_Date'].dt.date

    df_trends.sort_values(by='Trends_Date', inplace=True)

    df = get_updated_df('data/bitcoin_data.csv')

    df.rename(columns={'Price': 'Close'}, inplace=True)

    # -----Convering columns to correct data types-----
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date # Removes the time component from datetime (ex: 00:00)
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

    #-----Bollinger Bands-----
    # 20 day Simple Moving Average
    df['BB_Middle'] = df['Close'].rolling(window=20).mean().round(1)

    # 20 day rolling standard deviation
    df['BB_Std'] = df['Close'].rolling(window=20).std()

    # Upper Band = Middle + 2 * Std
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std'].round(1)

    # Lower Band = Middle - 2 * Std
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std'].round(1)

    # -----Bitcoin vs Google Trends-----
    df_trends = pd.read_csv('data/btcgoogle.csv', parse_dates=['Trends_Date'])
    df_trends['Trends_Date'] = df_trends['Trends_Date'].dt.date
    df_trends = df_trends.sort_values(by='Trends_Date', ascending=True)

    # ----- Bitcoin vs Stock Market-----
    df_sp = get_updated_sp('data/snp500.csv')
    df_sp = df_sp.rename(columns={'Price': 'Close'})
    # df_sp = df_sp.rename(columns={'Price' : 'Close', 'Vol.' : 'Volume', 'Change %' : 'Change_Percentage'})
    # df_sp['Date'] = pd.to_datetime(df_sp['Date'])
    # df_sp['Date'] = df_sp['Date'].dt.date
    #df_sp['Change_Percentage'] = df_sp['Change_Percentage'].str.replace('%','').astype(float)
    # for col in ['Close', 'Open', 'High', 'Low']:
    #     df_sp[col] = df_sp[col].str.replace(',','').astype(float)
    # df_sp = df_sp.sort_values(by='Date', ascending=True)

    # -----Visualizations-----

    #-----Bitcoin Price-----
    st.title('Bitcoin Data Explorer')
    st.subheader('Bitcoin Price (USD)')
    st.write("A historical daily closing price chart for Bitcoin in USD. Hover to see exact prices on any date. This overview highlights Bitcoin's long-term price evolution.")
    fig_simple = go.Figure()
    fig_simple.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='BTC Price (USD)',
        line={'color': '#F5C518', 'width': 2}
    ))
    
    fig_simple.update_layout(
        title='Bitcoin Price',
        hovermode='x unified'
    )

    fig_simple.update_xaxes(title_text='Date')
    fig_simple.update_yaxes(title_text='Price (USD)', tickformat=',')

    st.plotly_chart(fig_simple,use_container_width=True)


    #-----Investment Calculator-----
    st.markdown("---")
    st.subheader('Bitcoin Investment Calculator')
    st.write("Choose a start date, end date, and an investment amount. We'll calculate how much your investment would be worth and your ROI. A simple 'what-if' tool for exploring potential gains or losses.")
    # 1) Start Date
    start_date = st.date_input(
        'Start Date',
        value=pd.to_datetime('2020-01-01') # Default date
    )
    # 2) End Date
    end_date = st.date_input(
        'End Date',
        value=pd.to_datetime('2024-01-01') # Default date
    )
    # 3) Investment Amount
    investment_amount = st.number_input(
        'Investment Amount (USD)',
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    if start_date > end_date:
        st.error("Error: Start date cannot be after end date. Please choose valid dates.")
        st.stop()

    start_price_row = df[df['Date'] == start_date]
    end_price_row = df[df['Date'] == end_date]

    if len(start_price_row)==0:
        st.error(f'No price data for start date {start_date}. Please pick another date.')
    else:
        start_price = float(start_price_row['Close'].iloc[0])
    if len(end_price_row)==0:
        st.error(f'No price data for end date {end_date}. Please pick another date.')
    else:
       end_price = float(end_price_row['Close'].iloc[0])

    btc_bought = investment_amount / start_price
    final_usd_value = btc_bought * end_price
    if investment_amount==0.0:
        roi_percent = 0
    else:
        roi_percent = ((final_usd_value - investment_amount) / investment_amount) * 100

    st.write(f'**Profit**: ${final_usd_value-investment_amount:,.2f}')
    st.write(f"**ROI**: {roi_percent:.2f}%")

    #-----Bitcoin with Indicators-----
    st.markdown("---")
    st.subheader('Bitcoin Price with Technical Indicators')
    st.write("Toggle popular indicators often used by traders. Each helps analyze price trends and potential market conditions.")
    st.write("(**Tip: Scroll below the chart to find details about each indicator!**)")
    col1, col2 = st.columns([1, 4])

    with col1:
        # Checkboxes for each Moving Average and RSI
        show_7day  = st.checkbox("7 MA",  value=False)
        show_21day = st.checkbox("21 MA", value=False)
        show_30day = st.checkbox("30 MA", value=False)
        show_rsi = st.checkbox("RSI", value=False)
        show_bbands = st.checkbox("Bollinger Bands (20, 2.0)", value=True)

    with col2:
        fig = make_subplots(specs=[[{'secondary_y': True}]])

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
            ),
            secondary_y=False
            )

        if show_21day:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['21_Day_MA'],
                mode='lines',
                name='21-Day MA',
                line={'color': '#2ECC71', 'width': 1}
            ),
            secondary_y=False
            )

        if show_30day:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['30_Day_MA'],
                mode='lines',
                name='30-Day MA',
                line={'color': '#FF5E79', 'width': 1}
            ),
            secondary_y=False)

        if show_rsi:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line={'color': 'white', 'width': 2},
                opacity=0.5
            ),
            secondary_y=True
            )
        
        if show_bbands:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['BB_Upper'],
                mode='lines',
                name='Upper Bollinger Band',
                line={'color': '#BB6BD9', 'width': 1},
                opacity=0.6
            ),
            secondary_y=False
            )

            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['BB_Lower'],
                mode='lines',
                name='Lower Bollinger Band',
                line={'color': '#AA5BFF', 'width': 1},
                opacity=0.6,
                fill='tonexty',  # Fill to the previous trace
                fillcolor='rgba(128, 128, 128, 0.25)' # Light gray shading
            ),
            secondary_y=False
            )

            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['BB_Middle'],
                mode='lines',
                name='Middle Bollinger Band',
                line={'color': '#9B51E0', 'width': 1}
            ),
            secondary_y=False
            )

        fig.update_layout(
            title="Bitcoin Price",
            hovermode="x unified"
        )

        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price (USD)',secondary_y=False, tickformat=',')
        fig.update_yaxes(title_text='RSI',secondary_y=True)

        st.plotly_chart(fig)
        st.write("**Moving Averages (7 day, 21 day, 30 day):**")
        st.write("- These MAs smooth out short-term price fluctuations, helping identify overall trends. A short MA (like 7-day) reacts quickly to price changes, while longer MAs (21- or 30-day) provide more stable trend views.")
        st.write("**RSI (Relative Strength Index):**")
        st.write("- RSI measures the speed and magnitude of recent price changes, oscillating between 0 and 100. Traditionally, RSI above 70 indicates overbought conditions; below 30 suggests oversold.")
        st.write("**Bollinger Bands:**")
        st.write("- Bollinger Bands show a middle line (the moving average) plus upper/lower bands that are typically ±2 standard deviations from the MA. They can help gauge volatility — when bands are wide, volatility is higher.")
    
    #-----Bitcoin vs Stock Market-----
    st.markdown("---")
    st.subheader('Bitcoin Price vs Stock Market (S&P 500 Index)')
    st.write("Compares Bitcoin's daily price to S&P 500 index prices. See if there's any correlation between Bitcoin and traditional stock market trends.")
    fig_stock = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_stock.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='BTC Price (USD)',
        line={'color': '#F5C518', 'width': 2}
    ),
    secondary_y=False
    )

    fig_stock.add_trace(go.Scatter(
        x=df_sp['Date'],
        y=df_sp['Close'],
        mode='lines',
        name='S&P 500 Close Price',
        line={'color': '#0073e6', 'width': 2}
    ),
    secondary_y=True
    )

    fig_stock.update_layout(
        title='Bitcoin Price vs. Stock Market (S&P 500)',
        hovermode='x unified'
    )

    fig_stock.update_xaxes(title_text='Date')
    fig_stock.update_yaxes(title_text='BTC Price (USD)', secondary_y=False, tickformat=',')
    fig_stock.update_yaxes(title_text='S&P 500 Price (USD)', secondary_y=True, tickformat=',')

    st.plotly_chart(fig_stock, use_container_width=True)

    #-----Bitcoin Price vs Google Trends-----
    st.markdown("---")
    st.subheader('Bitcoin Price vs Google Trends')
    st.write("Compares Bitcoin's daily price to monthly Google search interest (Trend Score). See if public attention correlates with major price changes.")
    fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trends.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='BTC Price (USD)',
        line={'color': '#F5C518', 'width': 2}
    ),
    secondary_y=False
    )

    fig_trends.add_trace(go.Scatter(
        x=df_trends['Trends_Date'],
        y=df_trends['Trend_Score'],
        mode='lines+markers',
        name='Google Trend Score',
        line={'color': '#ADD8E6', 'width': 2}
    ),
    secondary_y=True
    )

    fig_trends.update_layout(
        title='Bitcoin Price vs. Google Trends',
        hovermode='x unified'
    )

    fig_trends.update_xaxes(title_text='Date')
    fig_trends.update_yaxes(title_text='BTC Price (USD)', secondary_y=False, tickformat=',')
    fig_trends.update_yaxes(title_text='Trend Score (0-100)', secondary_y=True)

    st.plotly_chart(fig_trends, use_container_width=True)

    # Format the dates into "Day Month Year" format
    minPriceDateFormatted = pd.to_datetime(minPriceDate).strftime("%d %B %Y")
    maxPriceDateFormatted = pd.to_datetime(maxPriceDate).strftime("%d %B %Y")
    winningStreakStartFormatted = pd.to_datetime(winningStreakStart).strftime("%d %B %Y")
    winningStreakEndFormatted = pd.to_datetime(winningStreakEnd).strftime("%d %B %Y")
    losingStreakStartFormatted = pd.to_datetime(losingStreakStart).strftime("%d %B %Y")
    losingStreakEndFormatted = pd.to_datetime(losingStreakEnd).strftime("%d %B %Y")

    #-----Fun Facts-----
    st.markdown("---")
    st.subheader('Fun Facts about Bitcoin (based on the data)!')
    st.warning(f"🏆 **All-time Minimum Price**: **${minPrice:,.2f}** on **{minPriceDateFormatted}**")
    st.info(f"🚀 **All-time Maximum Price**: **${maxPrice:,.2f}** on **{maxPriceDateFormatted}**")
    st.success(f"🟢 **Longest Winning Streak**: **{maxWinningStreak} days**\n- Started on: **{winningStreakStartFormatted}**\n- Ended on: **{winningStreakEndFormatted}**")
    st.error(f"🔴 **Longest Losing Streak**: **{maxLosingStreak} days**\n- Started on: **{losingStreakStartFormatted}**\n- Ended on: **{losingStreakEndFormatted}**")


if __name__ == "__main__":
    main()