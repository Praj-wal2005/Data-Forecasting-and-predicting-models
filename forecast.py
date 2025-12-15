import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly

# --- CONFIGURATION ---
START_DATE = '2020-01-01'
PREDICTION_DAYS = 365 

def fetch_data(ticker):
    """Fetches historical stock data from Yahoo Finance."""
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=START_DATE)
    
    # Handle cases where download fails or returns empty data
    if df.empty:
        return None

    df.reset_index(inplace=True)
    
    # Prepare data for Prophet (Requires columns 'ds' and 'y')
    # Flatten multi-index columns if present (common in new yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Check if 'Close' is at level 0 or level 1
            if 'Close' in df.columns.get_level_values(0):
                 # If MultiIndex is (Price, Ticker), we just want the 'Close' column
                df = df.xs('Close', axis=1, level=0, drop_level=True)
            else:
                # Standard drop for other formats
                df.columns = df.columns.droplevel(1)
        except Exception as e:
            print(f"Note: Index flattening adjustment needed: {e}")
            pass # Continue if standard access works below

    # Check if 'Date' exists after reset_index, if not it might be the index name
    if 'Date' not in df.columns and 'Date' not in df.index.names:
         # Sometimes yfinance keeps Date as index even after reset if parameters differ
         df.reset_index(inplace=True)
        
    # Ensure we have the right columns
    # Use 'Adj Close' if available, otherwise 'Close'
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    data = df[['Date', target_col]].copy()
    data.rename(columns={'Date': 'ds', target_col: 'y'}, inplace=True)
    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)
    
    return data

def build_and_train_model(data):
    """Initializes and trains the Prophet model."""
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05 
    )
    # Only add country holidays if the data spans enough time to make sense
    try:
        model.add_country_holidays(country_name='US')
    except:
        pass # Skip if date range issues occur
        
    model.fit(data)
    return model

def make_forecast(model, periods):
    """Generates future dates and predicts values."""
    future = model.make_future_dataframe(periods=periods)
    future = future[future['ds'].dt.dayofweek < 5] # Remove weekends
    forecast = model.predict(future)
    return forecast

# --- FUNCTION 1: FORECAST FROM STOCK TICKER ---
def run_prediction(ticker):
    try:
        # 1. Get Data
        data = fetch_data(ticker)
        if data is None:
            return None, "Error: Could not fetch data. Check ticker symbol."

        # 2. Train Model
        model = build_and_train_model(data)

        # 3. Forecast
        forecast = make_forecast(model, PREDICTION_DAYS)
        
        # 4. Get latest prediction for text summary
        latest_prediction = forecast.iloc[-1]
        prediction_text = f"Forecast for {ticker}: Predicted price on {latest_prediction['ds'].strftime('%Y-%m-%d')} is ${latest_prediction['yhat']:.2f}"

        # 5. Generate Plot HTML
        fig = plot_plotly(model, forecast)
        fig.update_layout(
            title=f'Price Forecast for {ticker}',
            xaxis_title="Date",
            yaxis_title="Stock Price (USD)",
            template="plotly_dark",
            autosize=True
        )
        
        plot_html = plotly.io.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
        return plot_html, prediction_text

    except Exception as e:
        return None, f"An error occurred: {str(e)}"

# --- FUNCTION 2: FORECAST FROM UPLOADED FILE ---
def run_prediction_from_file(df):
    """
    Processes an uploaded pandas DataFrame and generates a forecast.
    """
    try:
        # 1. Prepare Data
        # We need to find the date column and the value column
        df.columns = [str(c).lower() for c in df.columns] # make lowercase to search easily
        
        # Find column with 'date' or 'time'
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
        # Find column with 'close', 'value', or 'sales'
        val_col  = next((c for c in df.columns if 'close' in c or 'value' in c or 'sales' in c), None)

        # Fallback: if not found by name, take 1st and 2nd columns
        if not date_col: date_col = df.columns[0]
        if not val_col: val_col = df.columns[1]

        # Rename to Prophet requirements ('ds' and 'y')
        data = df.rename(columns={date_col: 'ds', val_col: 'y'})
        data = data[['ds', 'y']] # Keep only strictly these two
        data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None) # Fix date format
        
        # Drop any rows where data is missing
        data = data.dropna()

        # 2. Train Model
        model = build_and_train_model(data)

        # 3. Forecast
        forecast = make_forecast(model, PREDICTION_DAYS)
        
        # 4. Text Summary
        latest_prediction = forecast.iloc[-1]
        prediction_text = f"Forecast Result: Predicted value on {latest_prediction['ds'].strftime('%Y-%m-%d')} is {latest_prediction['yhat']:.2f}"

        # 5. Generate Plot
        fig = plot_plotly(model, forecast)
        fig.update_layout(
            title='Forecast based on Uploaded Data',
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark",
            autosize=True
        )
        
        plot_html = plotly.io.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
        return plot_html, prediction_text

    except Exception as e:
        return None, f"Error processing file: {str(e)}"