import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, output_path):
    """
    Fetches historical financial data from Yahoo Finance and saves it to a CSV file.
    
    Args:
        tickers (list): A list of ticker symbols.
        start_date (str): The start date for the data (e.g., '2015-07-01').
        end_date (str): The end date for the data (e.g., '2025-07-31').
        output_path (str): The path to save the CSV file.
    """
    print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    data.to_csv(output_path)
    print(f"Data successfully fetched and saved to {output_path}")

if __name__ == '__main__':
    tickers = ['TSLA', 'BND', 'SPY']
    start_date = '2015-07-01'
    end_date = '2025-07-31'
    output_path = 'data/raw/financial_data.csv'
    fetch_data(tickers, start_date, end_date, output_path)