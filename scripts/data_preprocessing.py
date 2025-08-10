import pandas as pd
import numpy as np

def clean_data(df):
    """
    Handles missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): The raw data DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("Cleaning data...")
    # Fill missing values using forward fill, then backward fill
    df = df.ffill().bfill()
    # Drop any remaining NaNs (should be none after bfill)
    df.dropna(inplace=True)
    print("Data cleaned successfully.")
    return df

def create_features(df):
    """
    Calculates daily percentage change (returns).
    
    Args:
        df (pd.DataFrame): The cleaned prices DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with daily returns.
    """
    print("Calculating daily returns...")
    returns = df.pct_change().dropna()
    print("Daily returns calculated.")
    return returns

def save_processed_data(prices_df, returns_df, path='../data/processed/'):
    """
    Saves the processed price and returns data to a specified path.
    """
    prices_df.to_csv(f'{path}clean_prices.csv')
    returns_df.to_csv(f'{path}returns.csv')
    print("Processed data saved to 'data/processed' directory.")

if __name__ == '__main__':
    # Example usage (assumes a raw data DataFrame 'prices' exists)
    # clean_prices = clean_data(prices)
    # returns = create_features(clean_prices)
    # save_processed_data(clean_prices, returns)
    pass