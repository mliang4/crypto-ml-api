import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_crypto_data(symbol='BTC', days=365):
    """Fetch historical crypto data from CoinGecko API (free)"""
    url = f'https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    # Ensure volume list matches prices length
    volumes = [v[1] for v in data['total_volumes']]
    if len(volumes) < len(df):
        # Pad with NaN if volumes are shorter
        volumes += [float('nan')] * (len(df) - len(volumes))
    df['volume'] = volumes[:len(df)]
    df['volume'] = [v[1] for v in data['total_volumes']]
    
    return df

def create_features(df):
    """Engineer features for prediction"""
    df = df.copy()
    
    # Price-based features
    df['price_change'] = df['price'].pct_change()
    df['price_ma_7'] = df['price'].rolling(window=7).mean()
    df['price_ma_30'] = df['price'].rolling(window=30).mean()
    df['price_std_7'] = df['price'].rolling(window=7).std()
    
    # Volume features
    df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
    # Target: next day price movement (1 = up, 0 = down)
    # Compares each day's price to the next day's price; if next day's price is higher, target=1, else target=0
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    for lag in [1, 3, 7]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Target: next day price movement (1 = up, 0 = down)
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

if __name__ == '__main__':
    # Test the functions
    print("Fetching Bitcoin data...")
    df = fetch_crypto_data('bitcoin', days=365)
    print(f"Downloaded {len(df)} days of data")
    
    print("\nCreating features...")
    df_features = create_features(df)
    print(f"Created {len(df_features.columns)} features")
    # Save to disk
    import os
    os.makedirs('data', exist_ok=True)
    df_features.to_csv('data/bitcoin_features.csv', index=False)
    print("\nData saved to data/bitcoin_features.csv")