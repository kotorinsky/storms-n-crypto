import pandas as pd
import argparse
import glob

# Define the column names for the DataFrame
columns = [
    "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

# Initialize an empty dictionary to store timestamp and open price pairs
timestamp_open_price = {}

# Function to load data from a CSV file and update the dictionary
def load_data(file_path):
    # Read CSV file with no header in the file, so headers are None
    # and assign the columns using the names parameter
    df = pd.read_csv(file_path, header=None, names=columns)
    
    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        # Use open_time as key and open price as value
        timestamp_open_price[row['open_time']] = row['open']

def date_to_unix(date_str):
    dt = pd.to_datetime(date_str)  # Convert string to datetime
    dt = dt.floor('T')  # Floor to the nearest minute
    return int(dt.timestamp() * 1000)  # Convert to Unix time in milliseconds

def find_next_15min_prices(start_unix_time):
    prices = []
    for i in range(16):  # Next 15 minutes including the starting minute
        time_key = start_unix_time + i * 60000  # 60000 milliseconds = 1 minute
        price = timestamp_open_price.get(time_key, None)
        if price is not None:
            prices.append(price)
        else:
            print('price at time is none', start_unix_time)
    return prices

def is_valid_date(date_str):
    try:
        dt = pd.to_datetime(date_str)  # Attempt to convert to datetime
        dt = dt.floor('T')
        dt.timestamp()
        return True
    except:
        return False  # Return False if conversion fails

def main(in_file_path, out_file_path):
    # Glob pattern to match all CSV files
    file_paths = 'data/btc-prices/BTCUSDT-1m-*.csv'
    
    # Use glob to find all files matching the pattern
    for file_path in glob.glob(file_paths):
        print("Load price data:", file_path)
        load_data(file_path)
    
    tweets_df = pd.read_csv(in_file_path)
    valid_dates_df = tweets_df[tweets_df['Timestamp'].apply(is_valid_date)]
    valid_dates_df['next_15min_prices'] = valid_dates_df['Timestamp'].apply(lambda x: find_next_15min_prices(date_to_unix(x)))
    file_path = out_file_path 
    filtered_df = valid_dates_df[valid_dates_df['next_15min_prices'].apply(lambda x: len(x) > 0)]
    
    filtered_df.to_csv(file_path, index=False)
    print("Successfully saved converted csv to:", file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('in_file_path', type=str, help='Input file path')
    parser.add_argument('out_file_path', type=str, help='Output file path')

    args = parser.parse_args()
    if not (args.in_file_path and args.out_file_path):
       parser.error("You must provide both 'in_file_path' and 'out_file_path' arguments.")
    main(args.in_file_path, args.out_file_path)
