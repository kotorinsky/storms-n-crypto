import pandas as pd


# Function to load dataset from a file
def load_data(filename):
    data = pd.read_csv(filename)
    texts = data.iloc[:, 0].astype(str)  # Extract texts
    labels = data.iloc[:, 1].astype('int')  # Extract labels
    return texts, labels

# Load additional dataset from a file
def load_additional_data(filename):
    data = pd.read_csv(filename)
    return data['text'].astype(str)

# Load crypto keywords from a file
def load_crypto_keywords(filename):
    data = pd.read_csv(filename)
    full_names = data.iloc[:, 1].dropna().astype(str).tolist()  # Full names of cryptocurrencies
    abbreviations = data.iloc[:, 2].dropna().astype(str).tolist()  # Abbreviations of cryptocurrencies
    return set(full_names + abbreviations)
