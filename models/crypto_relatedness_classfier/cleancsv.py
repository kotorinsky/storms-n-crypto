import pandas as pd
import re

# read csv
df = pd.read_csv('data/crypto-without-unknown.csv')

# Print the column name to confirm the correct column name
print("列名：", df.columns)

# Define a function for preprocessing text
def preprocess_text(text):
    # Make sure the text is a string
    if pd.isnull(text):
        return ""
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove text mentioned by @
    text = re.sub(r'@\S+', '', text)
    return text

# Use the correct column name 'text' to process the data
df['text'] = df['text'].apply(preprocess_text)

# The pre-processed data is displayed
print(df.head())

# Save the processed DataFrame back to CSV
df.to_csv('data/tweets-from-influentials-process.csv', index=False)
