import argparse
import pandas as pd
import os
import re
from dotenv import load_dotenv
import openai

load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")  # Global variable to store the API key
if not api_key:
    exit("an openai api key need to be specified in the .env file")
openai.api_key = api_key 

def read_and_clean_data(filepath):
    # Load the csv file into a pandas DataFrame
    df = pd.read_csv(filepath)
    
    # Define regex patterns to match @xxxx and https://xxx
    pattern_at = r'@\w+'
    pattern_http = r'https?://[^\s]+'
    pattern_newline = r'\n'
 
    # Define a function to apply to each tweet to remove the patterns
    def clean_tweet(tweet):
        tweet = re.sub(pattern_at, '', tweet)
        tweet = re.sub(pattern_http, '', tweet)
        return re.sub(pattern_newline, ' ', tweet)
        
    # Apply the function to each tweet in the DataFrame
    df['text'] = df['text'].apply(clean_tweet)
    
    return df



def classify_sentences(sentences):
    sentences = "Sentences: \n" + "\n".join(sentences)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": """
    please help label some tweets, label them whether they are crypto related.  A crypto related tweet is a tweet which talks about or hint specific crypto.   Label all below sentences using true/false.  True means crypto related.
    
    e.g.  "ethereum road map for Q4 is out"  - this should be labelled as true, as its related to ethereum
    
    "Let cultural evolution figure that out; it's smarter than me or any of us."  this should be labeled as false.
    example output(append True/False at the end of the each setence):
    xxxxx: True,
    xxxxx: False 
            """
        },
        {"role": "user", "content": sentences}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def process_data(df):
    results = []
    for i in range(0, len(df), 5):
        # Get the next 5 tweets
        tweets = df.iloc[i:i+5]
        print("Processing ", i, "th tweets")
        # Classify the sentences
        response = classify_sentences(tweets.text.tolist())

        # Parsing the result
        # Split the response into sentence-label pairs
        pairs = response.split('\n')

        # Validate the number of results. If not equal to number of input sentences,
        # assign all sentences to 'Unknown'
        if len(pairs) != len(tweets):  
            results.extend(['Unknown'] * len(tweets))
        else:
            # Extract the label (as Boolean) from the end of each pair and add it to the results list
            for pair in pairs:
                if 'True' in pair[-10:]:
                    results.append('True')
                elif 'False' in pair[-10:]:
                    results.append('False')
                else:
                    results.append('Unknown')

    # Add the results to a new column in the DataFrame
    df['is_crypto_related'] = results

    return df


if __name__ == '__main__':
    directory = 'data/tweets-from-influentials/'
    all_dfs = []

    # For each CSV file in the directory
    for filename in os.listdir(directory):
        # Only process csv files
        if filename.endswith('.csv'):
            df = read_and_clean_data(os.path.join(directory, filename))
            df = df.iloc[0:1]
            df = df.drop('created_at', axis=1)
            df = df.drop('id', axis=1)
            all_dfs.append(df)

    tweets = pd.concat(all_dfs, ignore_index=True)
    print('Total number of tweets:', len(tweets))
    df = process_data(tweets)
    df.to_csv('data/tweets-from-influentials.csv', index=False)
