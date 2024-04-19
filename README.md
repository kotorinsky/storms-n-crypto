# Storms and Crypto

## Setup
Install the required dependencies
```
pip install -r requirements.txt
```

## Data tagging 
### Tag crypto tweets
We used ChatGpt to help tag if a tweet is crypto related.
1. Set your your open ai api key in `.env` file
```
OPEN_AI_API_KEY=your-api-key
```
2.
```
$ python process/crypto-relatedness-tagging.py
```
A `data/tweets-from-influentials.csv` will be generated after.
