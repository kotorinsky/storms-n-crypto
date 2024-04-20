from models.sentiment_analysis.sa_test import SentimentAnalysisModel
from models.crypto_relatedness_classifier.test_model import load_model
from models.crypto_relatedness_classifier.test_model import load_crypto_keywords
from models.crypto_relatedness_classifier.test_model import preprocess_input
from models.crypto_relatedness_classifier.test_model import predict
from models.crypto_relatedness_classifier.test_model import check_keywords


def predict_is_crypto_related(tweet):
    # Placeholder function for predicting if a tweet is crypto-related
    model_filename = 'models/crypto_relatedness_classifier/trained_model2.joblib'
    model = load_model(model_filename)
    crypto_keywords = load_crypto_keywords('data/top-50.csv')
    #print("Please input text for prediction:")
    #input_text = input()  # Use input() to get text from the user after the script starts running
    input_text = tweet
    result = check_keywords(input_text, crypto_keywords)
    
    if result == 1:
        return 'is related to bitcoin' in tweet.lower()
    else:
        result = predict(model, input_text)
        if result == 1:
            return 'is related to bitcoin' in tweet.lower()
        else:
            return 'not related to bitcoin' in tweet.lower()
        #print(f"Prediction for input '{input_text}': {result}")
    #return 'bitcoin' in tweet.lower()

def predict_sentiment(tweet):
    # Placeholder function for predicting the sentiment of a tweet
    s_model = SentimentAnalysisModel()
    return s_model.sentiment_analysis(tweet)  # Placeholder sentiment




if __name__ == "__main__":
    tweets = [
        'You can now buy a Tesla with bitcoin',
        'Tesla would trial run accepting DOGE for merchandise',
        'Tesla buys \$1.5 billion in bitcoin',
        'Tesla would no longer accept BTC as payment',
        'Tesla will make some merch buyable with Doge & see how it goes',
        'Doge',
        'Ur welcome',
        'Working with Doge devs to improve system transaction efficiency. Potentially promising',
        'Important to support',
        'Tesla has suspended vehicle purchases using Bitcoin. We are concerned about rapidly increasing use of fossil fuels for Bitcoin mining and transactions, especially coal, which has the worst emissions of any fuel. Cryptocurrency is a good idea on many levels and we believe it has a promising future, but this cannot come at great cost to the environment. Tesla will not be selling any Bitcoin and we intend to use it for transactions as soon as mining transitions to more sustainable energy. We are also looking at other cryptocurrencies that use <1% of Bitcoin’s energy/transaction.'
    ]

    for tweet in tweets:
        is_crypto_related = predict_is_crypto_related(tweet)
        sentiment = predict_sentiment(tweet)
        
        action = ""
        if is_crypto_related and sentiment == "Positive":
            action = "Buy that crypto rocket ship! 🚀"
        elif not is_crypto_related:
            action = "Just chill and watch the fireworks! 🎇"
        elif is_crypto_related and sentiment == "Negative":
            action = "Time to cash in on the moon ride! 🌕"
        
        print(f"Tweet: {tweet}")
        print(f"Is Crypto Related: {is_crypto_related}")
        print(f"Sentiment: {sentiment}")
        print(f"Action: {action}")
        print("\n")
