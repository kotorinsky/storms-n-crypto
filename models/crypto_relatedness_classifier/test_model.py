import sys
import joblib
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re  # For using regular expressions

# Load necessary NLTK modules (assuming they are already downloaded)
nltk.download('punkt')  # Tokenizers

def load_model(filename):
    return joblib.load(filename)

def preprocess_input(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)

def predict(model, text):
    processed_text = preprocess_input(text)
    prediction = model.predict([processed_text])
    return prediction[0]

# Load crypto keywords from a file
def load_crypto_keywords(filename):
    data = pd.read_csv(filename)
    full_names = data.iloc[:, 1].dropna().astype(str).tolist()  # Full names of cryptocurrencies
    abbreviations = data.iloc[:, 2].dropna().astype(str).tolist()  # Abbreviations of cryptocurrencies
    return set(full_names + abbreviations)

# Function to check if input contains any of the crypto keywords
def check_keywords(input_text, crypto_keywords):
    for keyword in crypto_keywords:
        # Match exact whole word with case-sensitive checking
        if re.search(r'\b{}\b'.format(re.escape(keyword)), input_text):
            return 1
    return 0

def main():
    #model_filename = 'models/crypto_relatedness_classifier/trained_model1.joblib'
    model_filename = 'models/crypto_relatedness_classifier/trained_model2.joblib'
    model = load_model(model_filename)
    crypto_keywords = load_crypto_keywords('data/top-50.csv')
    print("Please input text for prediction:")
    input_text = input()  # Use input() to get text from the user after the script starts running
    result = check_keywords(input_text, crypto_keywords)
    
    if result == 1:
        print("Output:", result)
    else:
        result = predict(model, input_text)
        print(f"Prediction for input '{input_text}': {result}")





if __name__ == '__main__':
    main()
    



