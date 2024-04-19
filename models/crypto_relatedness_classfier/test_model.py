import sys
import joblib
import nltk
from nltk.corpus import stopwords

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

if __name__ == '__main__':
    model_filename = 'trained_model.joblib'
    model = load_model(model_filename)
    
    print("Please input text for prediction:")
    input_text = input()  # Use input() to get text from the user after the script starts running
    result = predict(model, input_text)
    print(f"Prediction for input '{input_text}': {result}")
