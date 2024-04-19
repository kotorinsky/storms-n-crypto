import pandas as pd
import nltk
import re  # For using regular expressions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import yake

# Download necessary NLTK modules
nltk.download('punkt')  # Tokenizers
nltk.download('stopwords')  # Stopwords for filtering
nltk.download('wordnet')  # Lexical database for lemmatization
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('maxent_ne_chunker')  # NE chunker
nltk.download('words')  # For NE chunker

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

# Initialize YAKE keyword extractor
def initialize_yake():
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 20
    return yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords)

# Preprocess text to extract features and check for direct matches in crypto keywords
def preprocess(text, keyword_extractor, crypto_keywords):
    tokens = nltk.word_tokenize(text)  # Tokenize text
    tagged = nltk.pos_tag(tokens)  # POS tagging
    keywords = keyword_extractor.extract_keywords(text)  # Extract keywords using YAKE
    lemmatizer = WordNetLemmatizer()
    words = []
    for word, tag in tagged:
        if word.isalpha() and word not in stopwords.words('english'):  # Remove stopwords and non-alphabetic tokens
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))  # Lemmatize words based on POS
            words.append(lemma)
    features = ' '.join(words)  # Join words to form the final processed text
    has_direct_match = any(re.search(r'\b{}\b'.format(re.escape(keyword)), text) for keyword in crypto_keywords)  # Check for direct keyword matches
    return features, has_direct_match

# Helper function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Build machine learning model using a pipeline
def build_model():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),  # Convert text to TF-IDF vectors
        ('classifier', LogisticRegression())  # Logistic regression classifier
    ])
    return pipeline

def main():
    crypto_keywords = load_crypto_keywords('data/top-50.csv')
    texts, labels = load_data('data/tweets-from-influentials-process.csv')
    keyword_extractor = initialize_yake()
    processed_texts = []
    direct_match_labels = []
    for text in texts:
        features, has_direct_match = preprocess(text, keyword_extractor, crypto_keywords)
        processed_texts.append(features)
        direct_match_labels.append(int(has_direct_match))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(processed_texts, labels, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train)  # Train the model
    predictions = model.predict(X_test)  # Predict on the test set
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Classification Report:\n', classification_report(y_test, predictions))

    additional_data = load_additional_data('data/elon-csv-cleaned.csv')
    additional_processed_data = []
    additional_direct_labels = []
    for text in additional_data:
        features, has_direct_match = preprocess(text, keyword_extractor, crypto_keywords)
        additional_processed_data.append(features)
        additional_direct_labels.append(int(has_direct_match))

    # Predict additional data
    additional_predictions = [label if label == 1 else model.predict([text])[0] for label, text in zip(additional_direct_labels, additional_processed_data)]
    positive_samples = additional_data[[pred == 1 for pred in additional_predictions]]
    print('Number of positive samples:', len(positive_samples))
    print('Positive samples data:\n', positive_samples)

if __name__ == '__main__':
    main()
