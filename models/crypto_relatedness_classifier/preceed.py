import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import yake

# Download necessary NLTK modules
nltk.download('punkt')  # Tokenizers
nltk.download('stopwords')  # Stopwords for filtering
nltk.download('wordnet')  # Lexical database for lemmatization
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('maxent_ne_chunker')  # NE chunker
nltk.download('words')  # For NE chunker

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
