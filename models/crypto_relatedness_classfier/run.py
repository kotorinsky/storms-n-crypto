from readcsv import load_data,load_crypto_keywords
from preceed import initialize_yake, preprocess
from train import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

    

if __name__ == '__main__':
    main()
