from sklearn.model_selection import train_test_split
from readcsv import load_data, load_additional_data, load_crypto_keywords
from preceed import initialize_yake, preprocess
from train import build_model

def test_elon():
    crypto_keywords = load_crypto_keywords('top-50.csv')
    texts, labels = load_data('crypto-processed.csv')
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
    # predictions = model.predict(X_test)  # Predict on the test set
    # print('Accuracy:', accuracy_score(y_test, predictions))
    # print('Classification Report:\n', classification_report(y_test, predictions))

    additional_data = load_additional_data('elon-csv-cleaned.csv')
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
    test_elon()