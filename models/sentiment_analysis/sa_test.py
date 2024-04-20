import numpy as np
import os
import preprocessor as tp
import re
import torch
import torch.nn as nn

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

class LSTM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_size,
        vocab_size,
        output_size,
        padding_idx,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            2,
            bidirectional=True,
            dropout=0.5,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, ids, length):
        # print(length.shape, ids.shape)
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        prediction = self.fc(hidden)
        return prediction

class SentimentAnalysisModel:
    def __init__(self, max_length = 53):
        path = os.path.dirname(os.path.abspath(__file__))
        self.vocabulary = torch.load(path + '/vocab.pth')
        self.padding_idx = self.vocabulary["<pad>"]
        self.model = LSTM(embedding_dim=300, hidden_size=64, vocab_size=len(self.vocabulary), output_size=2, padding_idx=self.padding_idx)
        self.model.load_state_dict(torch.load(path + '/lstm.pt'))
        self.lemmatizer = WordNetLemmatizer()
        self.w_tokenizer = TweetTokenizer()
        self.max_length = max_length
        
    
    def prepare_for_sentiment(self, tweet):
        text = tp.clean(tweet)
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s\s+", " ", text)
        text = re.sub(r"\d+[kKmM]*", "", text)
        tokens = [(self.lemmatizer.lemmatize(w)) for w in self.w_tokenizer.tokenize((text))]
        length = len(tokens)
        idx = self.vocabulary.lookup_indices(tokens)
        if len(idx) < self.max_length:
            idx = idx + [self.padding_idx] * max(0, self.max_length - len(idx))
        return np.array([np.array(idx)]), length

    def predict(self, x, length):
        # print(x.shape, length)
        y_pred = self.model(torch.from_numpy(x), [length])
        y = torch.argmax(y_pred, dim=1)
        if y == 0:
            return 'Negative'
        elif y == 1:
            return 'Positive'
        else:
            return 'error'
        
    def sentiment_analysis(self, tweet):
        tokens, length = self.prepare_for_sentiment(tweet)
        return self.predict(tokens, length)
        


