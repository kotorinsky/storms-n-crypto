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





## crypto relatedness classifier

The structure is shown in the following figure:

![alt text](README.assets/crypto_relatedness_classfier.png)



crypto_relatedness_classfier.py  full version of the crypto relatedness classifier

cleancsv.py   Determine if the data contains links (mostly advertisements) or @related content and remove them from the text

readcsv.py         csv file reading

preceed.py        data preprocessing

run.py                model training and save the model file to local, and output the test results

test_Elon.py      Test the model on a dataset of Elon Musk's tweets

test_model.py   User-defined input strings are supported. The system determines whether the input strings are related to cryptocurrency

trained_model1.joblib  Models trained against tweets-from-influentials-process.csv

trained_model2.joblib  Models trained against tweets-from-influentials.csv





