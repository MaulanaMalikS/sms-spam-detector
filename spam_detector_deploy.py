import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('/Data/Pemrograman/Python/ Artificial Intelligence/Latihan/Dataset/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms'])
model = joblib.load('/Data/Pemrograman/Python/ Artificial Intelligence/Latihan/Model/spam_detector.joblib')
vectorizer = TfidfVectorizer(stop_words='english')
X_train, X_test, y_train, y_test = train_test_split(dataset['sms'], dataset['label'], test_size=0.25, random_state=0)
vectorizer.fit(X_train)

st.write("""
    # SMS Spam Detector

    Detect whether an sms is spam or not
""")

sms = [st.text_area('SMS')]
prediction = [[""]]

if sms[0] :
    sms_df = pd.DataFrame(sms)
    X = sms_df[0]
    X_tfidf = vectorizer.transform(X)

    prediction = model.predict(X_tfidf)

if prediction[0] == 0 :
    prediction = 'Not SPAM'

if prediction[0] == 1 :
    prediction = 'SPAM'

if prediction[0][0] :
    st.write("""
        ### Prediction
    """, '**', prediction, '**')
    