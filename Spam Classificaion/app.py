
import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk


nltk.download('punkt')


with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


stemmer = PorterStemmer()

def preprocess(text):
    """
    Preprocess a single SMS message:
    - Lowercase
    - Remove non-alphanumeric characters
    - Tokenize
    - Apply stemming
    - Join tokens back to string
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    tokens = [stemmer.stem(w) for w in word_tokenize(text)]
    return ' '.join(tokens)


st.title("SMS Spam Classifier")
st.write("Enter a message to predict if it is Ham or Spam.")


message = st.text_input("Enter your SMS message:")

if st.button("Predict") and message:

    cleaned = preprocess(message)
    
    if not cleaned.strip():
        st.warning("Please enter a valid message!")
    else:
     
        vectorized = tfidf.transform([cleaned]).toarray()
        
      
        prediction = model.predict(vectorized)[0]
        
       
        result = "Spam" if prediction == 1 else "Ham"
        st.write("**Prediction:**", result)



# Optional: example messages
st.write("---")
st.write("Example messages you can try:")
st.write("1. Free entry in 2 a wkly comp to win FA Cup final tickets.")
st.write("2. Hey, are we meeting for lunch today?")
