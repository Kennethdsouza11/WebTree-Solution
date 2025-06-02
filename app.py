import streamlit as st
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk


nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # converting to lowercase for better analysis
    text = text.lower()
    
    # handling repeated characters like goooood will change to good
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    # keeping imp punctuation and emojis as they might convey imp info which tells if the text is pos or not
    text = re.sub(r'[^a-zA-Z0-9\s.,!?$%&*()_+\-=\[\]{};\'"\\|,.<>\/?]', ' ', text)
    
    # handling negation
    negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor', 'cannot', 'didnt', 'doesnt', 
                     'dont', 'hadnt', 'hasnt', 'havent', 'isnt', 'wasnt', 'werent', 'wont', 'wouldnt'}
    
    # tokenizing
    words = text.split()
    
    # removing stopwords but keep negation words
    stop_words = set(stopwords.words('english')) - negation_words
    words = [word for word in words if word not in stop_words]
    
    # apply stemming, converting the words to its base root form
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)


st.set_page_config(
    page_title="Sentiment Analysis App",
    layout="centered"
)

# loading the trained model and comp
@st.cache_resource
def load_model():
    model = joblib.load('content_moderation_model.joblib')
    vectorizer = joblib.load('text_vectorizer.joblib')
    return model, vectorizer

# funct to get pred and conf score
def get_prediction(text, model, vectorizer):
    
    processed_text = preprocess_text(text)
    
    text_features = vectorizer.transform([processed_text])
    
    prediction = model.predict(text_features)[0]
    
    confidence_scores = model.decision_function(text_features)[0] #calculating conf scores
    
    confidence = 1 / (1 + np.exp(-confidence_scores)) #convert to prob-like score using sigmoid
    if prediction == 0:
        confidence = 1 - confidence
    
    return prediction, confidence


def main():
    st.title("Sentiment Analysis App")
    st.write("Enter your text below to analyze its sentiment:")
    
    # loading model
    model, vectorizer = load_model()
    
    # text inp
    user_input = st.text_area("Enter your text here:", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input:
            # get pred
            prediction, confidence = get_prediction(user_input, model, vectorizer)
            
            # displaying res
            st.write("---")
            st.subheader("Results:")
            
            # displaying sentiment
            if prediction == 1:
                st.write("Sentiment: Positive")
            else:
                st.write("Sentiment: Negative")
            
            # showing conf score with prog bar
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(confidence)
            
            # disp detailed exp
            st.write("---")
            st.subheader("Explanation:")
            if confidence > 0.8:
                st.write("The model is very confident in its prediction.")
            elif confidence > 0.6:
                st.write("The model is moderately confident in its prediction.")
            else:
                st.write("The model is less confident in its prediction.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 