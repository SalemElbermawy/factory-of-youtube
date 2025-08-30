import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("🧠 Classification of 'comment' ")

text = st.text_area("Open up to me and tell me what's on your mind.")



if st.button("Click here"):
    if text.strip():
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            mood = "Posetive Comment 😊✨"
            
        elif compound <= -0.05:
            mood = "Negative Comment 😔❓🔻"
            
        else:
            mood = "NEUTRAL 😐👍"
            

        st.success(f"State: {mood}")
        
