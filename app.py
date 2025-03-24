import streamlit as st
from transformers import pipeline

# Load the model
model_name = "michellejieli/emotion_text_classifier"
classifier = pipeline("text-classification", model=model_name)

# Streamlit UI
st.title("🎭 Emotion Detection App")
st.write("Enter a sentence and get its predicted emotion!")

# Text input
user_input = st.text_area("Enter your text here:", "")

# Predict button
if st.button("Predict Emotion"):
    if user_input.strip():
        result = classifier(user_input)
        emotion = result[0]['label']
        score = result[0]['score']
        st.success(f"**Predicted Emotion:** {emotion} (Confidence: {score:.2f})")
    else:
        st.warning("Please enter some text!")

# Run the app with: streamlit run app.py
