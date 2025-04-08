# import streamlit as st
# from transformers import pipeline

# # Load the model
# model_name = "michellejieli/emotion_text_classifier"
# classifier = pipeline("text-classification", model=model_name)

# # Streamlit UI
# st.title("🎭 Emotion Detection App")
# st.write("Enter a sentence and get its predicted emotion!")

# # Text input
# user_input = st.text_area("Enter your text here:", "")

# # Predict button
# if st.button("Predict Emotion"):
#     if user_input.strip():
#         result = classifier(user_input)
#         emotion = result[0]['label']
#         score = result[0]['score']
#         st.success(f"**Predicted Emotion:** {emotion} (Confidence: {score:.2f})")
#     else:
#         st.warning("Please enter some text!")

# # Run the app with: streamlit run app.py
# import streamlit as st
# from transformers import pipeline
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

# # Spotify API Credentials (Replace with your own)
# SPOTIFY_CLIENT_ID = "90a9503c028c4868aa9423081e58e59b"
# SPOTIFY_CLIENT_SECRET = "bc32eb23e6844c13b95e15c0ab24581d"


# # Authenticate with Spotify API
# sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
#     client_id=SPOTIFY_CLIENT_ID,
#     client_secret=SPOTIFY_CLIENT_SECRET
# ))

# # Load emotion classification model
# model_name = "michellejieli/emotion_text_classifier"
# classifier = pipeline("text-classification", model=model_name)

# # Emotion-to-Genre Mapping
# emotion_to_genre = {
#     "anger": "rock",
#     "disgust": "metal",
#     "fear": "ambient",
#     "joy": "pop",
#     "neutral": "classical",  # Defaulting neutral to happy
#     "sadness": "sad",
#     "surprise": "rap"
# }

# # URL for the default Spotify logo image
# spotify_logo_url = "https://storage.googleapis.com/pr-newsroom-wp/1/2023/05/Spotify_Primary_Logo_RGB_Green.png"

# st.title("🎭 Emotion-Based Music Recommendation 🎵")
# st.write("Enter a sentence, detect its emotion, and get music recommendations!")

# # User input
# user_input = st.text_area("Enter your text here:", "")

# # Predict emotion
# if st.button("Predict Emotion"):
#     if user_input.strip():
#         result = classifier(user_input)
#         emotion = result[0]['label'].lower()
#         score = result[0]['score']
#         st.success(f"**Predicted Emotion:** {emotion.capitalize()} (Confidence: {score:.2f})")

#         # Get recommended genre (default to happy)
#         genre = emotion_to_genre.get(emotion, "classical")

#         # Display songs
#         st.subheader("🎵 Recommended Songs")
#         results = sp.search(q=genre, type="track", limit=7)

#         if results['tracks']['items']:
#             for track in results['tracks']['items']:
#                 track_name = track['name']
#                 artist_name = track['artists'][0]['name']
#                 track_url = track['external_urls']['spotify']

#                 # Check for album image URL
#                 if 'images' in track['album'] and track['album']['images']:
#                     album_img_url = track['album']['images'][0]['url']
#                 else:
#                     album_img_url = spotify_logo_url  # Use Spotify logo if no album image is found

#                 # Display the song with album image and a link
#                 st.image(album_img_url, width=100)  # Album image
#                 st.markdown(f"🎶 [{track_name} - {artist_name}]({track_url})")
                
#         else:
#             st.warning("No songs found for this emotion.")

#         # Display playlist
#         st.subheader("📻 Recommended Spotify Playlist")
#         playlists = sp.search(q=f"{genre} playlist", type="playlist", limit=5)

#         if playlists['playlists']['items']:
#             playlist = playlists['playlists']['items'][0]
#             playlist_name = playlist['name']
#             playlist_url = playlist['external_urls']['spotify']
#             playlist_img_url = playlist['images'][0]['url']  # Get playlist image
#             st.markdown(f"📻 **[{playlist_name}]({playlist_url})**")
#             st.image(playlist_img_url, width=500)  # Playlist image
#         else:
#             st.warning("No playlists found for this emotion.")
#     else:
#         st.warning("Please enter some text!")

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
import torch.nn.functional as F
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
SPOTIFY_CLIENT_ID = "90a9503c028c4868aa9423081e58e59b"
SPOTIFY_CLIENT_SECRET = "bc32eb23e6844c13b95e15c0ab24581d"

# Spotify Authentication (replace with your credentials)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="YOUR_SPOTIFY_CLIENT_ID",
    client_secret="YOUR_SPOTIFY_CLIENT_SECRET"
))

# Load text emotion classifier
text_model_name = "michellejieli/emotion_text_classifier"
classifier = pipeline("text-classification", model=text_model_name)

# Load image emotion model
image_model_name = "dima806/facial_emotions_image_detection"
image_processor = AutoImageProcessor.from_pretrained(image_model_name)
image_model = AutoModelForImageClassification.from_pretrained(image_model_name)

# Emotion-to-Genre Mapping
emotion_to_genre = {
    "anger": "rock",
    "disgust": "metal",
    "fear": "ambient",
    "joy": "pop",
    "neutral": "classical",
    "sadness": "sad",
    "surprise": "rap"
}

# Spotify fallback logo
spotify_logo_url = "https://storage.googleapis.com/pr-newsroom-wp/1/2023/05/Spotify_Primary_Logo_RGB_Green.png"

# Predict emotion from image
def predict_emotions(image):
    result = {"Image Emotion": "No Emotion Detected", "Confidence": 0.0, "Error": None}
    if image is not None:
        try:
            img = Image.fromarray(np.uint8(image)).convert("RGB")
            inputs = image_processor(img, return_tensors="pt")
            with torch.no_grad():
                outputs = image_model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_idx = logits.argmax(-1).item()
            emotion = image_model.config.id2label[predicted_class_idx].lower()
            confidence = probabilities[0, predicted_class_idx].item()
            result["Image Emotion"] = emotion
            result["Confidence"] = round(confidence * 100, 2)
        except Exception as e:
            result["Error"] = str(e)
    return result

# Streamlit App
st.title("🎭 Emotion-Based Music Recommendation 🎵")
st.write("Enter a sentence, and we'll detect both your **text and facial** emotions to recommend music!")

user_input = st.text_area("Enter your text here:", "")

if st.button("Predict Emotion"):
    if user_input.strip():
        # Text emotion
        text_result = classifier(user_input)[0]
        text_emotion = text_result["label"].lower()
        text_conf = round(text_result["score"] * 100, 2)
        st.success(f"📄 **Text Emotion:** {text_emotion.capitalize()} (Confidence: {text_conf}%)")

        # Capture image from webcam
        st.info("📸 Capturing image from webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam.")
        else:
            ret, frame = cap.read()
            cap.release()

            if ret:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image", use_column_width=True)

                # Image emotion
                img_result = predict_emotions(frame)
                if img_result["Error"]:
                    st.error(f"Image Emotion Detection Error: {img_result['Error']}")
                    image_emotion = "neutral"
                    image_conf = 0.0
                else:
                    image_emotion = img_result["Image Emotion"]
                    image_conf = img_result["Confidence"]
                    st.success(f"🖼️ **Image Emotion:** {image_emotion.capitalize()} (Confidence: {image_conf}%)")
            else:
                st.error("Failed to capture image.")
                image_emotion = "neutral"
                image_conf = 0.0

        # Final emotion decision
        if text_conf >= image_conf:
            final_emotion = text_emotion
            final_conf = text_conf
            source = "Text"
        else:
            final_emotion = image_emotion
            final_conf = image_conf
            source = "Image"

        st.markdown(f"🎯 **Final Emotion (based on {source}): {final_emotion.capitalize()} (Confidence: {final_conf}%)**")

        # Recommend songs
        genre = emotion_to_genre.get(final_emotion, "classical")
        st.subheader("🎵 Recommended Songs")

        results = sp.search(q=genre, type="track", limit=7)
        if results['tracks']['items']:
            for track in results['tracks']['items']:
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                track_url = track['external_urls']['spotify']
                album_img_url = (
                    track['album']['images'][0]['url']
                    if track['album']['images']
                    else spotify_logo_url
                )
                st.image(album_img_url, width=100)
                st.markdown(f"🎶 [{track_name} - {artist_name}]({track_url})")
        else:
            st.warning("No songs found for this emotion.")

        # Recommend playlist
        st.subheader("📻 Recommended Spotify Playlist")
        playlists = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        if playlists['playlists']['items']:
            playlist = playlists['playlists']['items'][0]
            st.markdown(f"📻 **[{playlist['name']}]({playlist['external_urls']['spotify']})**")
            st.image(playlist['images'][0]['url'], width=500)
        else:
            st.warning("No playlists found for this emotion.")
    else:
        st.warning("Please enter some text!")

