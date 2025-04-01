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
import streamlit as st
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API Credentials (Replace with your own)
SPOTIFY_CLIENT_ID = "90a9503c028c4868aa9423081e58e59b"
SPOTIFY_CLIENT_SECRET = "bc32eb23e6844c13b95e15c0ab24581d"

# Authenticate with Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# Load emotion classification model
model_name = "michellejieli/emotion_text_classifier"
classifier = pipeline("text-classification", model=model_name)

# Emotion-to-Genre Mapping
emotion_to_genre = {
    "anger": "rock",
    "disgust": "metal",
    "fear": "ambient",
    "joy": "happy",
    "neutral": "happy",  # Defaulting neutral to happy
    "sadness": "sad",
    "surprise": "pop"
}

# URL for the default Spotify logo image
spotify_logo_url = "https://storage.googleapis.com/pr-newsroom-wp/1/2023/05/Spotify_Primary_Logo_RGB_Green.png"

st.title("🎭 Emotion-Based Music Recommendation 🎵")
st.write("Enter a sentence, detect its emotion, and get music recommendations!")

# User input
user_input = st.text_area("Enter your text here:", "")

# Predict emotion
if st.button("Predict Emotion"):
    if user_input.strip():
        result = classifier(user_input)
        emotion = result[0]['label'].lower()
        score = result[0]['score']
        st.success(f"**Predicted Emotion:** {emotion.capitalize()} (Confidence: {score:.2f})")

        # Get recommended genre (default to happy)
        genre = emotion_to_genre.get(emotion, "happy")

        # Display songs
        st.subheader("🎵 Recommended Songs")
        results = sp.search(q=genre, type="track", limit=7)

        if results['tracks']['items']:
            for track in results['tracks']['items']:
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                track_url = track['external_urls']['spotify']

                # Check for album image URL
                if 'images' in track['album'] and track['album']['images']:
                    album_img_url = track['album']['images'][0]['url']
                else:
                    album_img_url = spotify_logo_url  # Use Spotify logo if no album image is found

                # Display the song with album image and a link
                st.image(album_img_url, width=100)  # Album image
                st.markdown(f"🎶 [{track_name} - {artist_name}]({track_url})")
                
        else:
            st.warning("No songs found for this emotion.")

        # Display playlist
        st.subheader("📻 Recommended Spotify Playlist")
        playlists = sp.search(q=f"{genre} playlist", type="playlist", limit=5)

        if playlists['playlists']['items']:
            playlist = playlists['playlists']['items'][0]
            playlist_name = playlist['name']
            playlist_url = playlist['external_urls']['spotify']
            playlist_img_url = playlist['images'][0]['url']  # Get playlist image
            st.markdown(f"📻 **[{playlist_name}]({playlist_url})**")
            st.image(playlist_img_url, width=500)  # Playlist image
        else:
            st.warning("No playlists found for this emotion.")
    else:
        st.warning("Please enter some text!")
