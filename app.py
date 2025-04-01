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
        results = sp.search(q=genre, type="track", limit=5)

        if results['tracks']['items']:
            for track in results['tracks']['items']:
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                track_url = track['external_urls']['spotify']

                # Display the song with a link
                st.markdown(f"🎶 [{track_name} - {artist_name}]({track_url})")

                # Get song recommendations
                track_id = track['id']
                recs = sp.recommendations(seed_tracks=[track_id], limit=3)
                rec_list = [f"🎵 {rec['name']} - {rec['artists'][0]['name']}" for rec in recs['tracks']]

                if rec_list:
                    with st.expander(f"Similar Songs to {track_name}"):
                        st.write("\n".join(rec_list))
        else:
            st.warning("No songs found for this emotion.")

        # Display playlist
        st.subheader("📻 Recommended Spotify Playlist")
        playlists = sp.search(q=f"{genre} playlist", type="playlist", limit=1)

        if playlists['playlists']['items']:
            playlist = playlists['playlists']['items'][0]
            playlist_name = playlist['name']
            playlist_url = playlist['external_urls']['spotify']
            st.markdown(f"📻 **[{playlist_name}]({playlist_url})**")
        else:
            st.warning("No playlists found for this emotion.")
    else:
        st.warning("Please enter some text!")
