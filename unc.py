import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("data/songs.csv")

# Select useful audio features
features = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'valence',
    'tempo'
]

# Remove missing values
df = df.dropna()

# Normalize feature values
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

# Calculate similarity matrix
similarity = cosine_similarity(scaled_features)

# Recommendation function
def recommend(song_name, num_recommendations=5):

    # Find song index
    song_index = df[df['track_name'] == song_name].index

    if len(song_index) == 0:
        print("Song not found.")
        return

    song_index = song_index[0]

    # Get similarity scores
    similarity_scores = list(enumerate(similarity[song_index]))

    # Sort songs by similarity
    sorted_songs = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Skip first song (itself)
    sorted_songs = sorted_songs[1:num_recommendations+1]

    print(f"\nBecause you liked '{song_name}':\n")

    for i in sorted_songs:
        print(df.iloc[i[0]]['track_name'])

# Test
recommend("Blinding Lights")