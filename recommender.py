import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/songs.csv')

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

df = df.dropna()
#df = df.sample(5000, random_state=42)
df = df.reset_index(drop=True)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

def recommend(song_name, num_recommendations=5):

    song_index = df[
    df['track_name'].str.lower().str.strip()
    ==
    song_name.lower().strip()
].index

    if len(song_index) == 0:
        print("Song not found!")
        return

    song_index = song_index[0]

    # Get vector for selected song
    song_vector = scaled_features[song_index].reshape(1, -1)

    # Compute similarity ONLY for this song
    sim_scores = list(
        enumerate(
            cosine_similarity(song_vector, scaled_features)[0]
        )
    )

    # Sort by similarity
    sorted_songs = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the same song
    sorted_songs = [
        song for song in sorted_songs
        if song[0] != song_index
    ]

    # Top recommendations
    sorted_songs = sorted_songs[:num_recommendations]

    print(f"\nBecause you liked '{song_name}':\n")

    for song in sorted_songs:
        print(df.iloc[song[0]]['track_name'])

# Test
recommend("august")