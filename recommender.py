import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer



df = pd.read_csv('data/songs.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df = df.dropna()
df = df.reset_index(drop=True)
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

scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(df[features])

df['metadata'] = (
    df['artists'].astype(str) + ' ' +
    df['track_genre'].astype(str) + ' ' +
    df['album_name'].astype(str)
)

vectorizer = TfidfVectorizer(stop_words='english')

metadata_matrix = vectorizer.fit_transform(df['metadata'])

def recommend(song_name, artist_name=None, num_recommendations=5):

    matches = df[
        df['track_name'].str.lower().str.strip()
        ==
        song_name.lower().strip()
    ]

    if artist_name:

        matches = matches[
            matches['artists'].str.lower().str.contains(
                artist_name.lower().strip(),
                na=False
            )
        ]

    if len(matches) == 0:
        print("Song not found!")
        return

    song_index = matches.index[0]

    selected_song = df.iloc[song_index]

    print("\nSelected Song:")
    print(
        f"{selected_song['track_name']} "
        f"by {selected_song['artists']}"
    )



    song_vector = scaled_features[song_index].reshape(1, -1)

    audio_similarity = cosine_similarity(
        song_vector,
        scaled_features
    )[0]

    

    metadata_similarity = cosine_similarity(
        metadata_matrix[song_index],
        metadata_matrix
    )[0]

    

    final_similarity = (
        0.7 * audio_similarity +
        0.3 * metadata_similarity
    )

    sim_scores = list(enumerate(final_similarity))

    sorted_songs = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sorted_songs = [
        song for song in sorted_songs
        if song[0] != song_index
    ]

    print(f"\nBecause you liked '{song_name}':\n")

    

    seen_tracks = set()

    recommendations_printed = 0

    for song in sorted_songs:

        idx = song[0]
        score = song[1]

        track_name = df.iloc[idx]['track_name']
        artist = df.iloc[idx]['artists']

        # Skip duplicate tracks
        unique_key = (track_name, artist)

        if unique_key in seen_tracks:
            continue

        seen_tracks.add(unique_key)

        print(
            f"{track_name} "
            f"by {artist} "
            f"(score: {score:.3f})"
        )

        recommendations_printed += 1

        if recommendations_printed >= num_recommendations:
            break

# =========================
# TEST
# =========================

recommend("august", "Taylor Swift")