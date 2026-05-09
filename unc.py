import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/songs.csv').dropna()

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

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

scaled_features = MinMaxScaler().fit_transform(df[features])

df['metadata'] = (
    df['artists'].astype(str) + ' ' +
    df['track_genre'].astype(str) + ' ' +
    df['album_name'].astype(str)
)

metadata_matrix = TfidfVectorizer(
    stop_words='english'
).fit_transform(df['metadata'])

def recommend(song_name, artist_name=None, n=5):

    matches = df[
        df['track_name'].str.lower().str.strip()
        ==
        song_name.lower().strip()
    ]

    if artist_name:
        matches = matches[
            matches['artists'].str.lower().str.contains(
                artist_name.lower(),
                na=False
            )
        ]

    if len(matches) == 0:
        print("Song not found!")
        return

    idx = matches.index[0]

    audio_sim = cosine_similarity(
        scaled_features[idx].reshape(1, -1),
        scaled_features
    )[0]

    meta_sim = cosine_similarity(
        metadata_matrix[idx],
        metadata_matrix
    )[0]

    final_sim = 0.7 * audio_sim + 0.3 * meta_sim

    scores = sorted(
        list(enumerate(final_sim)),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nBecause you liked '{song_name}':\n")

    seen = set()
    count = 0

    for i, score in scores:

        if i == idx:
            continue

        track = df.iloc[i]['track_name']
        artist = df.iloc[i]['artists']

        if (track, artist) in seen:
            continue

        seen.add((track, artist))

        print(f"{track} by {artist} ({score:.3f})")

        count += 1

        if count >= n:
            break

recommend("august", "Taylor Swift")