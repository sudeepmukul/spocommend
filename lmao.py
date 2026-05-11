# mood_camera_dj_v2.py

import cv2
import random
import numpy as np
import pandas as pd

from deepface import DeepFace
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("data/songs.csv").dropna()

# Remove weird duplicates
df = df.drop_duplicates(
    subset=['track_name', 'artists']
)

# OPTIONAL:
# Keep only reasonably popular songs
if 'popularity' in df.columns:
    df = df[df['popularity'] > 55]

df = df.reset_index(drop=True)

# ============================================
# AUDIO FEATURES
# ============================================

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

# ============================================
# SCALE FEATURES
# ============================================

scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(
    df[features]
)

# ============================================
# MOOD TARGET VECTORS
# ============================================

# Feature Order:
# danceability
# energy
# loudness
# speechiness
# acousticness
# instrumentalness
# valence
# tempo

MOOD_TARGETS = {

    "happy": np.array([
        0.9,
        0.85,
        0.7,
        0.3,
        0.2,
        0.1,
        0.95,
        0.8
    ]),

    "sad": np.array([
        0.25,
        0.2,
        0.4,
        0.45,
        0.9,
        0.4,
        0.2,
        0.3
    ]),

    "angry": np.array([
        0.75,
        1.0,
        0.9,
        0.4,
        0.1,
        0.1,
        0.25,
        0.95
    ]),

    "fear": np.array([
        0.2,
        0.3,
        0.4,
        0.5,
        0.8,
        0.6,
        0.15,
        0.25
    ]),

    "neutral": np.array([
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5
    ]),

    "surprise": np.array([
        0.8,
        0.9,
        0.7,
        0.3,
        0.2,
        0.2,
        0.85,
        0.9
    ])
}

# ============================================
# VIBE NAMES
# ============================================

VIBE_NAMES = {

    "happy": [
        "Golden Hour Chaos",
        "Sunset Victory Lap",
        "Smiling At Nothing",
        "Windows Down Energy"
    ],

    "sad": [
        "Existential Metro Ride",
        "2 AM Ceiling Staring",
        "Rain Against Neon Lights",
        "Emotionally Buffering"
    ],

    "angry": [
        "Villain Arc Cardio",
        "Locked-In Destruction",
        "Gym Demon Awakening",
        "Rage Against Reality"
    ],

    "fear": [
        "NPC With Hidden Lore",
        "Paranoia In 4K",
        "Unknown Signal Detected",
        "Dark Hallway Energy"
    ],

    "neutral": [
        "Late Night Coding Session",
        "Quiet Cosmic Drifting",
        "Mentally In Another Timeline",
        "Static But Cinematic"
    ],

    "surprise": [
        "Unexpected Side Quest",
        "Reality Just Glitched",
        "Plot Twist Energy",
        "Chaos But Beautiful"
    ]
}

# ============================================
# GENRE FILTERING
# ============================================

MOOD_GENRES = {

    "happy": [
        "pop",
        "dance",
        "party",
        "happy"
    ],

    "sad": [
        "sad",
        "acoustic",
        "indie",
        "piano"
    ],

    "angry": [
        "rock",
        "metal",
        "phonk",
        "punk"
    ],

    "fear": [
        "ambient",
        "electronic",
        "dark",
        "cinematic"
    ],

    "neutral": [
        "lofi",
        "chill",
        "study",
        "indie"
    ],

    "surprise": [
        "electronic",
        "experimental",
        "dance"
    ]
}

# ============================================
# CAMERA START
# ============================================

cap = cv2.VideoCapture(0)

print("\n🎥 Mood Camera DJ V2 Started")
print("Press Q to scan mood")
print("Press ESC to quit\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Mood Camera DJ", frame)

    key = cv2.waitKey(1)

    # ============================================
    # ANALYZE MOOD
    # ============================================

    if key == ord('q'):

        print("\n🧠 Reading emotional frequency...\n")

        try:

            analysis = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            mood = analysis[0]['dominant_emotion']

            print(f"🎭 Detected Mood: {mood.upper()}")

            # Fallback
            if mood not in MOOD_TARGETS:
                mood = "neutral"

            vibe = random.choice(
                VIBE_NAMES[mood]
            )

            print(f"🌌 Generated Vibe: {vibe}\n")

            # ============================================
            # FILTER BY GENRES
            # ============================================

            genre_keywords = MOOD_GENRES[mood]

            filtered_df = df[
                df['track_genre']
                .str.lower()
                .str.contains(
                    '|'.join(genre_keywords),
                    na=False
                )
            ]

            # fallback if too few songs
            if len(filtered_df) < 20:
                filtered_df = df.copy()

            filtered_indices = filtered_df.index

            filtered_features = scaled_features[
                filtered_indices
            ]

            # ============================================
            # TARGET VECTOR
            # ============================================

            target_vector = MOOD_TARGETS[mood]

            # ============================================
            # DISTANCE CALCULATION
            # ============================================

            distances = euclidean_distances(
                target_vector.reshape(1, -1),
                filtered_features
            )[0]

            scores = sorted(
                list(enumerate(distances)),
                key=lambda x: x[1]
            )

            # ============================================
            # SHOW RESULTS
            # ============================================

            print("🎵 Your Cinematic Playlist:\n")

            shown = 0
            seen = set()

            for idx, distance in scores:

                real_idx = filtered_indices[idx]

                track = df.iloc[real_idx]['track_name']
                artist = df.iloc[real_idx]['artists']
                genre = df.iloc[real_idx]['track_genre']

                if (track, artist) in seen:
                    continue

                seen.add((track, artist))

                print(
                    f"{shown+1}. "
                    f"{track} - {artist}"
                )

                print(
                    f"   Genre: {genre}"
                )

                print(
                    f"   Emotional Distance: "
                    f"{distance:.3f}\n"
                )

                shown += 1

                if shown >= 5:
                    break

            # ============================================
            # AI DJ COMMENTARY
            # ============================================

            dj_lines = [

                f"📻 AI DJ: Deploying soundtrack for your current life arc.",

                f"📻 AI DJ: Emotional synchronization complete.",

                f"📻 AI DJ: Tonight's atmosphere has been calibrated.",

                f"📻 AI DJ: You currently radiate '{vibe}' energy.",

                f"📻 AI DJ: This playlist feels illegally cinematic."
            ]

            print(random.choice(dj_lines))

        except Exception as e:

            print("\n❌ Error:", e)

        print("\nPress Q again for another scan.\n")

    # ============================================
    # EXIT
    # ============================================

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()