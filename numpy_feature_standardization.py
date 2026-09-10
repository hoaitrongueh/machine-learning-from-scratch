import numpy as np

features = np.array([
    [180, 0.2],
    [240, 0.8],
    [300, 0.5],
])

column_means=np.mean(features,axis=0)
column_stds=np.std(features,axis=0)

scaled_features=(features-column_means)/column_stds

new_song = np.array([360, 0.65])

new_song_scaled=(new_song-column_means)/column_stds
print(new_song_scaled)