# train model
import pandas as pd
import numpy as np
from datetime import datetime
from ast import literal_eval

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

def now_datetime():
    now = datetime.now()
    now_str = now.strftime("%m/%d/%Y, %H:%M:%S")
    return now_str

print("START Data Preparation: ", now_datetime())

spotify_artist_csv = pd.read_csv('data/artists.csv')
spotify_tracks_csv = pd.read_csv('data/tracks.csv')
kpop_gg_csv = pd.read_csv('data/kpop_gg.csv')
kpop_bg_csv = pd.read_csv('data/kpop_bg.csv')
kpop_idols_csv = pd.read_csv('data/kpop_all_idols.csv')
#kpop_mv_csv = pd.read_csv('data/kpop_music_videos.csv')

# Rename spotify data columns, convert to proper data types, and normalize string data types
spotify_artist = spotify_artist_csv.rename(columns = {'id': 'id_artists', 'name': 'artists', 'popularity': 'artist_popularity'})
# Filter only kpop genre artists
k_genres = ["'k-pop'", "'k-pop girl group'", "'korean ost'", "'korean city pop'", "'korean pop'", "'classic k-pop'", "'k-pop boy group'", "'korean r&b'", "'k-rock'", "'k-rap'"]  
k_genres_list = '|'.join(k_genres)
spotify_kpop_artist = spotify_artist.loc[spotify_artist['genres'].str.contains(k_genres_list, case=False)].reset_index(drop = True)
spotify_kpop_artist['artists'] = spotify_kpop_artist['artists'].str.lower()

# Rename column names of spotify_tracks
spotify_tracks = spotify_tracks_csv.rename(columns = {'id': 'id_tracks', 'name': 'tracks_name', 'popularity': 'tracks_popularity', 'mode': 'modality'})
# convert datetime column
spotify_tracks['release_date'] = pd.to_datetime(spotify_tracks['release_date'])
# explode artists column because there are songs performed by more than one artists, just get the most popular artist
spotify_tracks['id_artists'] = spotify_tracks['id_artists'].apply(literal_eval)
exploded_tracks = spotify_tracks.explode('id_artists').drop_duplicates()

kpop_groups_csv = pd.concat([kpop_gg_csv, kpop_bg_csv])
kpop_groups = kpop_groups_csv.rename(columns = {'Name': 'group', 'Debut': 'debut', 'Company': 'company', 'Members': 'members', 'Orig. Memb.': 'orig_members', 'Active': 'active'})

kpop_groups['members_lost'] = (kpop_groups['members'] - kpop_groups['orig_members'] < 0).astype(int)
# if the group is on hiatus change to active
kpop_groups['active'] = kpop_groups.loc[kpop_groups['active'] == 'hiatus', 'active'] = 'active'
kpop_groups['group'] = kpop_groups['group'].str.lower()
kpop_groups['debut'] = pd.to_datetime(kpop_groups['debut'])
kpop_groups = kpop_groups[['group', 'debut', 'company', 'members_lost', 'active']]

# check if group comes from big company
big5_companies = ['JYP', 'SM', 'YG', 'HYBE', 'Big Hit', 'Be:lift', 'Cube']
big5_list = '|'.join(big5_companies)
kpop_groups['big5_company'] = kpop_groups.company.str.contains(big5_list)

# Rename kpop_idols data columns, grouping to see average and std of age for each group
kpop_idols = kpop_idols_csv.rename(columns = {'Stage Name': 'stage_name', 'Group': 'group', 'Other Group': 'other_group', 'Date of Birth': 'birth_date', 'Gender': 'gender'})
kpop_idols['group'] = kpop_idols['group'].str.lower()
kpop_idols.loc[kpop_idols['group'] == "tvxq", 'group'] = "tvxq!"
kpop_idols.loc[kpop_idols['group'] == "cosmic girls", 'group'] = "wjsn"
kpop_idols['other_group'] = kpop_idols['other_group'].str.lower()
kpop_idols = (
 kpop_idols.assign(other_group=kpop_idols['other_group'].str.split(','))
   .explode('other_group')
   .reset_index(drop=True)
)
kpop_idols = kpop_idols[['stage_name', 'group', 'other_group', 'birth_date', 'gender']]
kpop_idols['birth_date'] = pd.to_datetime(kpop_idols['birth_date'])

max_date = spotify_tracks['release_date'].max()

def datediff(debut, release):
    diff_days = (release-debut).days
    return diff_days

def mean_group_age(kpop_idols, group, release_date):

    # calculate mean age in a group per max_date
    if (group in kpop_idols.group.tolist()):
        kpop_idols = kpop_idols[kpop_idols.group == group].reset_index(drop= True)
        kpop_idols['release_date'] = release_date
        age_group = []
        for i in range(len(kpop_idols)):
            birthdate = kpop_idols.loc[i, 'birth_date']
            today = kpop_idols.loc[i, 'release_date']
            age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
            age_group.append(age)
        #kpop_idols['age'] = kpop_idols['release_date'].dt.year - kpop_idols['birth_date'].dt.year - ((kpop_idols['release_date'].dt.month, kpop_idols['release_date'].dt.day) < (kpop_idols['birth_date'].dt.month, kpop_idols['birth_date'].dt.day))
    
    elif (group in kpop_idols.other_group.tolist()):
        kpop_idols = kpop_idols[kpop_idols.other_group == group].reset_index(drop= True)
        kpop_idols['release_date'] = release_date
        age_group = []
        for i in range(len(kpop_idols)):
            birthdate = kpop_idols.loc[i, 'birth_date']
            today = kpop_idols.loc[i, 'release_date']
            age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
            age_group.append(age)
    
    else:
        age_group = np.nan
    
    return np.mean(age_group)


# Get kpop groups that listed in Spotify music platform
spotify_kpop_artist_v2 = pd.merge(spotify_kpop_artist, kpop_groups, how = 'inner', right_on='group', left_on = 'artists')
spotify_kpop_tracks = pd.merge(exploded_tracks.drop('artists', axis = 1), spotify_kpop_artist_v2, how = 'inner', on = ['id_artists'])
spotify_kpop_tracks['days_after_debut'] = spotify_kpop_tracks.apply(lambda row: datediff(row['debut'], row['release_date']), axis=1)
spotify_kpop_tracks['days_after_debut'] = np.where(spotify_kpop_tracks['days_after_debut'] < 0, 0,spotify_kpop_tracks['days_after_debut'] )
spotify_kpop_tracks['mean_age_released'] = spotify_kpop_tracks.apply(lambda row: mean_group_age(kpop_idols, row['group'], row['release_date']), axis=1)

spotify_kpop_tracks_add = spotify_kpop_tracks.copy()
spotify_kpop_tracks_add['mean_age_released'] = spotify_kpop_tracks_add.apply(lambda row: mean_group_age(kpop_idols, row['group'], row['release_date']), axis=1)

spotify_kpop_df = spotify_kpop_tracks[['id_tracks', 'tracks_name', 'id_artists', 'artists', 'group',
        'tracks_popularity', 'artist_popularity', 'duration_ms', 'explicit',  'release_date', 'danceability', 'energy',
        'time_signature', 'loudness', 'modality', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key',
        'followers', 'debut', 'release_date', 'active', 'big5_company', 'days_after_debut', 'mean_age_released']]

# see each column data
# if using trees method, some columns like explicit

used_columns = ['tracks_popularity', 'artist_popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
        'time_signature', 'loudness', 'modality', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key',
        'followers', 'big5_company', 'days_after_debut', 'mean_age_released'] 

# Filter to columns that wanted to be analyzed
df = spotify_kpop_df[used_columns]

# Convert data to proper form and types
explicit_ls = {1: 'explicit', 0: 'non_explicit'}
df.explicit= df.explicit.map(explicit_ls)

pitch_class_ls = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
df.key = df.key.map(pitch_class_ls)

mode_ls = {0: 'minor', 1: 'major'}
df.modality = df.modality.map(mode_ls)

df[['time_signature', 'modality', 'key', 'big5_company']] =  df[['time_signature', 'modality', 'key', 'big5_company']].astype(str)

# Get numerical and categorical variable names
numerical_columns = list(df.dtypes[df.dtypes != "object"].index)
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)
base = numerical_columns + categorical_columns
numerical_columns.remove('tracks_popularity')
base.remove('tracks_popularity')

# Delete missing value
df = df.dropna(axis = 0)

print("FINISH Data Preparation: ", now_datetime())

# -------------------------------------------------------------------------------------------------------------------------------------
# DATA TRAINING -----------------------------------------------------------------------------------------------------------------------

# Split train, validation, and test

print("START Data Training: ", now_datetime())

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Separate y column or dependent variable
y_train = df_train.tracks_popularity.values
y_val = df_val.tracks_popularity.values
y_test = df_test.tracks_popularity.values

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.tracks_popularity.values
del df_full_train['tracks_popularity']

categorical_columns.remove('explicit')
categorical_columns.remove('time_signature')
base = numerical_columns + categorical_columns

features = base.copy()
full_train_dicts = df_full_train[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_full_train = dv.fit_transform(full_train_dicts)
dicts_test = df_test[features].to_dict(orient='records')
X_test = dv.transform(dicts_test)

rf = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_leaf=50, random_state=1)
rf.fit(X_full_train, y_full_train)

y_pred = rf.predict(X_test)
score = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE for test data: {}".format(score))

print("FINISH Data Training: ", now_datetime())

import bentoml
saved_model = bentoml.sklearn.save_model("spotify_regression", rf,
custom_objects = {"dictVectorizer": dv})

print(f"Model saved: {saved_model}")
