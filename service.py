import bentoml

from bentoml.io import JSON

from pydantic import BaseModel

class SpotifyTrackPopularity(BaseModel):
    artist_popularity: int
    duration_ms: int
    explicit: str
    danceability: float
    energy: float
    time_signature: str
    loudness: float
    modality: str
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    key: str
    followers: float
    big5_company: str
    days_after_debut: int
    mean_age_released: float

model_ref = bentoml.sklearn.get("spotify_regression:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("popularity_regressor", runners=[model_runner])

@svc.api(input=JSON(pydantic_model=SpotifyTrackPopularity), output=JSON()) # decorate endpoint as in json format for input and output
async def popularity_reg(track_profile): # parallelized requests at endpoint level (async)
    # transform pydantic class to dict to extract key-value pairs 
    profile = track_profile.dict()
    # transform data from client using dictvectorizer
    vector = dv.transform(profile)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = await model_runner.predict.async_run(vector) 
    popularity = int(prediction)
    # bentoml inference level parallelization (async_run)
    print(prediction)
    return { "Track Popularity": popularity }




