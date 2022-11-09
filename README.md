## Predicting Track Popularity from K-pop Groups based on the Profile of Track and Group

Korean music has been acknowledged massively in the entire world. The training system for korean idols are still happening. Every year in Korea, many entertainment companies make debuts their produced k-pop groups. Spotify as the global streaming platform distributed k-pop songs for all over the world. 

I want to begin an analysis for predicting k-pop track's popularity using song/track's profile (musicality, duration, artists popularity, spotify followers) and k-pop group's profile (average age of members, days from debut, their company, active status). In this analysis, I used dataset from Spotify (source: Kaggle.com) for songs/tracks profile and https://dbkpop.com/ for K-pop groups profile scraped with BeautifulSoup. 

These are the scripts i used:
- `scrape_data.ipynb` : Script for scraping Kpop Database tables in kpdb.com 
- `analysis.ipynb` : Script for preparing dataset, doing EDA, regression analysis, and save the model


Data sources: 
- [Spotify Dataset 1921-2020, 600k+ Tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks)
- [Kpop Database](https://dbkpop.com/)