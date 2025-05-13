"""Netflix Data Analysis"""

import pandas as pd
import matplotlib.pyplot as plt

#Loading the Data
df = pd.read_csv('merged_ratings.csv')
print(df[['title' ,'type']].head()) #Showing the Title and Type of First 5 movies

print(df.columns) #Showing the column titles

#Loading DataFrames for Movies and TV Shows
movies_df = df[df['type'] == 'Movie']
tv_shows_df = df[df['type'] == 'TV Show']
print(movies_df[['title', 'type', 'imdb_score']].head()) # Display the first 50 rows of the movies DataFrame
print(tv_shows_df[['title', 'type', 'imdb_score']].head()) # Display the first 50 rows of the TV shows DataFrame

import numpy as np
#Function to calculate the average of imdb scores
def imdb_score_distribution(shows_df):
  imdb_scores = shows_df['imdb_score'].head(100).values  # Convert the 'imdb_score' column to a NumPy array first 100 values
  imdb_scores_sum = np.sum(imdb_scores)  # Calculate the sum of IMDb scores
  imdb_scores_avg = np.mean(imdb_scores)  # Calculate the average of IMDb scores

  print("Average of IMDb scores:", round(imdb_scores_avg, 2))

imdb_score_distribution(movies_df) # To find average of imdb scores of Movies
imdb_score_distribution(tv_shows_df) # To find average of imdb scores of TV Shows

import matplotlib.pyplot as plt
import numpy as np

after_2015 = df[df['release_year'] >= 2015] #DataFrame to show releases after 2015

# Group the data by 'release_year' and 'type' and count the occurrences
grouped_data = after_2015.groupby(['release_year', 'type']).size().unstack(fill_value=0)

# Plot the clustered column chart
grouped_data.plot(kind='bar', figsize=(12, 6))
plt.xlabel('Release Year')
plt.ylabel('No. of Movies')
plt.title('Movies and TV Shows by Release Year')
plt.legend(title='Type')
plt.show()