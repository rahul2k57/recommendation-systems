import pandas as pd
filepath = './ml-1m/'

#Loading Movies
movies = pd.read_csv(filepath + 'movies.dat',sep='::',engine='python',
encoding='latin-1',names=['MovieID', 'Title', 'Genres'])

#Loading Ratings
ratings = pd.read_csv(filepath + 'ratings.dat',sep='::',
engine='python',names=['UserID','MovieID','Rating','Timestamp'])

#Loading Users
users = pd.read_csv(filepath + 'users.dat', sep='::',
engine='python',names=['UserID','Gender','Age','Occupation','Zip-code'])

print(f"Movies: {movies.shape}, Ratings: {ratings.shape}, Users: {users.shape}")
print("\n")

# View the first 5 movies
print(movies.head())
print("\n")

# View the first 5 ratings
print(ratings.head())
print("\n")

#View the first 5 users
print(users.head())


#Checking for the data sparsity
n_ratings = len(ratings)
n_movies = ratings['MovieID'].nunique()
n_users = ratings['UserID'].nunique()

sparsity = (1 - n_ratings / (n_users * n_movies)) * 100
print(f"Sparsity: {sparsity:.2f}%")

# Distribtuion of Movie Ratings
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
sns.countplot(x='Rating', data=ratings, palette='viridis')
plt.title('Distribution of Movie Ratings')
plt.show()

# Movie Popularity Distribution (The Long Tail - Popularity Bias)

# Counting how many times a movie appears in the ratings dataframe
movie_counts = ratings.groupby('MovieID').size().sort_values(ascending=False)

# Plotting the graph
plt.figure(figsize=(12,6))
plt.plot(movie_counts.values)
plt.fill_between(range(len(movie_counts)), movie_counts.values, color='#3498db',alpha=0.4)
plt.title('Movie Popularity Distribution',fontweight='bold',pad=15)
plt.xlabel('(Movies) Ranked from Most Popular to Least')
plt.ylabel('Number of Ratings received')
plt.grid(axis='y', linestyle='--',alpha=0.4)

plt.text(100, movie_counts.max(),'The Short Head (Blockbusters)',color='red',fontweight='bold')
plt.text(2000, 500, 'The "Long Tail" (Niche/Specific)', color='black', fontweight='bold')
plt.show()

# User Activity Levels

# 1. Counting no of ratings given by user
user_counts = ratings.groupby('UserID').size()

# 2. Plotting the distribution
plt.figure(figsize=(12, 6))
plt.hist(user_counts, bins=50, color='#e67e22', edgecolor='black', alpha=0.6)

plt.title('User Activity Levels', fontsize=15,fontweight='bold')
plt.xlabel('Number of Ratings given by a User', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.axvline(user_counts.mean(), color='red', linestyle='--',
label=f'Average: {user_counts.mean():.2f}')

plt.legend()
plt.show()

print(f"Minimum ratings by a user: {user_counts.min()}")
print(f"Average ratings by a user: {user_counts.mean():.2f}")

# Most Frequent Genres in the Dataset

# Splitting the genres and converting into lists
# Here Explode() creates a new row for every genre present in it, while keeping the same MovieID
genres_split = movies['Genres'].str.split('|').explode()

# 2. Plotting the horizontal bar chart
plt.figure(figsize=(12, 6))
sns.countplot(
    y=genres_split,
    order=genres_split.value_counts().index,
    hue=genres_split,
    legend=False,
    palette='magma'
)

plt.title('Most Frequent Genres in Dataset', fontsize=15,fontweight='bold')
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.show()

# 1. Calculating average rating and count for each movie
movie_stats = ratings.groupby('MovieID').agg({'Rating': ['mean', 'count']})
movie_stats.columns = ['Avg_Rating', 'Rating_Count']
# Plotting the scatterplot (Average Rating vs Number of Ratings)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rating_Count', y='Avg_Rating', data=movie_stats, alpha=0.4)

plt.title('Average Rating vs. Number of Ratings', fontsize=15,fontweight='bold')
plt.xlabel('Number of Ratings (Popularity)')
plt.ylabel('Average Rating (Quality)')
plt.axhline(ratings['Rating'].mean(), color='red', linestyle='--', label='Global Average')
plt.legend()
plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Intializing the vector
tfidf = TfidfVectorizer(stop_words='english')

# Constructing the tfidf matrix by fitting and transforming the genre data
tfidf_matrix = tfidf.fit_transform(movies['Genres'])
print(tfidf_matrix.shape)

# This will be creating 3883X3883 matrix and the values between 0 and 1
# Comparing the movies with other movies and their values is similarity
cosine_similarities = linear_kernel(tfidf_matrix,tfidf_matrix)

# Creating a mapping between movie titles and their row indices
indices = pd.Series(movies.index, index=movies['Title']).drop_duplicates()


# Recommendation Function - Content Based Filtering
def recommendation(title, cosine_similarities=cosine_similarities):

  # This will convert movie name into its row number
  movie_index = indices[title]

  # we will be checking similarity score of movie with every other movie
  # Here enumerate creates pairs for every score and then converts into lists
  similarity_scores = list(enumerate(cosine_similarities[movie_index]))

  # It will sort according to the similarity score (x[1])
  sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

  # stores the top 10 matches
  top_10_matches = sorted_scores[1:11]

  # After sorting, it removes similarity scores and just stores movieID
  recommended_ids = []
  for item in top_10_matches:
      movie_id = item[0]
      recommended_ids.append(movie_id)

  recommendations = movies['Title'].iloc[recommended_ids]
  return recommendations





recommendation('Toy Story (1995)')

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# 1. Prepare the User-Movie Matrix
# Rows = Users, Columns = Movies
user_movie_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating')

print(f"Original Matrix Shape: {user_movie_matrix.shape}")
# 2. Mean Centering (The "Secret" to a low RMSE)
# We calculate the average rating for each user
user_means = user_movie_matrix.mean(axis=1)

# We subtract the mean from the ratings.
# This makes '0' represent the user's average rating, not a missing value.
matrix_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

# 3. Apply Truncated SVD (Matrix Factorization)
# n_components=50 is standard for the MovieLens dataset
svd = TruncatedSVD(n_components=50, random_state=42)
user_features = svd.fit_transform(matrix_centered)
movie_features = svd.components_

# FIX 1: Calculated the correlation matrix so your recommendation functions don't crash
corr_matrix = np.corrcoef(movie_features.T) 

# FIX 2: matrix_low_rank was not defined; replaced with movie_features
print(f"Reduced Matrix Shape: {movie_features.shape}")

# 4. PREDICTING RATINGS & CALCULATING RMSE
# Multiply the User-Vibes by the Movie-Vibes
# Then add the user's mean back to get a 1-5 star prediction
predictions = np.dot(user_features, movie_features) + user_means.values.reshape(-1, 1)

# We only want to compare the ratings that actually EXIST in our data
mask = ~user_movie_matrix.isna()
actual_ratings = user_movie_matrix.values[mask]
predicted_ratings = predictions[mask]

# The Final Grade
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

print("-" * 30)
print(f"✅ COLLABORATIVE FILTERING COMPLETE")
print(f"Final Model RMSE: {rmse:.4f}")
print("-" * 30)

def get_collab_recommendations(movie_id, top_n=10):
    try:
        # Get the index of the movie in our matrix
        movie_idx = list(user_movie_matrix.columns).index(movie_id)

        # Find similarity scores
        similarities = corr_matrix[movie_idx]

        # Sort and pick the top matches
        similar_indices = np.argsort(similarities)[- (top_n + 1):-1][::-1]

        return user_movie_matrix.columns[similar_indices]
    except ValueError:
        return "Movie ID not found in ratings."

# Example: Get recommendations for Movie ID 1 (Toy Story)
print(get_collab_recommendations(1))

def get_final_recommendations(movie_id, top_n=10):
    # 1. Get the IDs from our SVD function
    recommended_ids = get_collab_recommendations(movie_id, top_n)

    # 2. Filter your movies dataframe to show the titles
    # Assuming your movies dataframe is 'df_movies'
    results = movies[movies['MovieID'].isin(recommended_ids)]

    return results[['Title', 'Genres']]

# Test it with a movie you know!
# Example: Toy Story (1)
print("Recommendations based on your taste:")
get_final_recommendations(1)

def get_hybrid_dashboard(movie_title, top_n=5):
    try:
        # 1. Find the Movie Info
        movie_row = movies[movies['Title'].str.contains(movie_title, case=False)].iloc[0]
        movie_id = movie_row['MovieID']
        actual_title = movie_row['Title']

        print(f"---  Hybrid Recommendations for: {actual_title} ---")
        print(f"Genre: {movie_row['Genres']}\n")

        # --- ENGINE A: COLLABORATIVE (SVD) ---
        # Logic: "People who liked this also liked..."
        movie_idx = list(user_movie_matrix.columns).index(movie_id)
        collab_similarities = corr_matrix[movie_idx]
        collab_indices = np.argsort(collab_similarities)[-(top_n + 1):-1][::-1]
        collab_ids = user_movie_matrix.columns[collab_indices]
        collab_recs = movies[movies['MovieID'].isin(collab_ids)][['Title', 'Genres']]

        # --- ENGINE B: CONTENT-BASED (GENRE) ---
        # Logic: "Movies with similar themes/genres..."
        # (Using the logic from your Phase 2)
        # Note: Replace 'cosine_sim' with whatever your Phase 2 similarity matrix was named
        idx = movies[movies['MovieID'] == movie_id].index[0]
        content_scores = list(enumerate(cosine_similarities[idx]))
        content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        content_indices = [i[0] for i in content_scores]
        content_recs = movies.iloc[content_indices][['Title', 'Genres']]

        #Returning the recommendations
        return collab_recs, content_recs

    except Exception as e:
        # Return the error as a string so app.py can catch it
        return f"Error: {str(e)}"

# Test the Final Hybrid Dashboard
get_hybrid_dashboard("Toy Story")


rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print(f" COLLABORATIVE FILTERING COMPLETE")
print(f"Final Model RMSE: {rmse:.4f}")