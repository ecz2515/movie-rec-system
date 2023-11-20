import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from api_utils import make_movie_api_call

def extract_movie_data(response_data):
    movie_id = response_data['id']
    title = response_data['titleText']['text']
    genres = [genre['text'] for genre in response_data['genres']['genres']]
    rating = response_data['ratingsSummary']['aggregateRating'] / 2
    year = response_data['releaseYear']['year']
    return movie_id, title, genres, rating, year

def create_movies_dataframe(movie_names, api_key):
    movies_data = {'IMDb_ID': [], 'Title': [], 'Genre': [], 'IMDb_Rating': [], 'Year': []}
    
    for movie in movie_names:
        response_data = make_movie_api_call(api_key, movie)
        movie_id, title, genres, rating, year = extract_movie_data(response_data)
        
        # Add data to the DataFrame
        movies_data['IMDb_ID'].append(movie_id)
        movies_data['Title'].append(title)
        movies_data['Genre'].append(genres)
        movies_data['IMDb_Rating'].append(rating)
        movies_data['Year'].append(year)

    print('\nProcessing done! Here is our dataframe:\n')
    movies_df = pd.DataFrame(movies_data)
    print(movies_df)
    return movies_df

def add_titanic_data(movies_df):
    titanic_data = {
        'IMDb_ID': 'tt0120338',
        'Title': 'Titanic',
        'Genre': ['Drama', 'Romance'],
        'IMDb_Rating': 7.9 / 2,  # IMDB ratings are out of 10
        'Year': 1997
    }
    movies_df.loc[24] = titanic_data

def encode_genres(movies_df):
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(movies_df['Genre'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
    encoded_movies_df = pd.concat([movies_df.drop('Genre', axis=1), genre_df], axis=1)
    return encoded_movies_df, mlb.classes_

def standardize_columns(movies_encoded):
    scaler = StandardScaler()
    movies_encoded[['Year', 'IMDb_Rating']] = scaler.fit_transform(movies_encoded[['Year', 'IMDb_Rating']])
    return movies_encoded

def load_user_ratings():
    user_ratings_data = {
        'Movie_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        'User_Rating': [4.833333333, 3, 4, 4, 4.666666667, 4.076923077, 3.714285714, 4.25, 2.166666667, 4.333333333,
                        3.75, 3.818181818, 2.333333333, 4.230769231, 4, 4.272727273, 3.75, 4.3, 3.9, 4.222222222,
                        3.857142857, 3, 3.666666667, 4, 3.4]
    }
    return pd.DataFrame(user_ratings_data)

def merge_user_ratings(movies_encoded, user_ratings):
    merged_data = pd.merge(user_ratings, movies_encoded, left_on='Movie_ID', right_index=True)
    return merged_data

def split_data_for_training(merged_data):
    X = merged_data.drop(['Movie_ID', 'IMDb_ID', 'Title', 'User_Rating'], axis=1)
    y = merged_data['User_Rating']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_knn_model(X_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn

def predict_ratings(knn, new_movies_features):
    return knn.predict(new_movies_features)[0]
