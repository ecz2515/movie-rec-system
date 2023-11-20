from flask import Flask, render_template, request
from api_utils import load_api_key, make_movie_api_call
from ml_utils import train_knn_model, predict_ratings, create_movies_dataframe, add_titanic_data, encode_genres, \
    standardize_columns, load_user_ratings, merge_user_ratings, split_data_for_training

import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

app = Flask(__name__)

# Initialize new_movie_names as an empty list
new_movie_names = []

# Initialize global variables to store ML-related results
knn_model, genre_classe = None, None
api_key = load_api_key()

def setup_ml_environment():
    global knn_model, genre_classes

    movie_names = ['The Shawshank Redemption', 'The Notebook', 'The Lord of the Rings: The Return of the King',
                    'Forrest Gump', 'Spider-Man: Into the Spider-Verse', 'Interstellar', 'Pride & Prejudice',
                    'Sen to Chihiro no kamikakushi', 'Alien', 'Parasite', 'The Princess Bride', 'Frozen',
                    'The Shining', 'Avengers: Endgame', 'Raiders of the Lost Ark', 'WALL·E', 'The Martian',
                    'La La Land', 'The Hunger Games', 'Legally Blonde', '21 Jump Street', 'Elf', 'Ghostbusters',
                    'Kimi no na wa.']

    # Load API key and create movies dataframe
    movies_df = create_movies_dataframe(movie_names, api_key)

    # Add Titanic data
    add_titanic_data(movies_df)

    # Encode genres
    movies_encoded, genre_classes = encode_genres(movies_df)

    # Standardize columns
    movies_encoded = standardize_columns(movies_encoded)

    # Load user ratings
    user_ratings = load_user_ratings()

    # Merge user ratings with movie data
    merged_data = merge_user_ratings(movies_encoded, user_ratings)

    # Split data for training
    X_train, X_test, y_train, y_test = split_data_for_training(merged_data)

    # Train the KNeighborsRegressor
    knn_model = train_knn_model(X_train, y_train)



# Call the setup function during application startup
setup_ml_environment()

@app.route('/', methods=['GET', 'POST'])
def index():
    global new_movie_names  # Use global to modify the outer variable

    api_key = load_api_key()

    if request.method == 'POST' and api_key is not None:
        # Handle user input when a new movie is submitted
        movie_name = request.form['movie_name']
        
        # Replace the existing movie in the list
        new_movie_names = [movie_name]

        # Predict ratings for new movie
        new_movies_data = {'IMDb_Rating': [], 'Year': []}
        for genre in genre_classes:
            new_movies_data[genre] = []

        for movie_name in new_movie_names:
            response_data = make_movie_api_call(api_key, movie_name)
            rating = response_data['ratingsSummary']['aggregateRating'] / 2
            year = response_data['releaseYear']['year']
            cur_genres = [genre['text'] for genre in response_data['genres']['genres']]
            image_url = response_data['primaryImage']['url']

            # Add data to the DataFrame
            new_movies_data['IMDb_Rating'].append(rating)
            new_movies_data['Year'].append(year)

            # Use list comprehension to generate 1s and 0s for each genre
            genre_flags = [1 if genre in cur_genres else 0 for genre in genre_classes]

            # Extend the values in new_movies_data with the genre_flags list
            for genre, flag in zip(genre_classes, genre_flags):
                new_movies_data[genre].append(flag)

        new_movies_features = pd.DataFrame(new_movies_data)

        # Standardize the 'Year' and 'IMDb_Rating' columns
        standardize_columns(new_movies_features)

        # Predict with KNN
        print(f"Movie names: {new_movie_names}")
        our_rating_predictions = predict_ratings(knn_model, new_movies_features)

        return render_template('index.html', new_movie_prediction=our_rating_predictions,
                                new_movie_rating=rating, new_movie_year=year,
                                new_movie_genres=cur_genres, new_movie_poster_url=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
