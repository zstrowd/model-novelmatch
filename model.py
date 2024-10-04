import ast
import pandas as pd
import re
import logging
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def extracting_data(data):
    # using the re library to separate the rating and the amound of users rated
    match = re.match(r'(\d+(\.\d+)?)\s*\(\s*(\d+)\s*\)', data.strip())
    
    if match:
        rating = float(match.group(1))
        num_ratings = int(match.group(3))
        return rating, num_ratings
    else:
        logging.warning(f"Could not parse rating: {data}")  # Debugging message
        return None, None

def get_recommendations(title, combined_features, df, top_n=5):

    # Indexing the title to figure where in the df the user inputted title is
    try:
        idx = df.index[df['Title'] == title].tolist()[0]
    except IndexError:
        logging.error(f"Book title '{title}' not found in the dataset.")
        return []

    distances, indices = nn_model.kneighbors([combined_features.iloc[idx]], n_neighbors=top_n+1)

    similar_indices = indices.flatten()[1:]  # Skip the first as it's the same book
    similar_distances = distances.flatten()[1:] # Skip the first distance

    # Edge case: handle if there are less than top_n books
    if len(similar_indices) < top_n:
        logging.warning(f"Only found {len(similar_indices)} recommendations for '{title}'.")

    # Return the titles of the top N most similar books
    recommendations_with_scores = [
        {'title': df['Title'].iloc[i], 'score': 1 - similar_distances[idx]}  # 1 - distance gives similarity score
        for idx, i in enumerate(similar_indices)
    ]
    
    return recommendations_with_scores

def load_and_preprocess_data(file_path):

    # Loading the csv file
    try:
        df = pd.read_csv(file_path)
        logging.info('CSV file loading successful.')
    except FileNotFoundError:
        logging.error('Error: File not found. Please check File path or name.')
        raise

    # Adjust Genre column to be a list of lists instead of strings
    df['Genre'] = df['Genre'].apply(ast.literal_eval)

    # Extract rating values and number of ratings
    df['Rating_Value'], df['Num_Ratings'] = zip(*df['Rating'].apply(extracting_data))

    # Handle missing or invalid ratings by filling NaN values
    df['Rating_Value'].fillna(0, inplace=True)
    df['Num_Ratings'].fillna(0, inplace=True)

    return df

def prepare_features(df):
    # Preparing for genre rating features for model training
    genre = df['Genre'].tolist()

    # One-hot encode the genre column
    mlb = MultiLabelBinarizer()
    genre_mat = mlb.fit_transform(genre)
    genre_df = pd.DataFrame(genre_mat, columns=mlb.classes_)

    # Concatenate the one-hot encoded genre columns to the original dataframe
    df = pd.concat([df, genre_df], axis=1)

    # Normalize the Rating_Value column
    scaler = MinMaxScaler()
    df['Rating_Value_scaled'] = scaler.fit_transform(df[['Rating_Value']])

    # Combine genre and rating features for similarity calculations
    combined_features = pd.concat([genre_df, df[['Rating_Value_scaled']]], axis=1)

    return combined_features


# Preparing logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess the data
df = load_and_preprocess_data('novels.csv')

# Prepare combined features for recommendation
combined_features = prepare_features(df)

# Initialize the Nearest Neighbors model for cosine similarity
nn_model = NearestNeighbors(metric='cosine', algorithm='auto')
nn_model.fit(combined_features)

# Example: Get recommendations for a specific book title
book_title = "The Author's POV"
recommendations = get_recommendations(book_title, combined_features, df)

# Outputting the information
logging.info(f"Recommendations for '{book_title}':")
for rec in recommendations:
    logging.info(f"{rec['title']} - Similarity Score: {rec['score']:.2f}")