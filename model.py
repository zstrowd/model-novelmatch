import ast
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def extracting_data(data):
    # using the re library to separate the rating and the amound of users rated
    match = re.match(r'(\d+(\.\d+)?)\s*\(\s*(\d+)\s*\)', data.strip())
    
    if match:
        rating = float(match.group(1))
        num_ratings = int(match.group(3))
        return rating, num_ratings
    else:
        print(f"Warning: could not parse rating: {data}")  # Debugging message
        return None, None

def get_recommendations(title, cosine_sim, df, top_n=5):
    # Indexing the title to figure where in the df the user inputted title is
    idx = df.index[df['Title'] == title].tolist()[0]
    
    # Get similarity scores for all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sorting the books based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Getting the top 5 recommendations
    sim_scores = sim_scores[1:top_n+1]

    # Getting the titles of the books
    book_indices = [i[0] for i in sim_scores]

    #print(sim_scores)

    return df['Title'].iloc[book_indices]

df = pd.read_csv('novels.csv')

titles = df['Title'].tolist()
status = df['Status'].tolist()

# Adjusting data of Genre to be a list of lists instead of a string

df['Genre'] = df['Genre'].apply(ast.literal_eval)
genre = df['Genre'].tolist()

# Separating the count of users that rated the particular book and rating

df['Rating_Value'], df['Num_Ratings'] = zip(*df['Rating'].apply(extracting_data))
ratings = df['Rating_Value'].tolist()
num_ratings = df['Num_Ratings'].tolist()


# One-hot Encoding the genres list

mlb = MultiLabelBinarizer()
genre_mat = mlb.fit_transform(genre)
genre_df = pd.DataFrame(genre_mat, columns=mlb.classes_)

# Adding the new one-hot encoded genre list to original dataframe
df = pd.concat([df, genre_df], axis=1)

# Normalizing the ratings into the model
scaler = MinMaxScaler()
df['Rating_Value_scaled'] = scaler.fit_transform(df[['Rating_Value']])

# Combine the genre and rating features
combined_features = pd.concat([genre_df, df['Rating_Value_scaled']], axis=1)

# Calculate cosine similarity on the combined features
combined_cosine_sim = cosine_similarity(combined_features)
book = 'Lord of the Mysteries (Web Novel)'
recommendations = get_recommendations(book, combined_cosine_sim, df)


print(f'Based on the book you just read {book}. We recommend these 5 books:')
for i in recommendations:
    print(i)
    print('\n')
