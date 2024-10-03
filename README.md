# Book Recommendation System

## Overview
This project is a **book recommendation system** that suggests similar books to users based on both **genre** and **rating**. It uses **content-based filtering**, where books are recommended based on their similarity to other books. This is achieved using a combination of one-hot encoded genres and normalized ratings, with **cosine similarity** to calculate the similarity between books.

## Features
- **Content-Based Filtering**: Recommendations are made based on the book's features (genre and rating).
- **Genre Encoding**: Each book's genre is transformed using **One-Hot Encoding**, allowing the system to understand genre similarities between books.
- **Rating Integration**: Books with similar ratings are more likely to be recommended alongside genre-based similarities.
- **Cosine Similarity**: The similarity between books is calculated using **cosine similarity**, which measures the angle between two feature vectors.

## How It Works
1. **Data Preprocessing**: The book data includes `Title`, `Genre`, `Rating`, and `Status`. Genres are provided as a list of genres for each book.
2. **One-Hot Encoding**: The genres are one-hot encoded into binary columns, allowing the system to calculate the similarity between books based on shared genres.
3. **Rating Normalization**: The book ratings are normalized using **MinMaxScaler**, ensuring that ratings and genres are on a similar scale.
4. **Cosine Similarity**: The system computes the cosine similarity between books based on their one-hot encoded genres and normalized ratings.
5. **Recommendation Function**: A recommendation function takes a book title as input and returns the top N most similar books based on cosine similarity.

## Dataset
The dataset is a CSV file with the following columns:
- **Title**: The title of the book.
- **Genre**: A list of genres that the book belongs to.
- **Rating**: A rating with the format `5.0 (255)` where `5.0` is the rating and `255` is the number of ratings.
- **Status**: (Optional) The status of the book (e.g., ongoing or completed).

## Requirements
The project uses the following Python libraries:
- `pandas`
- `scikit-learn`
- `numpy`
  
Install the required libraries with:
```bash
pip install pandas scikit-learn numpy
