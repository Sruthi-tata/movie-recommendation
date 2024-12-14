import streamlit as st
import pandas as pd
import numpy as np
import difflib
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the trained models and vectorizer from pickle files
with open(r'C:\Users\Sruthi Tata\Downloads\rf_classifier.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

with open(r'C:\Users\Sruthi Tata\Downloads\tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load the dataset
movies_data = pd.read_csv(r"C:\Users\Sruthi Tata\Downloads\movies.csv")

# Preprocess the data (same steps as in the original code)
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

movies_data['combined_features'] = (
    movies_data['genres'] + ' ' + 
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' + 
    movies_data['cast'] + ' ' + 
    movies_data['director']
)

# Convert text data to feature vectors
X_text = tfidf_vectorizer.transform(movies_data['combined_features']).toarray()

# Add numerical features
movies_data['release_year'] = pd.to_datetime(
    movies_data['release_date'], errors='coerce'
).dt.year.fillna(0).astype(int)
movies_data['normalized_votes'] = movies_data['vote_count'] / movies_data['vote_count'].max()
movies_data['normalized_average'] = movies_data['vote_average'] / 10

scaler = StandardScaler()
X_numeric = scaler.fit_transform(movies_data[['release_year', 'normalized_votes', 'normalized_average']])

# Combine text and numeric features
X_combined = np.hstack((X_text, X_numeric))

# Ensure the 'liked' column exists (for classification)
if 'liked' not in movies_data.columns:
    movies_data['liked'] = (movies_data['vote_average'] >= 7.0).astype(int)

# Streamlit UI
st.title("Movie Recommendation System")

# Sidebar for model accuracy and classification report
st.sidebar.header("Model Accuracy and Classification Report")

# Display accuracy and classification report in the sidebar
if st.sidebar.button("Show Model Accuracy and Classification Report"):
    # Evaluate the model using the original data (this could be done using separate test data)
    y_pred = rf_classifier.predict(X_combined)
    accuracy = np.mean(y_pred == movies_data['liked'].values)
    st.sidebar.write(f"Model Accuracy: {accuracy:.2f}")
    
    # Display Classification Report
    report = classification_report(movies_data['liked'], y_pred)
    st.sidebar.text(report)

# User input for favorite movie in the center
movie_name = st.text_input("Enter your favorite movie name:")

# If movie name is provided, find recommendations
if movie_name:
    # Find the closest match
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        st.success(f"Close match found: **{close_match}**")

        # Find index and features of the movie
        index_of_movie = movies_data[movies_data.title == close_match].index[0]
        movie_features = X_combined[index_of_movie].reshape(1, -1)

        # Compute cosine similarity
        similarity_scores = cosine_similarity(movie_features, X_combined)
        movies_data['similarity'] = similarity_scores.flatten()

        # Get top recommendations based on similarity
        recommendations = movies_data.sort_values('similarity', ascending=False).head(10)

        st.write("### Movies Suggested for You:")
        for i, row in recommendations.iterrows():
            st.write(f"{i + 1}. **{row['title']}** - Similarity: {row['similarity']:.2f}")
    else:
        st.warning("No close match found.")

# Display the head of the dataset in the main area
st.write("### Dataset Preview:")
st.dataframe(movies_data.head())  # Display the first few rows as a table

