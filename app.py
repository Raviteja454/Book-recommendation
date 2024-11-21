import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data (the same as before)
df = pd.read_csv('your_books_data.csv')  # Replace with actual path to your dataset

# Select relevant columns
df = df[['BookID', 'Title', 'Author', 'Genre', 'CheckoutMonth', 'Number of Checkouts', 'ISBN', 'Format', 'Pages', 'Price', 'Rating', 'UserID']]

# Scaling and Clustering
scaler = StandardScaler()
df[['Price', 'Pages', 'Rating']] = scaler.fit_transform(df[['Price', 'Pages', 'Rating']])

# Example KMeans Clustering
kmeans = KMeans(n_clusters=5)
df['Cluster'] = kmeans.fit_predict(df[['Price', 'Pages', 'Rating']])

# Create a pivot table for user-item matrix
user_item_matrix = df.pivot_table(index='UserID', columns='Title', values='Rating').fillna(0)

# Compute cosine similarity between users
user_similarity_matrix = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Create a pivot table for book-item matrix
book_user_matrix = df.pivot_table(index='Title', columns='UserID', values='Rating').fillna(0)
book_similarity_matrix = cosine_similarity(book_user_matrix)
book_similarity_df = pd.DataFrame(book_similarity_matrix, index=book_user_matrix.index, columns=book_user_matrix.index)

# Recommendation Functions
def recommend_popular_by_ratings(top_n=10):
    popular_books = df.groupby('Title')['Rating'].mean().sort_values(ascending=False).head(top_n)
    return popular_books.reset_index()

def recommend_books(book_title, top_n=10):
    if book_title not in df['Title'].values:
        return "Book not found."
    
    cluster_label = df.loc[df['Title'] == book_title]['Cluster'].values[0]
    recommended_books = df[(df['Cluster'] == cluster_label) & (df['Title'] != book_title)]

    if recommended_books.empty:
        return "No similar books found in this cluster."
        
    recommended_titles = recommended_books[['Title','Author']].drop_duplicates().reset_index(drop=True)
    return recommended_titles[:top_n]

def recommend_books_collaborative_on_user(user_id, top_n=10):
    if user_id not in user_similarity_df.index:
        return ["User not found."]
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:top_n + 1]
    recommended_books_indices = user_item_matrix.loc[similar_users.index].mean(axis=0).sort_values(ascending=False).index.tolist()
    return recommended_books_indices[:top_n]

def recommend_books_collaborative_on_books(book_title, top_n=10):
    if book_title not in book_similarity_df.index:
        return ["Book not found."]
    
    similar_books = book_similarity_df[book_title].sort_values(ascending=False).iloc[1:top_n + 1]
    return similar_books.index.tolist()

# Streamlit UI
st.title("Book Recommendation System")

# Option for Popular Books
st.header("Popular Books by Ratings")
top_n_popular = st.slider('Select number of top popular books', 1, 20, 10)
if st.button('Show Popular Books'):
    popular_books = recommend_popular_by_ratings(top_n=top_n_popular)
    st.write(popular_books)

# Option for Book-based Recommendation
st.header("Recommend Books Based on a Book")
book_title = st.text_input("Enter the title of a book:")
top_n_cluster = st.slider('Select number of recommended books', 1, 10, 5)
if st.button('Recommend Similar Books by Cluster'):
    if book_title:
        recommended_books = recommend_books(book_title, top_n=top_n_cluster)
        st.write(recommended_books)
    else:
        st.write("Please enter a book title.")

# Option for User-based Collaborative Filtering
st.header("Recommend Books Based on User Similarity")
user_id = st.number_input("Enter User ID", min_value=1)
top_n_user = st.slider('Select number of recommended books for the user', 1, 10, 5)
if st.button('Recommend Books for User'):
    recommended_books_user = recommend_books_collaborative_on_user(user_id, top_n=top_n_user)
    st.write(recommended_books_user)

# Option for Book-based Collaborative Filtering
st.header("Recommend Books Based on a Book (Collaborative Filtering)")
book_title_cf = st.text_input("Enter book title for collaborative recommendation:")
top_n_book = st.slider('Select number of recommended books', 1, 10, 5)
if st.button('Recommend Similar Books by Collaborative Filtering'):
    if book_title_cf:
        recommended_books_cf = recommend_books_collaborative_on_books(book_title_cf, top_n=top_n_book)
        st.write(recommended_books_cf)
    else:
        st.write("Please enter a book title.")
streamlit run app.py
