import pickle
import warnings
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# conda activate project
# streamlit run app/app.py


# Load the pre-trained model with caching
@st.cache_resource
def load_model():
    with open("models/kmeans_scaler.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


# Load data (for getting images)
@st.cache_resource
def load_data():
    old_data = pd.read_csv(
        "data/Netflix Dataset Latest 2021.csv",
        encoding="latin1",
    )
    images = old_data["Image"]
    return images


# Initialize the recommendation system with caching
@st.cache_resource
def initialize_recommender():
    loaded_model = load_model()
    images = load_data()

    df = loaded_model["df"]
    tfidf = loaded_model["tfidf_vectorizer"]
    scaler = loaded_model["Scaler"]
    genre_encoder = loaded_model["Genre_encoder"]
    language_encoder = loaded_model["Language_encoder"]
    pca = loaded_model["PCA"]
    k_model = loaded_model["model"]

    recommender = KMEANS_RECOMMENDATION_SYSTEM(
        df, tfidf, scaler, genre_encoder, language_encoder, pca, k_model
    )
    recommender.preprocess()
    return recommender, images, df


class KMEANS_RECOMMENDATION_SYSTEM:
    def __init__(
        self, df, tfidf, scaler, genre_encoder, language_encoder, pca, k_model
    ):
        self.df = df
        self.tfidf = tfidf
        self.scaler = scaler
        self.genre_encoder = genre_encoder
        self.language_encoder = language_encoder
        self.pca = pca
        self.k_model = k_model

    def preprocess(self):
        text_features = list(self.df["text_features"])

        tfidf_matrix = self.tfidf.transform(text_features)

        # 2. One-hot encode genres
        genre_matrix = self.genre_encoder.transform(self.df["Genre_List"])

        # 3. One-hot encode languages
        language_matrix = self.language_encoder.transform(self.df["Language_List"])

        numerical_features = [
            "Composite_Score",
            "Log_IMDb_Votes",
        ]
        numerical_matrix = self.scaler.transform(self.df[numerical_features].fillna(0))

        feature_matrix = hstack(  # shape (n_items, n_features)
            [
                tfidf_matrix,
                csr_matrix(genre_matrix),
                csr_matrix(language_matrix),
                csr_matrix(numerical_matrix),
            ]
        )

        # Apply PCA to reduce dimensions
        self.feature_matrix_reduced = self.pca.transform(
            feature_matrix.toarray()
        )  # If feature_matrix is sparse, convert to dense

        self.df["cluster"] = self.k_model.labels_

    def get_similar_movies_recommendations(self, title, num_recommendations=12):
        """
        Get top movie recommendations similar to the target title using KMeans clusters and cosine similarity.

        Parameters:
        - title: str, the title of the movie to find recommendations for
        - df: DataFrame, the dataset with movie details and cluster labels
        - feature_matrix: sparse/dense matrix, the feature matrix used for clustering (e.g., TF-IDF + one-hot + numerical)
        - num_recommendations: int, number of recommendations to return

        Returns:
        - DataFrame with top recommended movies
        """
        # Check if the title exists in the dataset
        if title not in self.df["Title"].values:
            raise ValueError(f"Title '{title}' not found in the dataset.")

        # Get the cluster and feature vector of the target movie
        target_movie = self.df[self.df["Title"] == title]
        movie_cluster = target_movie["cluster"].iloc[
            0
        ]  # Extract the cluster of the target movie.
        target_idx = target_movie.index[0]  # Get the index of the target movie
        target_features = self.feature_matrix_reduced[target_idx].reshape(
            1, -1
        )  # Reshape for cosine similarity (converts into a 2D array)

        # Get all movies in the same cluster
        cluster_movies = self.df[self.df["cluster"] == movie_cluster]

        # Get the indices of movies in the same cluster, excluding the target movie.
        cluster_indices = cluster_movies[cluster_movies["Title"] != title].index

        # Extract the feature vectors for all movies in the same cluster
        cluster_features = self.feature_matrix_reduced[cluster_indices]

        # Calculate cosine similarity between the target movie features and other movies features
        similarities = cosine_similarity(target_features, cluster_features)[0]

        # Create a DataFrame with similarities
        recommendations = cluster_movies.loc[cluster_indices].copy()
        recommendations["Similarity"] = similarities

        # Sort by similarity (descending) and select top recommendations
        recommendations = recommendations.sort_values(by="Similarity", ascending=False)

        # Return top recommendations
        return recommendations[
            [
                "Title",
                "Genre",
                "Series or Movie",
                "Director",
                "Composite_Score",
                "Actors",
                "Summary",
                "Similarity",
                "IMDb Score",
                "cluster",
            ]
        ].head(num_recommendations)

    def get_popular_recommendations(self, genre=None, content_type=None, top_n=10):
        """Get popular movies based on composite scores"""
        filtered_df = self.df.copy()

        # Checks if a genre filter is provided and it's not "All"
        if genre and genre != "All":
            filtered_df = filtered_df[
                filtered_df["Genre"].str.contains(genre, case=False, na=False)
            ]

        # Checks if a content type filter is provided and it's not "All"
        if content_type and content_type != "All":
            filtered_df = filtered_df[filtered_df["Series or Movie"] == content_type]

        # returns the rows with the largest values in "IMDb Score"
        popular_movies = filtered_df.nlargest(top_n, "IMDb Score")[
            [
                "Title",
                "Genre",
                "Director",
                "Actors",
                "Series or Movie",
                "Composite_Score",
                "IMDb Score",
                "Summary",
            ]
        ]

        return pd.DataFrame(popular_movies)

        # Searches for movies by looking for a query string in specified columns

    def search_movies(self, query, search_in=None):
        """
        Search for movies based on different criteria
        ("Title", "Genre", "Director", "Actors")
        """
        results = pd.DataFrame()

        for column in search_in:
            if column in self.df.columns:
                matches = self.df[
                    self.df[column].str.contains(
                        query,
                        case=False,  # Case-insensitive search
                        na=False,  # Ignore NaN/empty values
                    )
                ]
                results = pd.concat([results, matches])

        # Removes duplicate rows that have the same index (same movie appearing multiple times)
        if not results.empty:
            results = results.loc[~results.index.duplicated(keep="first")]

        results = results.nlargest(12, "IMDb Score")
        return results[
            [
                "Title",
                "Genre",
                "Director",
                "Actors",
                "IMDb Score",
                "Series or Movie",
                "Summary",
            ]
        ]

    def get_unique_genres(self):
        """Get list of unique genres for dropdown"""
        all_genres = []
        # add all elements from the current genre_list to the all_genres list
        # [extend() is used instead of append() because we want to add multiple items from a list, not add the list itself]
        for genre_list in self.df["Genre_List"]:
            all_genres.extend(genre_list)
        return sorted(list(set(all_genres)))  # Creates a set to remove duplicate genres


# Cache expensive operations
@st.cache_data
def get_similar_movies_cached(_recommender, title, num_recommendations=12):
    return _recommender.get_similar_movies_recommendations(title, num_recommendations)


@st.cache_data
def get_popular_movies_cached(_recommender, genre, content_type, top_n):
    return _recommender.get_popular_recommendations(genre, content_type, top_n)


@st.cache_data
def search_movies_cached(_recommender, query, search_in):
    return _recommender.search_movies(query, search_in=search_in)


@st.cache_data
def get_unique_genres_cached(_recommender):
    return _recommender.get_unique_genres()


# Streamlit UI
st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.markdown("Discover movies similar to your favorites or find popular titles by genre")

# Initialize with a spinner
with st.spinner("Loading recommendation system..."):
    recommender, images, df = initialize_recommender()

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an option:", ["Similar Movies", "Popular Movies", "Search Movies"]
)

# Similar Movies Section
if option == "Similar Movies":
    st.header("Find Similar Movies")

    movie_titles = sorted(recommender.df["Title"].unique())
    selected_movie = st.selectbox("Select a movie:", movie_titles)

    num_recommendations = st.slider("Number of recommendations:", 5, 20, 12)

    if st.button("Find Similar Movies"):
        with st.spinner("Finding similar movies..."):
            # Show details of the selected movie first
            selected_movie_row = recommender.df[
                recommender.df["Title"] == selected_movie
            ].iloc[0]

            # st.subheader(f"🎥 Selected Movie: {selected_movie_row['Title']}")
            st.markdown(
                f"<h3 style='color:red;'>🎥 Selected Movie: {selected_movie_row['Title']}</h3>",
                unsafe_allow_html=True,
            )
            st.write(f"**🎬 Genre:** {selected_movie_row['Genre']}")
            st.write(f"**🎥 Director:** {selected_movie_row['Director']}")
            st.write(f"**⭐ Actors:** {selected_movie_row['Actors']}")
            st.write(f"**📺 Type:** {selected_movie_row['Series or Movie']}")
            st.write(f"**🍿 IMDb Score:** {selected_movie_row['IMDb Score']}")
            with st.expander("📝 Summary"):
                st.write(selected_movie_row["Summary"])
            st.markdown("---")
            recommendations = get_similar_movies_cached(
                recommender, selected_movie, num_recommendations
            )

        if recommendations is not None and not recommendations.empty:
            st.success(f"Movies similar to '{selected_movie}':")

            # Display as cards in columns
            cols = st.columns(2)
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                with cols[idx % 2]:
                    with st.container():
                        image = images.loc[df["Title"] == row["Title"]].iloc[0]
                        # st.subheader(row["Title"])
                        st.markdown(
                            f"<h3 style='color:red;'>{row['Title']}</h3>",
                            unsafe_allow_html=True,
                        )
                        st.write(f"🎬 **Genre:** {row['Genre']}")
                        st.write(f"🎥 **Director:** {row['Director']}")
                        st.write(f"⭐ **Actors:** {row['Actors']}")
                        st.write(f"📺 **Type:** {row['Series or Movie']}")
                        # st.write(f"**Score:** {row['Composite_Score']:.1f}")
                        st.write(f"🍿 **IMDb Score:** {row['IMDb Score']}")
                        with st.expander("📝 Summary"):
                            st.write(row["Summary"])
                        st.image(image, caption=row["Title"], width=300)
                        st.markdown("---")
        else:
            st.warning("No similar movies found or movie not in database.")

# Popular Movies Section
elif option == "Popular Movies":
    st.header("Popular Movies by Genre")

    genres = ["All"] + get_unique_genres_cached(recommender)
    selected_genre = st.selectbox("Select genre:", genres)

    content_types = ["All", "Movie", "Series"]
    selected_type = st.selectbox("Select content type:", content_types)

    num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)

    if st.button("Get Popular Movies"):
        with st.spinner("Finding popular movies..."):
            popular_movies = get_popular_movies_cached(
                recommender,
                selected_genre if selected_genre != "All" else None,
                selected_type if selected_type != "All" else None,
                num_recommendations,
            )

        if not popular_movies.empty:
            st.success(
                f"Popular {selected_genre if selected_genre != 'All' else ''} {selected_type if selected_type != 'All' else 'content'}:"
            )

            # Display as cards in columns
            cols = st.columns(2)
            for idx, (_, row) in enumerate(popular_movies.iterrows()):
                with cols[idx % 2]:
                    with st.container():
                        image = images.loc[df["Title"] == row["Title"]].iloc[0]
                        # st.subheader(row["Title"])
                        st.markdown(
                            f"<h3 style='color:red;'>{row['Title']}</h3>",
                            unsafe_allow_html=True,
                        )
                        st.write(f"**🎬 Genre:** {row['Genre']}")
                        st.write(f"**🎥 Director:** {row['Director']}")
                        st.write(f"**⭐ Actors:** {row['Actors']}")
                        st.write(f"**📺 Type:** {row['Series or Movie']}")
                        # st.write(f"**Score:** {row['Composite_Score']:.1f}")
                        st.write(f"🍿 **IMDb Score:** {row['IMDb Score']}")
                        with st.expander("📝 Summary"):
                            st.write(row["Summary"])
                        st.image(image, caption=row["Title"], width=300)
                        st.markdown("---")
        else:
            st.warning("No movies found with the selected criteria.")

# Search Movies Section
elif option == "Search Movies":
    st.header("Search Movies")

    search_query = st.text_input("Enter search term:")
    search_columns = st.multiselect(
        "Search in:",
        ["Title", "Genre", "Director", "Actors"],
        # default=["Title", "Genre"],
    )

    if st.button("Search") and search_query:
        with st.spinner("Searching..."):
            results = search_movies_cached(
                recommender, search_query, search_in=search_columns
            )

        if not results.empty:
            st.success(f"Found {len(results)} results for '{search_query}':")

            # Display as cards in columns
            cols = st.columns(2)
            for idx, (_, row) in enumerate(results.iterrows()):
                with cols[idx % 2]:
                    with st.container():
                        image = images.loc[df["Title"] == row["Title"]].iloc[0]
                        # st.subheader(row["Title"])
                        st.markdown(
                            f"<h3 style='color:red;'>{row['Title']}</h3>",
                            unsafe_allow_html=True,
                        )
                        st.write(f"🎬 **Genre:** {row['Genre']}")
                        st.write(f"🎥 **Director:** {row['Director']}")
                        st.write(f"⭐ **Actors:** {row['Actors']}")
                        st.write(f"📺 **Type:** {row['Series or Movie']}")
                        st.write(f"🍿 **IMDb Score:** {row['IMDb Score']}")
                        with st.expander("📝 Summary"):
                            st.write(row["Summary"])
                        st.image(image, caption=row["Title"], width=300)
                        st.markdown("---")
        else:
            st.warning("No results found.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This recommendation system uses content-based filtering and K-means clustering "
    "to suggest movies based on their features and similarities."
)
