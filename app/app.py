import pickle
import warnings
import random
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# pipreqs . --force --encoding=utf-8
# # Try common encodings
# pipreqs . --force --encoding=latin-1
# pipreqs . --force --encoding=iso-8859-1
# pipreqs . --force --encoding=cp1252
# conda activate project
# streamlit run app/app.py


# Load the pre-trained model with caching
@st.cache_resource
def load_model():
    with open("models/kmeans_scaler.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


# Initialize the recommendation system with caching
@st.cache_resource
def initialize_recommender():
    loaded_model = load_model()

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
    return recommender


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
                "Languages",
                "Series or Movie",
                "Director",
                "Actors",
                "View Rating",
                "Summary",
                "IMDb Score",
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
                "Languages",
                "Director",
                "Actors",
                "View Rating",
                "Series or Movie",
                "IMDb Score",
                "Summary",
            ]
        ]

        return popular_movies

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
                "Languages",
                "Director",
                "Actors",
                "View Rating",
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

    def get_unique_languages(self):
        """Get list of unique genres for dropdown"""
        all_languages = []
        # add all elements from the current genre_list to the all_genres list
        # [extend() is used instead of append() because we want to add multiple items from a list, not add the list itself]
        for language_list in self.df["Language_List"]:
            all_languages.extend(language_list)
        return sorted(
            list(set(all_languages))
        )  # Creates a set to remove duplicate genres


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


@st.cache_data
def get_unique_languages_cached(_recommender):
    return _recommender.get_unique_languages()


# Streamlit UI
st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)

st.title("🍿 Netflix Movie Recommendation System")
# st.markdown("Discover movies similar to your favorites or find popular titles by genre")

# Initialize with a spinner
with st.spinner("Loading recommendation system..."):
    recommender = initialize_recommender()


# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an option:", ["Similar Movies", "Popular Movies", "Search Movies"]
)


# Similar Movies Section
if option == "Similar Movies":
    st.header("Find Similar Movies")

    movie_titles = sorted(recommender.df["Title"].unique())
    selected_movie = st.selectbox("**🎥 Select a movie:**", movie_titles)

    genre_list = get_unique_genres_cached(recommender)
    rating_list = recommender.df["View Rating"].unique().tolist()
    language_list = get_unique_languages_cached(recommender)

    selected_genres = st.multiselect("**🎭 Genres (optional)**:", genre_list)
    selected_ratings = st.multiselect("**⭐ Ratings (optional)**:", rating_list)
    selected_languages = st.multiselect("**🗣️ Languages (optional):**", language_list)

    all_selected = [selected_genres, selected_ratings, selected_languages]

    num_recommendations = st.slider("📊 Number of recommendations:", 5, 20, 12)

    if st.button("🔍 Recommend"):
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
            st.write(f"**🗣️ Language:** {selected_movie_row['Languages']}")
            st.write(f"**🎥 Director:** {selected_movie_row['Director']}")
            st.write(f"**⭐ Actors:** {selected_movie_row['Actors']}")
            st.write(f"**⭐ Rating:** {selected_movie_row['View Rating']}")
            st.write(f"**📺 Type:** {selected_movie_row['Series or Movie']}")
            st.write(f"**🍿 IMDb Score:** {selected_movie_row['IMDb Score']}")
            with st.expander("📝 Summary"):
                st.write(selected_movie_row["Summary"])
            st.markdown("---")
            recommendations = get_similar_movies_cached(
                recommender, selected_movie, num_recommendations
            )

            if selected_genres:
                recommendations = recommendations[
                    recommendations["Genre"].apply(
                        lambda x: any(
                            genre.strip() in [g.strip() for g in str(x).split(",")]
                            for genre in selected_genres
                        )
                    )
                ]

            if selected_ratings:
                recommendations = recommendations[
                    recommendations["View Rating"].apply(
                        lambda x: any(r in str(x).split(",") for r in selected_ratings)
                    )
                ]

            if selected_languages:
                recommendations = recommendations[
                    recommendations["Languages"].apply(
                        lambda x: any(
                            lang.strip() in [l.strip() for l in str(x).split(",")]
                            for lang in selected_languages
                        )
                    )
                ]

        if recommendations is not None and not recommendations.empty:
            st.success(f"Recommendations (similar to {selected_movie}):")

            # Display as cards in columns
            cols = st.columns(2)
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                with cols[idx % 2]:
                    with st.container():
                        # st.subheader(row["Title"])
                        st.markdown(
                            f"<h3 style='color:red;'>{row['Title']}</h3>",
                            unsafe_allow_html=True,
                        )
                        st.write(f"🎬 **Genre:** {row['Genre']}")
                        # language = row.get("Languages", "Unknown")
                        st.write(f"**🗣️ Language:** {row['Languages']}")
                        st.write(f"🎥 **Director:** {row['Director']}")
                        st.write(f"⭐ **Actors:** {row['Actors']}")
                        st.write(f"**⭐ Rating:** {row['View Rating']}")
                        st.write(f"📺 **Type:** {row['Series or Movie']}")
                        st.write(f"🍿 **IMDb Score:** {row['IMDb Score']}")
                        with st.expander("📝 Summary"):
                            st.write(row["Summary"])
                        st.markdown("---")
        else:
            st.warning("No similar movies found or movie not in database.")

    # --------------------
    # Surprise Me Feature
    # --------------------
    if st.button("🎲 Surprise Me"):
        random_movie = random.choice(recommender.df["Title"].dropna().unique())
        st.info(f"Your random pick: **{random_movie}**")

    # --------------------
    # Trending Section
    # --------------------
    st.subheader("🔥 Trending Now")
    trending = recommender.df["Title"].value_counts().head(5).index.tolist()
    for trend in trending:
        st.write(f"🎬 {trend}")

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
                        # st.subheader(row["Title"])
                        st.markdown(
                            f"<h3 style='color:red;'>{row['Title']}</h3>",
                            unsafe_allow_html=True,
                        )
                        st.write(f"**🎬 Genre:** {row['Genre']}")
                        # language = row.get("Languages", "Unknown")
                        st.write(f"**🗣️ Language:** {row['Languages']}")
                        st.write(f"**🎥 Director:** {row['Director']}")
                        st.write(f"**⭐ Actors:** {row['Actors']}")
                        st.write(f"**⭐ Rating:** {row['View Rating']}")
                        st.write(f"**📺 Type:** {row['Series or Movie']}")
                        st.write(f"🍿 **IMDb Score:** {row['IMDb Score']}")
                        with st.expander("📝 Summary"):
                            st.write(row["Summary"])
                        st.markdown("---")
        else:
            st.warning("No movies found with the selected criteria.")

# Search Movies Section
elif option == "Search Movies":
    st.header("Search Movies")

    search_query = st.text_input(
        "Enter search term:",
        help="Enter a query to search for(ie. Director or Actor name)",
    )
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
                        # st.subheader(row["Title"])
                        st.markdown(
                            f"<h3 style='color:red;'>{row['Title']}</h3>",
                            unsafe_allow_html=True,
                        )
                        st.write(f"🎬 **Genre:** {row['Genre']}")
                        # language = row.get("Languages", "Unknown")
                        st.write(f"**🗣️ Language:** {row['Languages']}")
                        st.write(f"🎥 **Director:** {row['Director']}")
                        st.write(f"⭐ **Actors:** {row['Actors']}")
                        st.write(f"**⭐ Rating:** {row['View Rating']}")
                        st.write(f"📺 **Type:** {row['Series or Movie']}")
                        st.write(f"🍿 **IMDb Score:** {row['IMDb Score']}")
                        with st.expander("📝 Summary"):
                            st.write(row["Summary"])
                        st.markdown("---")
        else:
            st.warning("No results found.")

# Footer
st.sidebar.markdown("---")
