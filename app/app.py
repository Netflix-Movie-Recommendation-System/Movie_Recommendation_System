import streamlit as st
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process
import pickle

st.set_page_config(page_title="Netflix Recommender", page_icon="🍿", layout="wide")


# --------------------
# Load Data
# --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./data/Netflix Dataset Latest 2021.csv", encoding="latin1")

    drop_cols = [
        "Netflix Link",
        "IMDb Link",
        "Image",
        "Poster",
        "TMDb Trailer",
        "Trailer Site",
        "Awards Nominated For",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # df = df.dropna(subset=['Title'])
    return df


df = load_data()

# --------------------
# Try loading KMeans model
# --------------------

# def kmeans_scaler(movie_title, df, model, n=10):
#     """Dummy placeholder for compatibility with saved pickle model"""
#     return []


# try:
with open("./models/kmeans_scaler.pkl", "rb") as f:
    model = pickle.load(f)

# df = model["df"],
k_model = model["model"]
tfidf_vectorizer = model["tfidf_vectorizer"]
Scaler = model["Scaler"]
Genre_encoder = ["Genre_encoder"]
Language_encoder = model["Language_encoder"]
pca = model["PCA"]
st.success("✅ KMeans model loaded successfully.")
# except Exception as e:
#     model = None
#     st.warning(f"⚠️ Could not load KMeans model. Using TF-IDF only. Error: {e}")


# --------------------
# Custom Styling
# --------------------
st.markdown(
    """
    <style>
    .movie-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #1F1F1F;
        margin-bottom: 20px;
        transition: all 0.3s ease-in-out;
        border: 1px solid #333;
    }
    .movie-card:hover {
        transform: scale(1.03);
        background-color: #2A2A2A;
        border: 1px solid #E50914;
    }
    .movie-title {
        font-size: 22px;
        font-weight: bold;
        color: #E50914;
    }
    .movie-meta {
        font-size: 14px;
        color: #BBBBBB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍿 Netflix Movie Recommendation System")

# --------------------
# Pick correct text column for TF-IDF
# --------------------
text_features = []

# Combines multiple text-based columns into single strings
for idx, row in df.iterrows():
    combined_text = f"{row['Summary']} {row['Tags']} {row['Director']} {row['Actors']}"
    text_features.append(combined_text)

if text_features:

    @st.cache_resource
    def build_tfidf_model(text_features):
        tfidf_matrix = tfidf_vectorizer.transform(text_features)
        return tfidf_vectorizer, tfidf_matrix

    tfidf, tfidf_matrix = build_tfidf_model(text_features)
else:
    tfidf, tfidf_matrix = None, None


# --------------------
# Clean Dropdown Options
# --------------------
def split_unique_values(series):
    values = set()
    for entry in series.dropna():
        for v in str(entry).split(","):
            values.add(v.strip())
    return sorted(values)


# Detect correct column names dynamically
genre_col = next((c for c in df.columns if "genre" in c.lower()), None)
rating_col = next((c for c in df.columns if "rating" in c.lower()), None)
language_col = next((c for c in df.columns if "language" in c.lower()), None)

genre_list = split_unique_values(df[genre_col]) if genre_col else []
rating_list = split_unique_values(df[rating_col]) if rating_col else []
language_list = split_unique_values(df[language_col]) if language_col else []


# --------------------
# Fuzzy Search
# --------------------
def fuzzy_movie_search(query, choices, limit=5):
    results = process.extract(query, choices, limit=limit)
    return [r[0] for r in results]


# --------------------
# Filters
# --------------------
# --------------------
# Filters
# --------------------
# 🎥 Movie title search (searchable dropdown, same as "Did you mean")
selected_movie = st.selectbox(
    "🎥 Select a Movie:",
    options=sorted(df["Title"].dropna().unique()),
    help="Start typing to quickly find a movie",
)

# Other filters
selected_genres = st.multiselect("🎭 Genres (optional):", genre_list)
selected_ratings = st.multiselect("⭐ Ratings (optional):", rating_list)
selected_languages = st.multiselect("🗣️ Languages (optional):", language_list)

# 🔥 Number of recommendations
top_n = st.slider("📊 Number of recommendations:", min_value=1, max_value=20, value=5)

# --------------------
# Watchlist (Session State)
# --------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []


# --------------------
# Recommendation Logic
# --------------------
def get_movie_recommendations(selected_title, top_n=10):
    matches = df[df["Title"].str.lower() == selected_title.lower()]

    if matches.empty:
        return pd.DataFrame(), None

    selected_idx = matches.index[0]

    if tfidf_matrix is not None:
        cosine_similarities = cosine_similarity(
            tfidf_matrix[selected_idx], tfidf_matrix
        ).flatten()
        similar_indices = cosine_similarities.argsort()[::-1][1 : top_n + 1]
        recommendations = df.iloc[similar_indices].copy()
    else:
        recommendations = pd.DataFrame()

    if selected_genres and genre_col:
        recommendations = recommendations[
            recommendations[genre_col].apply(
                lambda x: any(g in str(x).split(",") for g in selected_genres)
            )
        ]
    if selected_ratings and rating_col:
        recommendations = recommendations[
            recommendations[rating_col].apply(
                lambda x: any(r in str(x).split(",") for r in selected_ratings)
            )
        ]
    if selected_languages and language_col:
        recommendations = recommendations[
            recommendations[language_col].apply(
                lambda x: any(l in str(x).split(",") for l in selected_languages)
            )
        ]

    return recommendations.head(top_n), df.loc[selected_idx]


# --------------------
# Show Results
# --------------------
if st.button("🔍 Recommend") and selected_movie:
    with st.spinner("🍿 Finding the best recommendations for you..."):
        recs, selected_movie_row = get_movie_recommendations(
            selected_movie, top_n=top_n
        )

    if selected_movie_row is None:
        st.warning("⚠️ No movie found with that title.")
    else:
        st.subheader(f"🎥 Selected Movie: {selected_movie_row['Title']}")
        if genre_col:
            st.write(f"🎭 **Genre:** {selected_movie_row.get(genre_col, 'N/A')}")
        if rating_col:
            st.write(f"⭐ **Rating:** {selected_movie_row.get(rating_col, 'N/A')}")
        if language_col:
            st.write(f"🗣️ **Language:** {selected_movie_row.get(language_col, 'N/A')}")
        if text_features and any(
            col in selected_movie_row.index for col in text_features
        ):
            st.write(f"📝 **{text_features}:** {selected_movie_row[text_features]}")

        if recs.empty:
            st.warning("⚠️ No similar movies found with given filters.")
        else:
            st.success("✅ Recommendations:")

            cols = st.columns(2)
            for i, (_, row) in enumerate(recs.iterrows(), start=1):
                with cols[(i - 1) % 2]:
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-title">{row['Title']}</div>
                            <div class="movie-meta">
                                🎭 {row.get(genre_col, 'N/A')}<br>
                                ⭐ {row.get(rating_col, 'N/A')}<br>
                                🗣️ {row.get(language_col, 'N/A')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.button(f"➕ Add to Watchlist", key=f"watch_{i}"):
                        st.session_state.watchlist.append(row["Title"])
                        st.success(f"✅ {row['Title']} added to watchlist!")

# --------------------
# Surprise Me Feature
# --------------------
if st.button("🎲 Surprise Me"):
    random_movie = random.choice(df["Title"].dropna().unique())
    st.info(f"Your random pick: **{random_movie}**")

# --------------------
# Trending Section
# --------------------
st.subheader("🔥 Trending Now")
trending = df["Title"].value_counts().head(5).index.tolist()
for t in trending:
    st.write(f"🎬 {t}")
