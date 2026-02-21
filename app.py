import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ==============================
# HEADER
# ==============================
st.title("🎬 Movie Recommendation System")
st.write(
    "Website ini memberikan rekomendasi film berdasarkan kemiripan genre, deskripsi, dan rating menggunakan AI."
)

st.divider()

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

df = load_data()

# ==============================
# FILTER DATA
# ==============================
movie_list = df["title"].tolist()

# ==============================
# LAYOUT FILTER (RAPI & RESPONSIVE)
# ==============================
with st.container(border=True):

    st.subheader("🎛️ Pengaturan Rekomendasi")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_movie = st.selectbox(
            "Pilih Film Favorit:",
            movie_list
        )

    with col2:
        min_rating = st.number_input(
            "Minimal Rating:",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    with col3:
        top_n = st.number_input(
            "Jumlah Rekomendasi:",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

    st.write("")

    search_button = st.button(
        "🎯 Cari Rekomendasi",
        use_container_width=True
    )

# ==============================
# FILTER BERDASARKAN RATING
# ==============================
filtered_df = df[df["rating"] >= min_rating].reset_index(drop=True)

# ==============================
# TF-IDF + COSINE SIMILARITY
# ==============================
filtered_df["combined"] = (
    filtered_df["genre"] + " " + filtered_df["description"]
)

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(filtered_df["combined"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# ==============================
# TMDB POSTER FUNCTION
# ==============================
def get_tmdb_poster(title):

    try:

        api_key = st.secrets["TMDB_API_KEY"]

        url = "https://api.themoviedb.org/3/search/movie"

        params = {
            "api_key": api_key,
            "query": title
        }

        response = requests.get(url, params=params)

        data = response.json()

        if data["results"]:

            poster_path = data["results"][0]["poster_path"]

            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

        return None

    except:
        return None


# ==============================
# HASIL REKOMENDASI
# ==============================
if search_button:

    movie_index = filtered_df[
        filtered_df["title"] == selected_movie
    ].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_movies = similarity_scores[1:top_n+1]

    # ==============================
    # NORMALISASI SCORE
    # ==============================
    scores_only = [score for _, score in recommended_movies]

    max_score = max(scores_only)
    min_score = min(scores_only)

    results = []

    for i, score in recommended_movies:

        if max_score == min_score:
            scaled_score = 10
        else:
            scaled_score = (
                1
                + ((score - min_score)
                   / (max_score - min_score)) * 9
            )

        results.append({
            "Title": filtered_df.iloc[i]["title"],
            "Year": filtered_df.iloc[i]["year"],
            "Genre": filtered_df.iloc[i]["genre"],
            "Rating": filtered_df.iloc[i]["rating"],
            "Similarity Score": round(scaled_score, 2)
        })

    results_df = pd.DataFrame(results)

    results_df.insert(
        0,
        "No",
        range(1, len(results_df) + 1)
    )

    st.divider()

    # ==============================
    # TABEL HASIL
    # ==============================
    st.subheader("📊 Tabel Rekomendasi")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==============================
    # POSTER VIEW
    # ==============================
    st.subheader("🎞️ Poster Rekomendasi")

    for i, score in recommended_movies:

        movie = filtered_df.iloc[i]

        poster = get_tmdb_poster(movie["title"])

        col1, col2 = st.columns([1, 3])

        with col1:

            if poster:

                st.image(
                    poster,
                    use_container_width=True
                )

            else:

                st.image(
                    "https://via.placeholder.com/300x450?text=No+Poster",
                    use_container_width=True
                )

        with col2:

            st.markdown(
                f"### 🎬 {movie['title']} ({movie['year']})"
            )

            st.write(
                f"**Genre:** {movie['genre']}"
            )

            st.write(
                f"**Rating:** ⭐ {movie['rating']}"
            )

            st.write(
                f"**Deskripsi:** {movie['description']}"
            )

        st.divider()


# ==============================
# DATASET VIEW
# ==============================
with st.expander("📂 Lihat Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )
