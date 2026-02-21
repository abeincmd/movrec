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

st.title("🎬 Movie Recommendation System")
st.write("Sistem rekomendasi film dengan AI + poster otomatis dari TMDB")

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("movies.csv")

# ==============================
# SIDEBAR FILTER
# ==============================
st.sidebar.header("⚙️ Pengaturan")

min_rating = st.sidebar.slider("Minimal Rating", 0.0, 10.0, 7.0)
top_n = st.sidebar.slider("Jumlah Rekomendasi", 1, 10, 5)

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
def get_tmdb_poster(movie_title):
    try:
        api_key = st.secrets["TMDB_API_KEY"]

        url = "https://api.themoviedb.org/3/search/movie"

        params = {
            "api_key": api_key,
            "query": movie_title
        }

        response = requests.get(url, params=params)

        data = response.json()

        if data["results"]:
            poster_path = data["results"][0]["poster_path"]

            if poster_path:
                full_url = "https://image.tmdb.org/t/p/w500" + poster_path
                return full_url

        return None

    except:
        return None


# ==============================
# PILIH FILM
# ==============================
movie_list = filtered_df["title"].tolist()

selected_movie = st.selectbox(
    "Pilih Film Favorit:",
    movie_list
)

# ==============================
# REKOMENDASI
# ==============================
if st.button("🎯 Cari Rekomendasi"):

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

    # ==========================
    # NORMALISASI SCORE 1–10
    # ==========================
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
                "Nonton di": filtered_df.iloc[i]["streaming_provider"],
                "Similarity Score": round(scaled_score, 2)
        })

    # ==========================
    # TABEL HASIL
    # ==========================
    results_df = pd.DataFrame(results)

    results_df.insert(
        0,
        "No",
        range(1, len(results_df) + 1)
    )

    results_df = results_df.reset_index(drop=True)

    st.subheader("📊 Tabel Rekomendasi")

    st.markdown("""
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        text-align: center !important;
        background-color: #111;
        color: white;
        padding: 10px;
    }
    td {
        padding: 10px;
    }
    td:nth-child(1),
    td:nth-child(3),
    td:nth-child(5),
    td:nth-child(6) {
        text-align: center;
    }
    td:nth-child(2),
    td:nth-child(4) {
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        results_df.to_html(index=False),
        unsafe_allow_html=True
    )

    # ==========================
    # POSTER VIEW
    # ==========================
    st.subheader("🎞️ Poster Rekomendasi")

    for i, score in recommended_movies:

        movie = filtered_df.iloc[i]

        poster_url = get_tmdb_poster(movie["title"])

        col1, col2 = st.columns([1, 3])

        with col1:

            if poster_url:
                st.image(
                    poster_url,
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
            st.write(
                f"**Nonton di:** 📺 {movie['streaming_provider']}"
            )

            st.write("---")


# ==============================
# DATASET VIEW
# ==============================
with st.expander("📂 Lihat Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )
