import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Sistem rekomendasi film berdasarkan kemiripan genre dan deskripsi.")

# =============================
# LOAD DATASET
# =============================
df = pd.read_csv("movies.csv")

# =============================
# SIDEBAR FILTER
# =============================
st.sidebar.header("⚙️ Filter & Pengaturan")

min_rating = st.sidebar.slider("Minimal Rating", 0.0, 10.0, 7.0)
top_n = st.sidebar.slider("Jumlah Rekomendasi", 1, 10, 5)

filtered_df = df[df["rating"] >= min_rating].reset_index(drop=True)

# =============================
# TF-IDF + COSINE SIMILARITY
# =============================
filtered_df["combined"] = (
    filtered_df["genre"].fillna("") + " " +
    filtered_df["description"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(filtered_df["combined"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# =============================
# PILIH FILM
# =============================
movie_list = filtered_df["title"].tolist()

selected_movie = st.selectbox(
    "Pilih Film Favorit:",
    movie_list
)

# =============================
# TOMBOL REKOMENDASI
# =============================
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

    # =============================
    # NORMALISASI SCORE 1–10
    # =============================
    scores_only = [score for _, score in recommended_movies]

    max_score = max(scores_only)
    min_score = min(scores_only)

    results = []

    for i, score in recommended_movies:

        if max_score == min_score:
            scaled_score = 10
        else:
            scaled_score = (
                1 +
                ((score - min_score) /
                 (max_score - min_score)) * 9
            )

        movie = filtered_df.iloc[i]

        results.append({
            "No": len(results) + 1,
            "Title": movie["title"],
            "Year": movie["year"],
            "Genre": movie["genre"],
            "Rating": movie["rating"],
            "Nonton di": movie["streaming_provider"],
            "Similarity Score": round(scaled_score, 2)
        })

    results_df = pd.DataFrame(results)

    # =============================
    # TABEL HTML RAPI
    # =============================
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
    td:nth-child(7) {
        text-align: center;
    }
    td:nth-child(2),
    td:nth-child(4),
    td:nth-child(6) {
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        results_df.to_html(index=False),
        unsafe_allow_html=True
    )

    # =============================
    # POSTER CARD VIEW
    # =============================
    st.subheader("🎞️ Poster Rekomendasi")

    for i, score in recommended_movies:

        movie = filtered_df.iloc[i]

        col1, col2 = st.columns([1, 3])

        with col1:

            poster_url = movie.get("poster_url", "")

            if pd.notna(poster_url) and poster_url != "":
                st.image(
                    poster_url,
                    use_container_width=True
                )
            else:
                st.image(
                    "https://via.placeholder.com/300x450.png?text=No+Poster",
                    use_container_width=True
                )

        with col2:

            st.markdown(
                f"### 🎬 {movie['title']} ({movie['year']})"
            )

            st.write(f"**Genre:** {movie['genre']}")

            st.write(f"**Rating:** ⭐ {movie['rating']}")

            st.write(
                f"**Deskripsi:** {movie['description']}"
            )

            st.write(
                f"**Nonton di:** 📺 {movie['streaming_provider']}"
            )

            st.write("---")

# =============================
# LIHAT DATASET
# =============================
with st.expander("📂 Lihat Dataset Film"):

    st.dataframe(
        df,
        use_container_width=True
    )
