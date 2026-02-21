import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====================================
# KONFIGURASI HALAMAN (MOBILE FIRST)
# ====================================
st.set_page_config(
    page_title="Movie Recommendatorzzz",
    page_icon="🎬",
    layout="centered"
)

# ====================================
# CSS MOBILE FRIENDLY
# ====================================
st.markdown("""
<style>

.block-container {
    max-width: 520px;
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
    font-weight: bold;
}

img {
    border-radius: 10px;
}

div[data-baseweb="select"] {
    z-index: 999 !important;
}

</style>
""", unsafe_allow_html=True)

# ====================================
# HEADER
# ====================================
st.markdown("""
<h1 style="text-align:center;">🎬 Movie Recommendatorzzz</h1>
<p style="text-align:center;color:gray;">
Website ini memberikan rekomendasi film berdasarkan kemiripan genre dan rating
</p>
""", unsafe_allow_html=True)

# ====================================
# LOAD DATA
# ====================================
df = pd.read_csv("movies.csv")

# gabungkan fitur
df["combined"] = (
    df["genre"].fillna("") +
    " " +
    df["description"].fillna("")
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# ====================================
# FORM INPUT (NO SIDEBAR)
# ====================================
with st.container(border=True):

    st.subheader("Pilih Film Favorit:")

# ====================================
# FORM MOBILE FIX (NO OVERLAP)
# ====================================
st.markdown("""
<style>

div[role="listbox"] {
    z-index: 9999 !important;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    padding: 12px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


with st.form("recommend_form"):

    st.markdown("**Pilih Film Favorit:**")

    selected_movie = st.selectbox(
        label="",
        options=df["title"].tolist(),
        label_visibility="collapsed"
    )

    min_rating = st.number_input(
        "Minimal Rating:",
        0.0, 10.0, 6.0
    )

    top_n = st.number_input(
        "Jumlah Rekomendasi:",
        1, 20, 5
    )

    recommend = st.form_submit_button("🎯 Cari Rekomendasi")

# ====================================
# REKOMENDASI
# ====================================
if recommend:

    movie_index = df[df["title"] == selected_movie].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_movies = similarity_scores[1:int(top_n)+1]

    st.markdown("### 🎞️ Poster Rekomendasi")

    no = 1

    for i, score in recommended_movies:

        movie = df.iloc[i]

        # filter rating di sini (fix)
        if movie["rating"] < min_rating:
            continue

        poster_url = movie.get("poster_url", "")

        col1, col2 = st.columns([1,2])

        with col1:

            if pd.notna(poster_url) and poster_url != "":
                st.image(poster_url, use_container_width=True)
            else:
                st.image(
                    "https://via.placeholder.com/300x450?text=No+Poster",
                    use_container_width=True
                )

        with col2:

            st.markdown(
                f"**{no}. {movie['title']} ({movie['year']})**"
            )

            st.write(f"Genre: {movie['genre']}")

            st.write(f"Rating: ⭐ {movie['rating']}")

            st.write(
                f"Nonton di : {movie['streaming_provider']}"
            )

            st.write(movie["description"])

        st.divider()

        no += 1
