import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Movie Recommendatorzzz",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# SESSION STATE UNTUK DETAIL
# =====================================
if "selected_detail" not in st.session_state:
    st.session_state.selected_detail = None

# =====================================
# CSS NETFLIX STYLE
# =====================================
st.markdown("""
<style>

.main-container {
    max-width: 1200px;
    margin: auto;
}

.movie-card button {
    background: none;
    border: none;
    padding: 0;
}

.movie-card img {
    border-radius: 10px;
    transition: transform 0.2s;
}

.movie-card img:hover {
    transform: scale(1.05);
}

.detail-card {
    background-color: #0e1117;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown("""
<div class="main-container">
<h1 style="text-align:center;">🎬 Movie Recommendatorzzz</h1>
<p style="text-align:center;color:gray;">
Klik poster untuk melihat detail film
</p>
</div>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("movies.csv")

df["combined"] = (
    df["genre"].fillna("") +
    " " +
    df["description"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# =====================================
# FORM
# =====================================
with st.form("recommend_form"):

    movie_options = [""] + df["title"].tolist()

    selected_movie = st.selectbox(
        "Pilih Film Favorit:",
        options=movie_options,
        index=0,
        format_func=lambda x: "Ketik judul film..." if x == "" else x
    )

    min_rating = st.number_input(
        "Minimal Rating:",
        0.0, 10.0, 6.0
    )

    top_n = st.number_input(
        "Jumlah Rekomendasi:",
        1, 20, 10
    )

    recommend = st.form_submit_button("🎯 Cari Rekomendasi")

# =====================================
# RECOMMENDATION GRID
# =====================================
if recommend and selected_movie != "":

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

    st.markdown("## 🎞️ Rekomendasi")

    # responsive columns
    cols = st.columns(5)

    for idx, (i, score) in enumerate(recommended_movies):

        movie = df.iloc[i]

        if movie["rating"] < min_rating:
            continue

        poster = movie.get("poster_url", "")

        if pd.isna(poster):
            poster = "https://via.placeholder.com/300x450"

        col = cols[idx % 5]

        with col:

            if st.button(
                "",
                key=f"movie_{i}",
                use_container_width=True
            ):
                st.session_state.selected_detail = i

            st.image(poster, use_container_width=True)
            st.caption(movie["title"])

# =====================================
# DETAIL PANEL
# =====================================
if st.session_state.selected_detail is not None:

    movie = df.iloc[st.session_state.selected_detail]

    poster = movie.get("poster_url", "")

    st.markdown("## 🎬 Detail Film")

    col1, col2 = st.columns([1,2])

    with col1:

        if pd.notna(poster):
            st.image(poster, use_container_width=True)

    with col2:

        st.markdown(f"### {movie['title']} ({movie['year']})")

        st.write(f"⭐ Rating: {movie['rating']}")

        st.write(f"🎭 Genre: {movie['genre']}")

        st.write(f"📺 Nonton di: {movie['streaming_provider']}")

        st.write("### Deskripsi")
        st.write(movie["description"])
