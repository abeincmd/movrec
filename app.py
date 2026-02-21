import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Movie Recommendator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# CSS NETFLIX STYLE
# =====================================
st.markdown("""
<style>

.row-container {
    display: flex;
    overflow-x: auto;
    gap: 12px;
    padding-bottom: 10px;
}

.movie-card {
    flex: 0 0 auto;
    border-radius: 12px;
    overflow: hidden;
}

.movie-card img {
    border-radius: 12px;
    transition: transform 0.2s;
}

.movie-card img {
    border-radius: 12px;
    transition: transform 0.25s ease;
}

.movie-card img:hover {
    transform: scale(1.08);
}

.movie-card img:hover {
    transform: scale(1.08);
}

.poster-large img {
    height: 270px;
}

.poster-medium img {
    height: 200px;
}

/* scrollbar */
.row-container::-webkit-scrollbar {
    height: 6px;
}

.row-container::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 10px;
}

h1 {
    text-align: center;
}

/* center tombol dalam form */
div[data-testid="stForm"] div.stFormSubmitButton {
    display: flex;
    justify-content: center;
}

/* optional: buat tombol lebih bagus */
div.stFormSubmitButton button {
    padding: 10px 24px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown("""
<div style="text-align:center; margin-top:0px;">
    <div style="font-size:36px;">🎬</div>
    <div style="font-size:36px; font-weight:700;">
        Movie Recommendator
    </div>
    <div style="color:gray;">
        Temukan film terbaik berdasarkan favoritmu
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("movies.csv")

df["combined"] = df["genre"].fillna("") + " " + df["description"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# =====================================
# FORM
# =====================================
with st.form("form"):

    movie_options = [""] + df["title"].tolist()

    selected_movie = st.selectbox(
        "Pilih Film Favorit:",
        movie_options,
        index=0,
        format_func=lambda x: "" if x == "" else x
    )

    st.write("")  # spacer

    # trik center tombol
    col_left, col_center, col_right = st.columns([1,2,1])

    with col_center:
        recommend = st.form_submit_button(
            "🎯 Cari Rekomendasi",
            use_container_width=True
        )
    
        
# =====================================
# FUNCTION RENDER ROW (FIXED VERSION)
# =====================================
def render_row(title, movies, size):

    # GANTI INI (ubah ukuran judul)
    st.markdown(
        f"<h4 style='margin-bottom:10px;'>{title}</h4>",
        unsafe_allow_html=True
    )

    html = "<div class='row-container'>"

    for _, movie in movies.iterrows():

        poster = movie.get("poster_url", "")

        if pd.isna(poster) or poster == "":
            poster = "https://via.placeholder.com/300x450"

        html += (
            "<div class='movie-card poster-" + size + "'>"
            "<img src='" + poster + "'>"
            "</div>"
        )

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

# =====================================
# BARIS 1 — REKOMENDASI
# =====================================
if recommend and selected_movie != "":

    movie_index = df[df["title"] == selected_movie].index[0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_idx = [i for i, _ in similarity_scores[1:15]]

    recommended_movies = df.iloc[recommended_idx]

    render_row(
        "🎯 Rekomendasi Untuk Kamu",
        recommended_movies,
        "large"
    )

# =====================================
# BARIS 2 — RANDOM
# =====================================
random_movies = df.sample(min(len(df), 15))

render_row(
    "👍 Mungkin Kamu Suka",
    random_movies,
    "medium"
)

# =====================================
# BARIS 3 — TOP RATED
# =====================================
top_rated = df.sort_values(
    by="rating",
    ascending=False
).head(15)

render_row(
    "⭐ Rating Tertinggi",
    top_rated,
    "medium"
)
