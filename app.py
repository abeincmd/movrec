import streamlit as st
import pandas as pd
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
# CSS FIX SPACING + NETFLIX STYLE
# =====================================
st.markdown("""
<style>

/* hilangkan jarak atas */
.block-container {
    padding-top: 1rem;
}

/* center header */
.header {
    text-align:center;
}

/* scroll horizontal */
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
    transition: transform 0.25s ease;
}

.movie-card:hover img {
    transform: scale(1.08);
}

.poster-large img {
    height: 270px;
}

.poster-medium img {
    height: 200px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown("""
<div class="header">
<div style="font-size:40px;">🎬</div>
<div style="font-size:28px; font-weight:700;">
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

# gabungkan fitur untuk similarity
df["combined"] = (
    df["genre"].fillna("") + " " +
    df["description"].fillna("")
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# =====================================
# FORM
# =====================================
with st.form("form"):

    # ini kunci supaya dropdown tidak langsung tampil semua
    movie_options = [""] + sorted(df["title"].tolist())

    selected_movie = st.selectbox(
        "Pilih Film Favorit:",
        options=movie_options,
        index=0,
        format_func=lambda x: "Ketik judul film..." if x == "" else x
    )

    st.write("")

    # tombol center
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        recommend = st.form_submit_button(
            "🎯 Cari Rekomendasi",
            use_container_width=True
        )

# =====================================
# FUNCTION RENDER ROW
# =====================================
def render_row(title, movies, size):

    st.markdown(
        f"<div style='font-size:18px; font-weight:600; margin-top:20px; margin-bottom:10px;'>{title}</div>",
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
# STOP DI SINI (sesuai permintaan)
# =====================================
