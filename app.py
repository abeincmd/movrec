import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# PAGE CONFIG (HARUS PALING ATAS)
# =====================================
st.set_page_config(
    page_title="Movie Recommendatorzzz",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================
# CSS FIX MOBILE DROPDOWN
# =====================================
st.markdown("""
<style>

/* center container */
.block-container {
    max-width: 520px;
    padding-top: 20px;
}

/* fix dropdown supaya selalu ke bawah */
div[data-baseweb="popover"] {
    transform: none !important;
    top: 100% !important;
    bottom: auto !important;
}

/* perbaiki posisi select */
div[data-baseweb="select"] {
    position: relative !important;
}

/* dropdown scroll */
div[role="listbox"] {
    max-height: 300px !important;
    overflow-y: auto !important;
    z-index: 9999 !important;
}

/* form styling */
form {
    border: 1px solid #333;
    padding: 16px;
    border-radius: 12px;
}

/* button full width */
.stFormSubmitButton button {
    width: 100%;
    border-radius: 10px;
    padding: 12px;
    font-weight: bold;
}

/* poster */
img {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown("""
<h1 style="text-align:center;">🎬 Movie Recommendatorzzz</h1>
<p style="text-align:center;color:gray;">
Website ini memberikan rekomendasi film berdasarkan kemiripan genre dan rating
</p>
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

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# =====================================
# FORM INPUT (STABIL MOBILE)
# =====================================
with st.form("recommend_form"):

    st.markdown("### Pilih Film Favorit:")

    selected_movie = st.selectbox(
        "",
        df["title"].tolist(),
        index=0,
        label_visibility="collapsed"
    )

    min_rating = st.number_input(
        "Minimal Rating:",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.1
    )

    top_n = st.number_input(
        "Jumlah Rekomendasi:",
        min_value=1,
        max_value=20,
        value=5
    )

    recommend = st.form_submit_button("🎯 Cari Rekomendasi")

# =====================================
# RECOMMENDATION OUTPUT
# =====================================
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

        if movie["rating"] < min_rating:
            continue

        col1, col2 = st.columns([1,2])

        with col1:

            poster = movie.get("poster_url", "")

            if pd.notna(poster) and poster != "":
                st.image(poster, use_container_width=True)
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
