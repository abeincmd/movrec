import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Movie Recommendatorzzz",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================
# CSS FIX MOBILE + DROPDOWN STYLE
# =====================================
st.markdown("""
<style>

/* center layout */
.block-container {
    max-width: 520px;
    padding-top: 20px;
}

/* form box */
.form-box {
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px;
}

/* radio jadi dropdown style */
div[role="radiogroup"] {
    border: 1px solid #333;
    border-radius: 10px;
    padding: 8px;
    background-color: #1e1e1e;
    max-height: 200px;
    overflow-y: auto;
}

/* radio item */
div[role="radio"] {
    padding: 10px !important;
    border-radius: 8px;
}

/* hover */
div[role="radio"]:hover {
    background-color: #333;
}

/* button */
.stButton button {
    width: 100%;
    border-radius: 10px;
    padding: 12px;
}

/* image */
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

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# =====================================
# FORM
# =====================================
st.markdown('<div class="form-box">', unsafe_allow_html=True)

st.markdown("**Pilih Film Favorit:**")

selected_movie = st.radio(
    "",
    df["title"].tolist(),
    index=0
)

min_rating = st.number_input(
    "Minimal Rating:",
    0.0,
    10.0,
    6.0
)

top_n = st.number_input(
    "Jumlah Rekomendasi:",
    1,
    20,
    5
)

recommend = st.button("🎯 Cari Rekomendasi")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# RECOMMENDATION
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

            if pd.notna(poster):
                st.image(poster, use_container_width=True)

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
