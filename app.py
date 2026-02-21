import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Movie Recommendatorzzz",
    layout="wide"
)

# =====================================
# SESSION STATE
# =====================================
if "selected_detail" not in st.session_state:
    st.session_state.selected_detail = None

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
# HEADER
# =====================================
st.title("🎬 Movie Recommendatorzzz")

# =====================================
# FORM
# =====================================
with st.form("form"):

    movie_options = [""] + df["title"].tolist()

    selected_movie = st.selectbox(
        "Pilih Film Favorit:",
        movie_options,
        index=0,
        format_func=lambda x: "Ketik judul film..." if x == "" else x
    )

    min_rating = st.number_input(
        "Minimal Rating",
        0.0, 10.0, 6.0
    )

    top_n = st.number_input(
        "Jumlah Rekomendasi",
        1, 20, 10
    )

    # =====================================
    # TOMBOL DI TENGAH
    # =====================================
    col1, col2, col3 = st.columns([2,1,2])

    with col2:
        recommend = st.form_submit_button("🎯 Cari")

# =====================================
# GRID POSTER
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

    st.subheader("Klik Poster")

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
                movie["title"],
                key=f"btn_{i}",
                use_container_width=True
            ):
                st.session_state.selected_detail = i

            st.image(poster, use_container_width=True)

# =====================================
# DETAIL
# =====================================
if st.session_state.selected_detail is not None:

    movie = df.iloc[st.session_state.selected_detail]

    st.divider()

    st.header("Detail Film")

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(movie["poster_url"], use_container_width=True)

    with col2:
        st.subheader(movie["title"])
        st.write("⭐ Rating:", movie["rating"])
        st.write("🎭 Genre:", movie["genre"])
        st.write("📺 Nonton di:", movie["streaming_provider"])
        st.write("📝 Deskripsi:")
        st.write(movie["description"])
