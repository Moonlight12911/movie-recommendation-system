"""
Streamlit interface for the movie recommendation system.

The deliberately thin part of the project. Its only jobs are to load the data,
draw widgets, call the engine, and display the result.

Run it with:  streamlit run app.py
"""

from __future__ import annotations

import html
import random
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import recommender
import tmdb_api

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# How long cached TMDB data stays fresh (1 hour).
CACHE_TTL_SECONDS = 3600

# Recommendation cards per row. Streamlit divides a row evenly, so without a
# cap of about this a 10-card request squeezes every poster to a sliver.
CARDS_PER_ROW = 5

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

.header-container {
    text-align: center;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
}

.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 40%, #d1d5db 80%, #9ca3af 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.sub-title {
    font-size: 0.95rem;
    color: #94a3b8;
}

.selected-movie-card {
    display: flex;
    gap: 20px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 20px;
    margin-top: 10px;
    margin-bottom: 14px;
    align-items: center;
}

.selected-movie-poster {
    width: 175px;
    min-width: 175px;
    aspect-ratio: 2/3;
    border-radius: 12px;
    object-fit: cover;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

/* Local stand-in for a missing poster, so no external service is involved. */
.poster-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: rgba(255, 255, 255, 0.05);
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
}

.selected-movie-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
    line-height: 1.25;
}

.selected-movie-desc {
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.55;
    margin-top: 8px;
    margin-bottom: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.movie-title-sm {
    font-size: 0.88rem;
    font-weight: 700;
    margin-top: 0.4rem;
    margin-bottom: 0.2rem;
    line-height: 1.25;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.25rem;
    color: #ffffff;
}

.badge-tag {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 3px;
    margin-bottom: 3px;
}

.badge-match {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    font-weight: 700;
}

.badge-rating {
    background: rgba(245, 158, 11, 0.16);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.badge-genre {
    background: rgba(139, 92, 246, 0.16);
    color: #c4b5fd;
    border: 1px solid rgba(139, 92, 246, 0.35);
}

.badge-info {
    background: rgba(255, 255, 255, 0.06);
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

div[data-testid="stImage"] img {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    aspect-ratio: 2/3;
    object-fit: cover;
    max-height: 220px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
    padding: 10px !important;
    background: rgba(18, 24, 38, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(229, 9, 20, 0.45);
    box-shadow: 0 8px 24px -4px rgba(229, 9, 20, 0.15);
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e50914 0%, #ff2e43 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(229, 9, 20, 0.35) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.5) !important;
}

div.stButton > button[kind="secondary"] {
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.08) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data loading
# --------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Loading movie catalogue...")
def load_movies(page_count: int = tmdb_api.DEFAULT_PAGE_COUNT) -> pd.DataFrame:
    """Fetch and clean the movie catalogue, preferring the local disk cache."""
    genre_map = tmdb_api.load_genre_map()
    raw_movies = tmdb_api.load_catalogue(page_count=page_count)
    return recommender.build_movie_dataframe(raw_movies, genre_map)


@st.cache_data(show_spinner=False)
def load_similarity_matrix(
    genre_texts: tuple[str, ...],
    overview_texts: tuple[str, ...],
    genre_weight: float,
) -> np.ndarray:
    """Build the blended genre + plot similarity matrix for the catalogue."""
    return recommender.build_similarity_matrix(
        list(genre_texts), list(overview_texts), genre_weight
    )


@st.cache_data(show_spinner=False)
def load_vocabularies(
    genre_texts: tuple[str, ...], overview_texts: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """Return the genre tokens and plot terms the vectorizers learned."""
    return (
        recommender.build_genre_vocabulary(list(genre_texts)),
        recommender.build_overview_vocabulary(list(overview_texts)),
    )


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------


def format_rating(rating: object) -> str:
    """Format a TMDB rating for display."""
    if rating is None or pd.isna(rating):
        return recommender.UNKNOWN_VALUE_LABEL
    return f"{float(rating):.1f}/10"


def format_release_date(release_date: object) -> str:
    """Format a release date for display."""
    if not isinstance(release_date, str) or not release_date:
        return recommender.UNKNOWN_VALUE_LABEL
    return release_date


def build_selection_label(title: str, release_year: str) -> str:
    """Build the dropdown text, e.g. 'Inception (2010)'."""
    return f"{title} ({release_year})"


def safe(value: object) -> str:
    """
    Escape a value for interpolation into a raw-HTML block.

    TMDB titles and plot summaries are user-contributed, and every st.markdown
    call below runs with unsafe_allow_html=True. Without escaping, a title
    holding an ampersand or an angle bracket would corrupt the surrounding
    markup - and could inject it.
    """
    return html.escape(str(value), quote=True)


def poster_html(poster_path: object, title: str) -> str:
    """Return an <img> for the selected-movie card, or a local placeholder."""
    poster_url = tmdb_api.build_poster_url(poster_path)
    if poster_url:
        return (
            f'<img src="{safe(poster_url)}" class="selected-movie-poster" '
            f'alt="{safe(title)}" />'
        )
    return (
        '<div class="selected-movie-poster poster-placeholder">No poster<br/>available</div>'
    )


def render_movie_card(movie: dict[str, Any]) -> None:
    """Draw one compact recommended movie card."""
    with st.container(border=True):
        poster_url = tmdb_api.build_poster_url(movie.get("poster_path"))
        if poster_url:
            st.image(poster_url, width="stretch")
        else:
            st.caption("No poster")

        match_pct = int(round(movie["similarity_score"] * 100))
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; margin-top:4px;">'
            f'<span class="badge-tag badge-match">🔥 {match_pct}%</span>'
            f'<span class="badge-tag badge-rating">⭐ {format_rating(movie["rating"])}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="movie-title-sm">{safe(movie["title"])} '
            f'({safe(movie["release_year"])})</div>',
            unsafe_allow_html=True,
        )

        genres_preview = " ".join(
            f'<span class="badge-tag badge-genre">{safe(g)}</span>'
            for g in movie["genres"][:2]
        )
        st.markdown(genres_preview, unsafe_allow_html=True)

        overview = movie.get("overview")
        if overview:
            with st.expander("📖 Synopsis", expanded=False):
                st.caption(overview)

        if st.button("🔍 Similar", key=f"rec_btn_{movie['movie_id']}", width="stretch"):
            st.session_state["selected_title"] = movie["title"]
            st.session_state["requested_title"] = movie["title"]
            st.rerun()


# --------------------------------------------------------------------------
# Header and sidebar
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="header-container">
        <div class="main-title">🎬 Movie Recommendation System</div>
        <div class="sub-title">Discover similar films using <b>Content-Based Filtering</b> & <b>Cosine Similarity</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=recommender.DEFAULT_RECOMMENDATION_COUNT,
        help="How many similar movies to recommend.",
    )
    genre_weight = st.slider(
        "Genre vs. plot balance",
        min_value=0.0,
        max_value=1.0,
        value=recommender.DEFAULT_GENRE_WEIGHT,
        step=0.1,
        help=(
            "1.0 uses genres only, which makes most films tie at 100%. "
            "Lower values mix in plot-summary similarity to break those ties."
        ),
    )
    if st.button("🔄 Reload catalogue from TMDB", width="stretch"):
        st.cache_data.clear()
        st.session_state.pop("requested_title", None)
        st.session_state.pop("selected_title", None)
        st.session_state["force_refresh"] = True
        st.rerun()

# --- Load data ------------------------------------------------------------
try:
    if st.session_state.pop("force_refresh", False):
        with st.spinner("Re-syncing catalogue from TMDB..."):
            tmdb_api.load_genre_map(force_refresh=True)
            tmdb_api.load_catalogue(
                page_count=tmdb_api.DEFAULT_PAGE_COUNT, force_refresh=True
            )
    movies = load_movies(tmdb_api.DEFAULT_PAGE_COUNT)
except tmdb_api.TMDBError as error:
    st.error(str(error))
    st.stop()

if movies.empty:
    st.error("TMDB responded, but no movies were found. Try reloading.")
    st.stop()

genre_texts = tuple(movies["genre_text"])
overview_texts = tuple(movies["overview"])
similarity_matrix = load_similarity_matrix(genre_texts, overview_texts, genre_weight)
genre_vocabulary, overview_vocabulary = load_vocabularies(genre_texts, overview_texts)

with st.sidebar:
    st.markdown("---")
    st.markdown("#### ℹ️ System Status")
    st.caption(f"● {len(movies):,}-movie catalogue loaded.")
    st.caption(f"● {len(genre_vocabulary)} genre features (CountVectorizer).")
    st.caption(f"● {len(overview_vocabulary):,} plot features (TF-IDF).")
    st.caption(f"● Blend: {genre_weight:.0%} genre / {1 - genre_weight:.0%} plot.")

# --------------------------------------------------------------------------
# Search and selected-movie hub
# --------------------------------------------------------------------------

# Built column-wise rather than with iterrows(), which builds a Series per row.
available_titles = list(movies["title"])
selection_labels = [
    build_selection_label(title, year)
    for title, year in zip(movies["title"], movies["release_year"])
]
label_to_title = dict(zip(selection_labels, available_titles))

if (
    "selected_title" not in st.session_state
    or st.session_state["selected_title"] not in available_titles
):
    st.session_state["selected_title"] = available_titles[0]

current_label_index = available_titles.index(st.session_state["selected_title"])

col_l, col_main, col_r = st.columns([1, 6, 1])

with col_main:
    with st.container(border=True):
        col_hdr_left, col_hdr_right = st.columns([1, 1])
        with col_hdr_left:
            st.markdown(
                "<span style='font-size: 1.05rem; font-weight: 700; color: #ffffff;'>"
                "🎯 Choose a Movie</span>",
                unsafe_allow_html=True,
            )
        with col_hdr_right:
            st.markdown(
                f"<div style='text-align: right;'>"
                f"<span class='badge-tag badge-info'>📚 {len(movies)} in DB</span>"
                f"<span class='badge-tag badge-genre'>🎭 {len(genre_vocabulary)} Genres</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        selected_label = st.selectbox(
            "Search or select a film from catalogue:",
            options=selection_labels,
            index=current_label_index,
            label_visibility="collapsed",
            help="Type to search through catalogue titles.",
        )
        selected_title = label_to_title[selected_label]
        st.session_state["selected_title"] = selected_title

        selected_index = recommender.find_movie_index(selected_title, movies)
        if selected_index is not None:
            sel_movie = movies.iloc[selected_index]
            genres_tags = "".join(
                f'<span class="badge-tag badge-genre">🎭 {safe(g)}</span>'
                for g in sel_movie["genres"]
            )
            overview_snip = sel_movie.get("overview") or "No synopsis available."

            st.markdown(
                f"""
                <div class="selected-movie-card">
                    {poster_html(sel_movie.get("poster_path"), sel_movie["title"])}
                    <div style="flex: 1;">
                        <div class="selected-movie-title">{safe(sel_movie['title'])} ({safe(sel_movie['release_year'])})</div>
                        <div style="margin-bottom: 4px;">
                            <span class="badge-tag badge-rating">⭐ {format_rating(sel_movie['rating'])}</span>
                            <span class="badge-tag badge-info">📅 {format_release_date(sel_movie['release_date'])}</span>
                            {genres_tags}
                        </div>
                        <div class="selected-movie-desc">{safe(overview_snip)}</div>
                        <div style="font-size: 0.74rem; color: #94a3b8;">
                            <b>Genre Features:</b> <code style="color: #38bdf8; background: rgba(0,0,0,0.35); padding: 1px 6px; border-radius: 4px;">{safe(sel_movie['genre_text'])}</code>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_btn_l, col_btn_surp, col_btn_rec, col_btn_r = st.columns([1, 2, 2.5, 1])
        with col_btn_surp:
            if st.button(
                "🎲 Surprise Me",
                key="surprise_btn",
                width="stretch",
                help="Pick a random film",
            ):
                random_choice = random.choice(available_titles)
                st.session_state["selected_title"] = random_choice
                st.session_state["requested_title"] = random_choice
                st.rerun()
        with col_btn_rec:
            if st.button(
                "✨ Recommend Similar",
                key="recommend_btn",
                type="primary",
                width="stretch",
                help="Rank every other film against this one",
            ):
                st.session_state["requested_title"] = selected_title

# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
if "requested_title" not in st.session_state:
    st.session_state["requested_title"] = selected_title

requested_title = st.session_state.get("requested_title")

if requested_title:
    try:
        recommendations = recommender.recommend_movies(
            movie_title=requested_title,
            movies=movies,
            similarity_matrix=similarity_matrix,
            top_n=recommendation_count,
        )
    except recommender.MovieNotFoundError:
        st.warning(
            f"'{requested_title}' is no longer in the loaded catalogue. "
            "Please choose a movie again."
        )
        st.session_state.pop("requested_title", None)
        st.stop()
    except ValueError as error:
        st.error(f"Could not build recommendations: {error}")
        st.stop()

    if not recommendations:
        st.warning(
            f"Nothing in the catalogue resembles '{requested_title}'. "
            "Try lowering the genre/plot balance in the sidebar."
        )
    else:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 2rem; margin-bottom: 1.25rem;">
                <h3 style="margin: 0; font-size: 1.4rem; font-weight: 700; color: #ffffff;">
                    Films Similar to <span style="color: #ff4d5a;">{safe(requested_title)}</span>
                </h3>
                <div style="font-size: 0.84rem; color: #94a3b8; margin-top: 4px;">Ranked by Content-Based Cosine Similarity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(recommendations) < recommendation_count:
            st.caption(
                f"Only {len(recommendations)} movie(s) scored above zero against "
                "this selection in the current catalogue."
            )

        # Fixed-width rows: st.columns splits a row evenly, so asking for one
        # column per recommendation makes 10 cards unreadably narrow.
        for row_start in range(0, len(recommendations), CARDS_PER_ROW):
            row = recommendations[row_start : row_start + CARDS_PER_ROW]
            columns = st.columns(CARDS_PER_ROW)
            for column, movie in zip(columns, row):
                with column:
                    render_movie_card(movie)

        selected_index = recommender.find_movie_index(requested_title, movies)
        if selected_index is not None:
            selected_movie = movies.iloc[selected_index]
            with st.expander("ℹ️ How were these recommendations calculated?", expanded=False):
                st.markdown(
                    f"Each score is a weighted average of two cosine similarities against "
                    f"**{requested_title}** — **{genre_weight:.0%}** from "
                    f"{len(genre_vocabulary)} genre features and "
                    f"**{1 - genre_weight:.0%}** from {len(overview_vocabulary):,} "
                    f"TF-IDF plot features."
                )
                query_genres = set(selected_movie["genres"])
                overlap_rows = []
                for r in recommendations:
                    rec_genres = set(r["genres"])
                    shared = sorted(query_genres & rec_genres)
                    diff = sorted(rec_genres - query_genres)
                    genre_score = recommender.genre_cosine(
                        selected_movie["genres"], r["genres"]
                    )
                    # Rearranged from the blend formula, so the plot half is
                    # exact rather than recomputed.
                    if genre_weight < 1.0:
                        plot_score = (
                            r["similarity_score"] - genre_weight * genre_score
                        ) / (1.0 - genre_weight)
                        plot_cell = f"{min(max(plot_score, 0.0), 1.0):.3f}"
                    else:
                        plot_cell = "—"
                    overlap_rows.append(
                        {
                            "Recommended Movie": r["title"],
                            "Blended Match": f"{r['similarity_score'] * 100:.1f}%",
                            "Genre Score": f"{genre_score:.3f}",
                            "Plot Score": plot_cell,
                            "Shared Genres": ", ".join(shared) if shared else "None",
                            "Other Genres": ", ".join(diff) if diff else "None",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(overlap_rows), width="stretch", hide_index=True
                )
