"""
The recommendation engine: content-based filtering.

    genres + plot text  ->  numeric vectors  ->  cosine similarity

Nothing here knows that TMDB exists, so it can be tested with a handful of
fake movies typed out by hand - no API key and no internet.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Shown instead of a rating or date TMDB did not supply.
UNKNOWN_VALUE_LABEL = "N/A"

DEFAULT_RECOMMENDATION_COUNT = 5

# How much of the blended score comes from genres vs. the plot overview.
# Genres alone give only ~19 features, so thousands of movies end up with
# identical vectors and tie at 1.0; the plot text is what separates them.
DEFAULT_GENRE_WEIGHT = 0.5

# Below this many movies, requiring a term twice would empty the vocabulary.
MIN_DOCUMENTS_FOR_DF_FILTER = 50


class MovieNotFoundError(LookupError):
    """
    Raised when a requested title is not in the dataset.

    Happens when the catalogue is reloaded while an old title is still
    selected; app.py catches it and shows a message instead of crashing.
    """


# --------------------------------------------------------------------------
# STEP 1 - Preprocessing: raw API dictionaries to a clean table
# --------------------------------------------------------------------------


def extract_genre_names(
    genre_ids: Iterable[Any] | None,
    genre_map: dict[int, str],
) -> list[str]:
    """
    Translate TMDB genre ids into names: [28, 878] -> ["Action", "Sci-Fi"].

    This is the join between TMDB's two endpoints. Unknown ids are skipped
    rather than guessed at, since a wrong genre means wrong recommendations.
    """
    if not genre_ids:
        return []

    genre_names: list[str] = []
    for genre_id in genre_ids:
        genre_name = genre_map.get(genre_id)
        if genre_name and genre_name not in genre_names:
            genre_names.append(genre_name)

    return genre_names


def build_genre_token(genre_name: str) -> str:
    """
    Collapse a genre name into one token: "Science Fiction" -> sciencefiction.

    CountVectorizer splits on whitespace, so the raw string would become two
    features ("science", "fiction") instead of one. That would both inflate a
    sci-fi movie's vector and make unrelated genres sharing a word look alike.
    """
    return genre_name.replace(" ", "").replace("-", "").casefold()


def build_genre_text(genre_names: Sequence[str]) -> str:
    """Join genres into one document: "action sciencefiction thriller"."""
    return " ".join(build_genre_token(name) for name in genre_names)


def _clean_text_field(value: Any) -> str | None:
    """Return a stripped string, or None when the API value is unusable."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _to_optional(value: Any) -> Any:
    """
    Convert a pandas missing-value marker back into a plain None.

    Once a column mixes strings and blanks, pandas stores the blanks as NaN.
    Converting at this module's boundary keeps the dictionaries returned by
    recommend_movies() to a simple contract: a real value, or None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None if pd.isna(value) else value


def _clean_rating(value: Any) -> float | None:
    """
    Return a vote average, or None when unknown.

    TMDB sends 0 for never-rated films. Showing "0.0" would claim the movie
    is terrible when the truth is that we do not know.
    """
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return rating if rating > 0 else None


def build_movie_dataframe(
    raw_movies: Iterable[dict[str, Any]],
    genre_map: dict[int, str],
) -> pd.DataFrame:
    """
    Clean raw TMDB dictionaries into a DataFrame, one row per usable movie.

    Columns: movie_id, title, genres, genre_text, overview, rating,
    release_date, release_year, poster_path.

    Missing values in these columns are pandas NaN, not None - test them with
    pd.isna(), or read them through recommend_movies(), which converts back.

    Real API data is messy and paginated, so every missing field, wrong type
    and duplicate is dealt with here once, letting the rest of the project
    trust its input.
    """
    cleaned_rows: list[dict[str, Any]] = []
    seen_movie_ids: set[int] = set()
    seen_titles: set[str] = set()

    for raw_movie in raw_movies:
        if not isinstance(raw_movie, dict):
            continue

        # A title is mandatory: it is how the user picks a movie.
        title = _clean_text_field(raw_movie.get("title")) or _clean_text_field(
            raw_movie.get("original_title")
        )
        if title is None:
            continue

        movie_id = raw_movie.get("id")
        if not isinstance(movie_id, int):
            continue

        # TMDB paginates a list that shifts underneath us, so the same film
        # can arrive twice. A duplicate is a perfect 1.0 self-match and would
        # waste a recommendation slot on the movie the user already picked.
        if movie_id in seen_movie_ids:
            continue
        # Different films can share a title (remakes). The interface keys on
        # title, so keep the first - which vote_count ordering makes the
        # better-known one.
        title_key = title.casefold()
        if title_key in seen_titles:
            continue

        # Genres are mandatory: a movie without them would be an all-zero
        # vector, scoring 0 against everything and never being recommendable.
        genres = extract_genre_names(raw_movie.get("genre_ids"), genre_map)
        if not genres:
            continue

        seen_movie_ids.add(movie_id)
        seen_titles.add(title_key)

        release_date = _clean_text_field(raw_movie.get("release_date"))
        # TMDB dates are ISO strings, so the year is the first four chars.
        release_year = release_date[:4] if release_date else UNKNOWN_VALUE_LABEL

        cleaned_rows.append(
            {
                "movie_id": movie_id,
                "title": title,
                "genres": genres,
                "genre_text": build_genre_text(genres),
                "overview": _clean_text_field(raw_movie.get("overview")) or "",
                "rating": _clean_rating(raw_movie.get("vote_average")),
                "release_date": release_date,
                "release_year": release_year,
                "poster_path": _clean_text_field(raw_movie.get("poster_path")),
            }
        )

    movies = pd.DataFrame(cleaned_rows)

    # Row i of the DataFrame must line up with row i of the similarity matrix.
    # Keeping the old labels after filtering would make movies.iloc[7] and
    # similarity_matrix[7] different films - wrong recommendations with no
    # crash, the worst kind of bug.
    return movies.reset_index(drop=True)


# --------------------------------------------------------------------------
# STEP 2 - Feature representation: text to numbers
# --------------------------------------------------------------------------


def build_genre_vectors(
    genre_texts: Sequence[str],
) -> tuple[Any, CountVectorizer]:
    """
    Vectorize genre text into a (n_movies, n_genres) document-term matrix.

    CountVectorizer fits a vocabulary ({"action": 0, "drama": 1, ...}) then
    counts tokens per movie: "action sciencefiction" -> [1, 0, 0, 1]. Counts
    are only ever 0 or 1 here, so this is effectively multi-hot encoding.

    The matrix is sparse because a movie has 2-3 genres out of ~19. Returns
    the fitted vectorizer too, so callers can inspect the vocabulary.
    """
    if len(genre_texts) == 0:
        raise ValueError("Cannot build genre vectors from an empty dataset.")

    vectorizer = CountVectorizer()
    genre_vectors = vectorizer.fit_transform(genre_texts)
    return genre_vectors, vectorizer


def build_genre_vocabulary(genre_texts: Sequence[str]) -> list[str]:
    """
    Return the genre tokens the vectorizer learned, in column order.

    Seeing "sciencefiction" as one entry rather than "science" and "fiction"
    as two is the quickest proof that build_genre_token() did its job.
    """
    _genre_vectors, vectorizer = build_genre_vectors(genre_texts)
    return vectorizer.get_feature_names_out().tolist()


def build_overview_vectors(
    overview_texts: Sequence[str],
) -> tuple[Any, TfidfVectorizer] | tuple[None, None]:
    """
    Vectorize plot overviews with TF-IDF, or (None, None) if that is not
    possible (every overview empty, or nothing survives the filters).

    TF-IDF rather than raw counts because plot text is unbounded prose: "the"
    appears everywhere and carries no signal, while "heist" or "samurai"
    appears rarely and is highly distinguishing. TF-IDF weights each term by
    how rare it is across the catalogue, which is exactly that intuition.

    English stop words are dropped, and on a real-sized catalogue a term must
    appear in at least two overviews to become a feature - a word used once
    can only ever match that one movie.
    """
    if len(overview_texts) == 0:
        raise ValueError("Cannot build overview vectors from an empty dataset.")

    min_df = 2 if len(overview_texts) >= MIN_DOCUMENTS_FOR_DF_FILTER else 1
    vectorizer = TfidfVectorizer(stop_words="english", min_df=min_df)

    try:
        overview_vectors = vectorizer.fit_transform(overview_texts)
    except ValueError:
        # Raised when the vocabulary comes out empty. Genres alone still work.
        return None, None

    if overview_vectors.shape[1] == 0:
        return None, None

    return overview_vectors, vectorizer


def build_overview_vocabulary(overview_texts: Sequence[str]) -> list[str]:
    """Return the plot terms TF-IDF learned, or [] when there are none."""
    _vectors, vectorizer = build_overview_vectors(overview_texts)
    return [] if vectorizer is None else vectorizer.get_feature_names_out().tolist()


# --------------------------------------------------------------------------
# STEP 3 - Similarity: comparing every movie with every other movie
# --------------------------------------------------------------------------


def build_similarity_matrix(
    genre_texts: Sequence[str],
    overview_texts: Sequence[str] | None = None,
    genre_weight: float = DEFAULT_GENRE_WEIGHT,
) -> np.ndarray:
    """
    Build the (n_movies, n_movies) similarity matrix.

    similarity_matrix[i][j] is how similar movie i is to movie j, from 0.0 to
    1.0. The diagonal is always 1.0, which the recommendation step skips.

    Cosine similarity measures the angle between two vectors:

        cosine(A, B) = (A . B) / (||A|| * ||B||)

    The dot product counts shared features; dividing by the magnitudes cancels
    out a movie simply having MORE of them, so a film tagged with six genres
    does not look closer to everything. That is why cosine beats plain overlap
    counting here.

    With overview_texts supplied, the result is a weighted average of two
    independent cosine scores:

        score = genre_weight * genre_cosine + (1 - genre_weight) * plot_cosine

    Blending is what makes the ranking meaningful. On genres alone, 1,965
    movies collapse into ~689 distinct vectors, so most recommendations tie at
    exactly 1.0 and the order is decided by nothing but TMDB vote count.

    Computed once up front, so each recommendation is only a row lookup.
    """
    if not 0.0 <= genre_weight <= 1.0:
        raise ValueError("genre_weight must be between 0.0 and 1.0.")

    genre_vectors, _vectorizer = build_genre_vectors(genre_texts)
    genre_similarity = cosine_similarity(genre_vectors)

    if overview_texts is None or genre_weight >= 1.0:
        return genre_similarity

    if len(overview_texts) != len(genre_texts):
        raise ValueError(
            "genre_texts and overview_texts must describe the same movies "
            f"({len(genre_texts)} vs {len(overview_texts)})."
        )

    overview_vectors, _overview_vectorizer = build_overview_vectors(overview_texts)
    if overview_vectors is None:
        return genre_similarity

    # cosine_similarity maps an all-zero row (a movie with no usable overview)
    # to 0.0 rather than NaN, so blank plots simply lean on genres.
    overview_similarity = cosine_similarity(overview_vectors)

    return genre_weight * genre_similarity + (1.0 - genre_weight) * overview_similarity


def genre_cosine(genres_a: Sequence[str], genres_b: Sequence[str]) -> float:
    """
    Cosine similarity of two genre lists, straight from set arithmetic.

    Because genre vectors are multi-hot, the dot product is just the size of
    the intersection and each magnitude is sqrt(number of genres). Lets the
    interface show the genre half of a blended score without rebuilding
    anything.
    """
    set_a, set_b = set(genres_a), set(genres_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / math.sqrt(len(set_a) * len(set_b))


# --------------------------------------------------------------------------
# STEP 4 - The recommendation algorithm
# --------------------------------------------------------------------------


def find_movie_index(movie_title: str, movies: pd.DataFrame) -> int | None:
    """
    Return the row position of a title, or None if it is absent.

    A position (0, 1, 2, ...) rather than a pandas label, because that is also
    the row number in the similarity matrix. Matching ignores case and
    surrounding spaces, so "inception" and " Inception " both work.
    """
    if not isinstance(movie_title, str):
        return None

    target = movie_title.strip().casefold()
    if not target:
        return None

    for row_position, title in enumerate(movies["title"]):
        if str(title).strip().casefold() == target:
            return row_position

    return None


def recommend_movies(
    movie_title: str,
    movies: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = DEFAULT_RECOMMENDATION_COUNT,
) -> list[dict[str, Any]]:
    """
    Return up to top_n movies most similar to movie_title, best first.

    Each dictionary holds the movie's display fields plus a similarity_score,
    so the interface can show how similar the recommendation is. The list is
    shorter than top_n when too few movies score above zero.

    Raises:
        MovieNotFoundError: If movie_title is not in the dataset.
        ValueError: If top_n < 1, or if the matrix and DataFrame lengths
            disagree - which would silently return the wrong movies.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # Catches the one bug that is otherwise invisible: a similarity matrix
    # built from a different (older) movie list than the one being indexed.
    if len(movies) != len(similarity_matrix):
        raise ValueError(
            "The movie dataset and the similarity matrix have different "
            f"sizes ({len(movies)} vs {len(similarity_matrix)}); "
            "they must be built from the same data."
        )

    selected_index = find_movie_index(movie_title, movies)
    if selected_index is None:
        raise MovieNotFoundError(f"'{movie_title}' is not in the loaded catalogue.")

    # This single row holds the selection's similarity to every movie, because
    # all the maths was done once in build_similarity_matrix().
    similarity_scores = similarity_matrix[selected_index]

    # Pair each score with its row position first: sorting bare scores would
    # destroy the record of which movie each one belongs to.
    indexed_scores: list[tuple[int, float]] = list(enumerate(similarity_scores))

    # Python's sort is stable, so tied movies keep catalogue order - and since
    # that order is by vote count, ties break towards the better-known film.
    indexed_scores.sort(key=lambda pair: pair[1], reverse=True)

    recommendations: list[dict[str, Any]] = []
    for row_position, score in indexed_scores:
        # A movie is a perfect 1.0 match with itself, so it leads the list.
        if row_position == selected_index:
            continue

        # Zero means nothing in common at all - not a recommendation, just the
        # least-bad row left. Refusing to pad is why the list can be short.
        if score <= 0:
            break

        movie_row = movies.iloc[row_position]
        rating = _to_optional(movie_row["rating"])
        recommendations.append(
            {
                "movie_id": int(movie_row["movie_id"]),
                "title": str(movie_row["title"]),
                "genres": list(movie_row["genres"]),
                "genre_text": str(movie_row["genre_text"]),
                "overview": str(movie_row["overview"]),
                "rating": float(rating) if rating is not None else None,
                "release_date": _to_optional(movie_row["release_date"]),
                "release_year": str(movie_row["release_year"]),
                "poster_path": _to_optional(movie_row["poster_path"]),
                "similarity_score": float(score),
            }
        )

        if len(recommendations) == top_n:
            break

    return recommendations
