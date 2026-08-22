"""
TMDB API layer - the only module that talks to The Movie Database over HTTP.

Two levels of function:
  fetch_*  pure network calls, no caching (easy to test with mocks)
  load_*   cache-backed wrappers used by the app: read disk, else fetch,
           and fall back to a stale cache when the network is unavailable
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

TMDB_BASE_URL = "https://api.themoviedb.org/3"

# TMDB sends relative poster paths ("/abc.jpg"); the client picks the size.
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# requests has no default timeout, and a hang would freeze Streamlit forever.
REQUEST_TIMEOUT_SECONDS = 10

MOVIES_PER_TMDB_PAGE = 20

# 100 pages x 20 movies = 2,000 movies.
DEFAULT_PAGE_COUNT = 100

# TMDB caps /discover/movie pagination.
MAX_TMDB_PAGE = 500

MAX_REQUEST_ATTEMPTS = 3

# Multiplied by the attempt number, so waits grow: 0.5s, then 1.0s.
RETRY_BACKOFF_SECONDS = 0.5

PROJECT_DIRECTORY = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIRECTORY / ".env")

# Reassigned by the test suite to keep checks off the real cache.
CATALOGUE_CACHE_FILE = PROJECT_DIRECTORY / ".tmdb_catalogue_cache.json"
GENRE_CACHE_FILE = PROJECT_DIRECTORY / ".tmdb_genre_cache.json"

# Connection pooling / keep-alive across the many catalogue pages.
_SESSION = requests.Session()

# Treated as "no key" so the user gets setup steps, not a raw HTTP 401.
PLACEHOLDER_API_KEYS = {
    "",
    "your_api_key_here",
    "your_tmdb_api_key_here",
    "changeme",
}


class TMDBError(Exception):
    """Any failure to get usable data out of TMDB, with a readable message."""


class _TransientTMDBError(Exception):
    """
    A failure worth retrying, private to this module.

    A reset connection may succeed on a second attempt; a rejected API key
    will fail identically every time. Only the first kind is retried.
    """


# --------------------------------------------------------------------------
# API key handling
# --------------------------------------------------------------------------


def get_api_key() -> str:
    """
    Return the TMDB key from the environment or Streamlit secrets.

    Raises:
        TMDBError: If the key is missing or still a placeholder.
    """
    api_key = os.getenv("TMDB_API_KEY", "").strip()

    # Fallback to Streamlit secrets (for Streamlit Community Cloud deployments)
    if not api_key or api_key.casefold() in PLACEHOLDER_API_KEYS:
        try:
            import streamlit as st

            if hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
                api_key = str(st.secrets["TMDB_API_KEY"]).strip()
        except Exception:
            pass

    if api_key.casefold() in PLACEHOLDER_API_KEYS:
        raise TMDBError(
            "No TMDB API key found.\n\n"
            "1. Create a free account at https://www.themoviedb.org/\n"
            "2. Copy your API key (v3 auth) from Settings > API\n"
            "3. For local use: add TMDB_API_KEY=your_key to .env\n"
            "   For Streamlit Cloud: add TMDB_API_KEY = \"your_key\" in App Settings > Secrets\n"
            "4. Restart the app."
        )

    return api_key


# --------------------------------------------------------------------------
# Disk cache helpers
# --------------------------------------------------------------------------


def _read_json_cache(path: Path) -> Any | None:
    """Return the parsed cache file, or None if it is missing or corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write_json_cache(path: Path, payload: Any) -> None:
    """Best-effort cache write. A read-only disk must not break the app."""
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        pass


# --------------------------------------------------------------------------
# The single shared request helper
# --------------------------------------------------------------------------


def _attempt_request(
    url: str, params: dict[str, Any], endpoint: str
) -> dict[str, Any]:
    """
    Make one attempt and return the decoded JSON body.

    Its only job is to classify what went wrong; the caller decides whether
    to try again.

    Raises:
        _TransientTMDBError: Timeout, dropped connection, 429, or 5xx.
        TMDBError: Rejected key, wrong endpoint, or an unreadable body.
    """
    try:
        response = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout as error:
        raise _TransientTMDBError(
            f"TMDB did not respond within {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from error
    except requests.exceptions.ConnectionError as error:
        # DNS failures, refused connections, and mid-transfer resets.
        raise _TransientTMDBError("Could not reach TMDB.") from error
    except requests.exceptions.RequestException as error:
        # Bad configuration rather than bad luck, so do not retry.
        raise TMDBError(f"The request to TMDB failed: {error}") from error

    if response.status_code == 401:
        raise TMDBError(
            "TMDB rejected the API key (HTTP 401). "
            "Check that TMDB_API_KEY in your .env file is the correct "
            "v3 auth key and has no extra spaces or quotes."
        )
    if response.status_code == 404:
        raise TMDBError(f"TMDB endpoint not found (HTTP 404): {endpoint}")
    if response.status_code == 429:
        raise _TransientTMDBError("TMDB rate limit reached (HTTP 429).")
    if response.status_code >= 500:
        raise _TransientTMDBError(
            f"TMDB had a server error (HTTP {response.status_code})."
        )
    if not response.ok:
        # Any other 4xx: repeating an invalid request changes nothing.
        raise TMDBError(
            f"TMDB returned an unexpected status code {response.status_code} "
            f"for {endpoint}."
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise TMDBError("TMDB returned a response that was not valid JSON.") from error

    if not isinstance(payload, dict):
        raise TMDBError("TMDB returned JSON in an unexpected shape.")

    return payload


def _request(endpoint: str, extra_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    GET a TMDB endpoint, retrying transient failures.

    Raises:
        TMDBError: For any problem. Transient ones are only reported after
            MAX_REQUEST_ATTEMPTS attempts.
    """
    # TMDB v3 takes the key as a query parameter; requests URL-encodes it.
    # Fetched once before the loop: no retry can conjure up a missing key.
    params: dict[str, Any] = {"api_key": get_api_key()}
    if extra_params:
        params.update(extra_params)

    url = f"{TMDB_BASE_URL}{endpoint}"

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            return _attempt_request(url, params, endpoint)
        except _TransientTMDBError as error:
            if attempt == MAX_REQUEST_ATTEMPTS:
                raise TMDBError(
                    f"{error} Tried {MAX_REQUEST_ATTEMPTS} times. "
                    "Please check your internet connection and try again."
                ) from error

            # Without the sleep this would be a tight loop of three requests
            # in a few milliseconds - exactly what gets a client throttled.
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise TMDBError("The request to TMDB failed.")


# --------------------------------------------------------------------------
# Pure network fetches
# --------------------------------------------------------------------------


def fetch_genre_map() -> dict[int, str]:
    """
    Fetch TMDB's genres as an {id: name} table, e.g. {28: "Action"}.

    The movie list endpoint only carries numeric genre_ids, so this lookup is
    what turns them into words - and into the model's text features.

    Raises:
        TMDBError: If the request fails or the genre list comes back empty.
    """
    payload = _request("/genre/movie/list", {"language": "en-US"})

    # .get() so a missing key raises our error, not a KeyError.
    raw_genres = payload.get("genres") or []

    genre_map: dict[int, str] = {}
    for genre in raw_genres:
        genre_id = genre.get("id")
        genre_name = genre.get("name")
        # A half-usable entry would silently poison every movie citing it.
        if isinstance(genre_id, int) and isinstance(genre_name, str) and genre_name.strip():
            genre_map[genre_id] = genre_name.strip()

    if not genre_map:
        raise TMDBError(
            "TMDB returned an empty genre list, so genres cannot be resolved."
        )

    return genre_map


def fetch_movies(page_count: int = DEFAULT_PAGE_COUNT) -> list[dict[str, Any]]:
    """
    Fetch raw movie dictionaries from /discover/movie, newest votes first.

    One failed page is tolerated - a partial catalogue still works - but if
    every page fails the first error is re-raised.

    Raises:
        TMDBError: If no movies could be collected at all.
    """
    if page_count < 1:
        raise ValueError("page_count must be at least 1.")

    collected_movies: list[dict[str, Any]] = []
    first_error: TMDBError | None = None

    for page_number in range(1, min(page_count, MAX_TMDB_PAGE) + 1):
        try:
            payload = _request(
                "/discover/movie",
                {
                    "sort_by": "vote_count.desc",
                    "include_adult": "false",
                    "include_video": "false",
                    "language": "en-US",
                    "page": page_number,
                },
            )
            results = payload.get("results") or []
            if not results:
                break
            collected_movies.extend(results)
        except TMDBError as error:
            if first_error is None:
                first_error = error
            continue

    if not collected_movies:
        raise first_error or TMDBError("TMDB returned no movies at all.")

    return collected_movies


# --------------------------------------------------------------------------
# Cache-backed loaders used by the app
# --------------------------------------------------------------------------


def load_genre_map(force_refresh: bool = False) -> dict[int, str]:
    """
    Return the genre table, preferring the disk cache.

    The genre list is ~19 rows that change perhaps once a year, so caching it
    is what lets the app start with no network at all. JSON object keys are
    always strings, hence the int() conversion on the way back in.
    """
    if not force_refresh:
        cached = _read_json_cache(GENRE_CACHE_FILE)
        if isinstance(cached, dict) and cached:
            return {int(key): value for key, value in cached.items()}

    try:
        genre_map = fetch_genre_map()
    except TMDBError:
        # A refresh that cannot reach TMDB should still yield a usable app.
        cached = _read_json_cache(GENRE_CACHE_FILE)
        if isinstance(cached, dict) and cached:
            return {int(key): value for key, value in cached.items()}
        raise

    _write_json_cache(GENRE_CACHE_FILE, {str(k): v for k, v in genre_map.items()})
    return genre_map


def load_catalogue(
    page_count: int = DEFAULT_PAGE_COUNT, force_refresh: bool = False
) -> list[dict[str, Any]]:
    """
    Return raw movies, preferring the disk cache.

    The cache is only trusted when it holds enough movies for the requested
    page count, so raising the page count still triggers a real fetch.
    """
    if page_count < 1:
        raise ValueError("page_count must be at least 1.")

    wanted = page_count * MOVIES_PER_TMDB_PAGE

    if not force_refresh:
        cached = _read_json_cache(CATALOGUE_CACHE_FILE)
        # Pages can come back short, so accept a cache that is close enough.
        if isinstance(cached, list) and len(cached) >= wanted * 0.75:
            return cached[:wanted]

    try:
        movies = fetch_movies(page_count=page_count)
    except TMDBError:
        cached = _read_json_cache(CATALOGUE_CACHE_FILE)
        if isinstance(cached, list) and cached:
            return cached[:wanted]
        raise

    _write_json_cache(CATALOGUE_CACHE_FILE, movies)
    return movies[:wanted]


# --------------------------------------------------------------------------
# Image helper
# --------------------------------------------------------------------------


def build_poster_url(poster_path: str | None) -> str | None:
    """
    Turn a TMDB poster_path into a full image URL, or None if there is none.

    The isinstance check comes first because this value can arrive from a
    pandas column as NaN, and `not value` raises TypeError for some pandas
    missing-value markers.
    """
    if not isinstance(poster_path, str) or not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
