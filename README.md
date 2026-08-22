# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Python, scikit-learn, TMDB REST API v3, and Streamlit. Pick any film from a ~2,000-movie catalogue and get the closest matches ranked by cosine similarity.

Similarity is computed by the machine learning engine itself — TMDB is used purely as a live data source. Each film is described by two feature blocks: its genres (multi-hot, via `CountVectorizer`) and its plot overview (weighted, via `TfidfVectorizer`). The two are scored separately with `cosine_similarity` and blended into one ranking.

---

## ✨ Features

- **~2,000-Film Catalogue** — Live ingestion from TMDB `/discover/movie`: 100 pages × 20 results, deduplicated to 1,965 unique films spanning all 19 official genres.
- **Blended Similarity Engine** — 19 genre features **plus** 5,556 TF-IDF plot features. Genres alone produce only ~689 distinct vectors across 1,965 films, so thousands of pairs tie at exactly 1.000; the plot component breaks those ties. A sidebar slider moves the weight from pure plot to pure genre.
- **Local Disk Caching** — Catalogue and genre map are cached as JSON, giving a ~0.15s warm start (0.01s to read 2,000 movies, 0.13s to build the 1965 × 1965 matrix). Live re-sync on demand.
- **Works Offline** — If TMDB is unreachable the loaders fall back to the cached copy instead of failing to boot. The app only errors out when there is no cache *and* no network.
- **Polished, Centered UX** — Dark glassmorphic UI with a 175px spotlight card, centered actions (`🎲 Surprise Me`, `✨ Recommend Similar`), and 5-column recommendation cards.
- **Explainability X-Ray** — Collapsible inspector showing shared vs. distinct genres per recommendation, plus the genre and plot sub-scores that add up to the blended total.
- **Robust Network Layer** — Persistent `requests.Session` connection pooling with exponential backoff, retrying transient failures while failing fast on a bad API key.

---

## 🛠️ Tech Stack

| Component         | Tool / Library                    |
| ----------------- | --------------------------------- |
| **Language**      | Python 3.11+ (verified on 3.13)   |
| **Data Source**   | TMDB REST API v3 (via `requests`) |
| **Data Engine**   | pandas & NumPy                    |
| **ML / Math**     | scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `cosine_similarity`) |
| **Interface**     | Streamlit (Wide-mode custom CSS)  |
| **Environment**   | python-dotenv                     |

---

## 📐 How It Works

```text
TMDB REST API v3
   │  /genre/movie/list  +  /discover/movie (100 pages = 2,000 movies)
   ▼
tmdb_api.py
   │  fetch_*  = pure network (session pool, retry backoff)
   │  load_*   = fetch_* wrapped in a JSON disk cache, with offline fallback
   ▼
recommender.build_movie_dataframe()
   │  Clean fields, drop duplicate ids, space-free genre tokens ("sciencefiction")
   ▼
   ├── CountVectorizer  ──▶ 19 multi-hot genre features   ──▶ cosine_similarity ──▶ G
   └── TfidfVectorizer  ──▶ 5,556 weighted plot features  ──▶ cosine_similarity ──▶ P
   ▼
build_similarity_matrix()
   │  S = w·G + (1 − w)·P          (w = genre_weight, default 0.5)
   │  One N × N matrix, precomputed once and cached
   ▼
recommend_movies()
   │  Top-N scores from row i, excluding the query film itself
   ▼
app.py (Streamlit)
   │  Spotlight preview, 5-column card grid, score breakdown table
```

Because the blend is a plain weighted average, the app can recover either component from the total without refitting anything: the genre score is exact set arithmetic (`|A∩B| / √(|A|·|B|)` — valid because genre vectors are multi-hot), and the plot score follows by rearranging the formula.

---

## 📁 Project Structure

```text
movie-recommendation-system/
├── app.py                    # Streamlit UI & interaction state
├── recommender.py            # Preprocessing, vectorizing, similarity & ranking engine
├── tmdb_api.py               # TMDB HTTP client, connection pooling & disk caching
├── requirements.txt          # Pinned dependencies
├── .env.example              # Template for TMDB API key
├── .streamlit/
│   └── config.toml           # Dark theme (required — the custom CSS assumes it)
├── .gitignore                # Excludes .env and the local JSON caches
└── README.md                 # Project documentation
```

---

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repository-url>
cd movie-recommendation-system
```

**2. Create and activate a virtual environment**

Python 3.11 or newer is required (`pandas` 3.x dropped 3.10).
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your TMDB API Key**
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Add your TMDB v3 API Key (get one free at [themoviedb.org](https://www.themoviedb.org/) under *Settings → API*):
```text
TMDB_API_KEY=your_actual_tmdb_api_key_here
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application runs at **`http://localhost:8501`** (Streamlit's default). Pass `--server.port 8600` to change it.

The first run has no cache, so it makes 100 API calls to build the catalogue — measured at ~33s on a home connection. Every run after that starts from disk in ~0.15s. Use **🔄 Reload from TMDB** in the sidebar to force a live re-sync.

### Deploying

On Streamlit Community Cloud, set `TMDB_API_KEY` under *App settings → Secrets* instead of committing a `.env`; Streamlit also exposes secrets as environment variables, which is what `tmdb_api.py` reads. The JSON caches are gitignored, so the first request after a deploy populates them.

---

## 📖 Limitations & Future Roadmap

- **Feature Scope**: Vectors combine genre tokens and TF-IDF plot overviews. Future iterations can add keywords, directors, and top cast, or swap bag-of-words for embeddings (e.g. Sentence-BERT) to catch films that describe the same idea in different words.
- **Cold-Start Cost**: The similarity matrix is dense and O(N²) in memory (~31 MB at N = 1,965). Beyond roughly 10,000 films it should become an approximate-nearest-neighbour index rather than a full matrix.
- **Collaborative Filtering**: Operates strictly on item metadata. Future versions can incorporate user ratings and SVD matrix factorization for personalized feeds.
