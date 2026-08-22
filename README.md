# 🎬 Movie Recommendation System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TMDB API](https://img.shields.io/badge/TMDB%20API-v3-01B4E4?logo=themoviedatabase&logoColor=white)](https://developer.themoviedb.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Requests](https://img.shields.io/badge/Requests-2E8B57?logo=python&logoColor=white)](https://requests.readthedocs.io/)
[![python-dotenv](https://img.shields.io/badge/python--dotenv-000000?logo=python&logoColor=white)](https://pypi.org/project/python-dotenv/)

A content-based movie recommendation system built with Python, scikit-learn, TMDB REST API v3, and Streamlit. Pick any film from a ~2,000-movie catalogue and get the closest matches ranked by cosine similarity.

Similarity is computed by the machine learning engine itself — TMDB is used purely as a live data source. Each film is described by two feature blocks: its genres (multi-hot, via `CountVectorizer`) and its plot overview (weighted, via `TfidfVectorizer`). The two are scored separately with `cosine_similarity` and blended into one ranking.

![Movie Recommendation System](https://github.com/Moonlight12911/movie-recommendation-system/blob/main/image.png)

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://movie-recommendation-mk.streamlit.app/)

---

## ✨ Features

- **🎬 ~2,000-Film Catalogue** — Live ingestion from TMDB `/discover/movie`: 100 pages × 20 results, deduplicated to 1,965 unique films spanning all 19 official genres.
- **🧠 Blended Similarity Engine** — 19 genre features **plus** 5,556 TF-IDF plot features. Genres alone produce only ~689 distinct vectors across 1,965 films, so thousands of pairs tie at exactly 1.000; the plot component breaks those ties. A sidebar slider moves the weight from pure plot to pure genre.
- **⚡ Local Disk Caching** — Catalogue and genre map are cached as JSON, giving a ~0.15s warm start (0.01s to read 2,000 movies, 0.13s to build the 1965 × 1965 matrix). Live re-sync on demand.
- **📡 Works Offline** — If TMDB is unreachable, the loaders fall back to the cached copy instead of failing to boot. The app only errors out when there is no cache *and* no network.
- **🎨 Polished, Centered UX** — Dark glassmorphic UI with a 175px spotlight card, centered actions (`🎲 Surprise Me`, `✨ Recommend Similar`), and 5-column recommendation cards.
- **🔍 Explainability X-Ray** — Collapsible inspector showing shared vs. distinct genres for each recommendation, plus the genre and plot sub-scores that add up to the blended total.
- **🛡️ Robust Network Layer** — Persistent `requests.Session` connection pooling with exponential backoff, retrying transient failures while failing fast on a bad API key.
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

### 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd movie-recommendation-system
```

### 2️⃣ Create a Virtual Environment

Python **3.11 or newer** is recommended.

**🪟 Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**🍎 macOS / 🐧 Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Your TMDB API Key 🔑

Create a `.env` file in the project root.

**🪟 Windows PowerShell**

```powershell
copy .env.example .env
```

**🍎 macOS / 🐧 Linux**

```bash
cp .env.example .env
```

Open `.env` and add your TMDB v3 API key:

```env
TMDB_API_KEY=your_actual_tmdb_api_key_here
```

Get your free TMDB API key from [The Movie Database](https://www.themoviedb.org/) under **Settings → API**.

> ⚠️ **Never commit your `.env` file or expose your API key publicly.**
> The `.gitignore` file excludes `.env` from Git.

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

**http://localhost:8501**

To use a different port:

```bash
streamlit run app.py --server.port 8600
```

### ⚡ First Run

The first run has no local cache, so the application fetches the movie catalogue from TMDB and builds the similarity matrices.

```text
TMDB API
   ↓
Fetch movie catalogue
   ↓
Process & clean data
   ↓
Create feature vectors
   ↓
Calculate similarity
   ↓
Cache data locally
   ↓
Launch application
```

The initial catalogue build may take around **30–40 seconds**, depending on your network connection and TMDB response times.

### 🚀 Subsequent Runs

After the initial setup, the application loads the cached catalogue and similarity data from disk, making subsequent launches significantly faster.

To fetch the latest movie data from TMDB, use:

**🔄 Reload from TMDB**

from the sidebar.

---
