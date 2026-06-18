# Tracking Urban Crime Trajectories Through Dynamic Neighborhood Embeddings

![Tests](https://github.com/rrramirrr/Tracking-Urban-Crime-Trajectories-Through-Dynamic-Neighborhood-Embeddings/actions/workflows/api-tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![H3](https://img.shields.io/badge/H3-resolution%208-orange)

> Urban region representation learning applied to Mexico City crime data (2016–2024).  
> Trained autoencoder + K-Means clustering + Markov trajectory analysis, served via a REST API.

---

## Overview

Most crime analysis treats neighborhoods as **static** units. This project asks a different question:

**Can we learn how a neighborhood's crime profile evolves over time — and classify that type of evolution?**

Using open data from Mexico City's FGJ (Fiscalía General de Justicia), we:

1. Build **spatio-temporal crime signatures** for 748 H3 hexagonal cells (resolution 8, ~460m diameter)
2. Train an **autoencoder** to compress 22-dimensional signatures into 8-dimensional latent embeddings
3. Apply **K-Means clustering** to identify 5 crime typologies
4. Classify each cell's year-by-year cluster sequence as one of 4 **trajectory types**
5. Serve all results through a **production REST API** with Docker

---

## Results

| Trajectory type | Count | Description |
|---|---|---|
| `fragmentada` | 379 (51%) | Irregular changes without clear pattern |
| `fluctuante` | 293 (39%) | Changes but returns to baseline |
| `monotónica` | 56 (7%) | Sustained directional shift |
| `estable` | 20 (3%) | Same cluster across all observed years |

Cluster 4 is the most stable — a cell in cluster 4 has a **46.5% probability of remaining in cluster 4** the following year (highest diagonal in the Markov transition matrix).

---

## Architecture

```
Raw FGJ data (534 MB)
        │
        ▼
┌─────────────────────────────────────────────┐
│           ML Pipeline (Jupyter)             │
│                                             │
│  NB01 Preprocessing                         │
│  NB02 H3 Spatio-temporal signatures (22-d) │
│  NB03 Autoencoder → 8-d embeddings          │
│  NB04 K-Means clustering (k=5)              │
│  NB05 Trajectory classification             │
│  NB06 Interactive maps                      │
└────────────────┬────────────────────────────┘
                 │ artifacts
        ┌────────┴────────┐
        │  encoder_weights│  ← PyTorch weights
        │  *.csv          │  ← precomputed results
        └────────┬────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│           urbancrime-api (FastAPI)          │
│                                             │
│  GET  /health                               │
│  GET  /cells/{h3_index}                     │
│  GET  /cells/{h3_index}/trajectory          │
│  GET  /clusters                             │
│  GET  /clusters/{id}                        │
│  GET  /clusters/transitions                 │
│  POST /embed   ← live encoder inference     │
└─────────────────────────────────────────────┘
```

---

## Quickstart

### Option A — Local (with venv)

```bash
git clone https://github.com/rrramirrr/Tracking-Urban-Crime-Trajectories-Through-Dynamic-Neighborhood-Embeddings
cd Tracking-Urban-Crime-Trajectories-Through-Dynamic-Neighborhood-Embeddings

cp urbancrime-api/.env.example urbancrime-api/.env
cd urbancrime-api

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Option B — Docker

```bash
cd urbancrime-api
docker compose up
```

The `../data` directory is mounted read-only into the container — no data is baked into the image.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status + dataset size |
| `GET` | `/cells/{h3_index}` | Cell summary: stability, cluster, trajectory type |
| `GET` | `/cells/{h3_index}/trajectory` | Full year-by-year cluster sequence |
| `GET` | `/clusters` | All 5 cluster profiles (crime composition + temporal patterns) |
| `GET` | `/clusters/{id}` | Cluster detail + list of member cells |
| `GET` | `/clusters/transitions` | Markov transition matrix between clusters |
| `GET` | `/predictions` | 2024 cluster predictions |
| `POST` | `/embed` | Encode a raw 22-d crime signature → 8-d latent vector |

**Example — get a cell's trajectory:**
```bash
curl http://localhost:8000/cells/88499516d9fffff/trajectory
```

**Example — live embedding inference:**
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"features": [0.33, 0.29, 0.12, 0.05, 0.01, 0.07, 0.03, 0.01, 0.02, 0.04,
                    0.19, 0.22, 0.31, 0.28, 0.14, 0.13, 0.14, 0.14, 0.13, 0.15, 0.14, 3.5]}'
```

---

## Technical decisions

**Why H3 instead of administrative boundaries?**  
H3 cells are uniform hexagons (~460m at resolution 8). Administrative boundaries (colonias, alcaldías) are irregular polygons of vastly different sizes, which introduces area-bias when comparing crime rates.

**Why an autoencoder instead of PCA?**  
PCA assumes linear relationships between features. A neighborhood with high nocturnal robbery and another with high daytime robbery are qualitatively different — the autoencoder learns non-linear boundaries that PCA misses.

**Why in-memory data store instead of a database?**  
748 cells × 9 years = 6,224 records, <10 MB total. Loading into Python dicts at startup gives O(1) lookups per request with no network overhead. A database would add operational complexity without any benefit at this scale.

**Why data mounted as a Docker volume instead of baked into the image?**  
Separating code from data is an MLOps principle. Retraining the model only requires updating the mounted volume — no image rebuild needed.

---

## Stack

| Layer | Technology |
|---|---|
| Spatial indexing | H3 (Uber) resolution 8 |
| Representation learning | PyTorch autoencoder |
| Clustering | scikit-learn K-Means |
| Trajectory analysis | Custom Markov + DTW |
| Change point detection | BOCPD + PELT |
| API framework | FastAPI + Pydantic v2 |
| Container | Docker (CPU-only torch, multi-stage build) |
| Data | FGJ CDMX open data (2016–2024) |

---

## Dataset

Source: [Carpetas de Investigación FGJ CDMX](https://datos.cdmx.gob.mx/dataset/carpetas-de-investigacion-fgj-de-la-ciudad-de-mexico)  
Period: January 2016 – January 2025  
Raw size: ~534 MB (not included in this repo)  
Processed: 748 H3 cells with valid annual coverage

---

## Project structure

```
├── Pipeline/                    # Training notebooks (run in order)
│   ├── 01_Preprocesamiento.ipynb
│   ├── 02_Firmas_Espacio_Temporales.ipynb
│   ├── 03_Reduccion_Dimensionalidad.ipynb
│   ├── 04_Clustering.ipynb
│   ├── 05_Trayectorias.ipynb
│   └── 06_Mapas.ipynb
├── data/
│   ├── models/                  # Trained artifacts (.pt, .pkl)
│   ├── processed/               # Pipeline outputs (CSVs)
│   ├── maps/                    # Interactive HTML maps (Folium)
│   └── figures/                 # Result visualizations
├── figures/                     # Preprocessing visualizations
├── urbancrime-api/              # Production REST API
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── routers/
│   │   ├── schemas/
│   │   └── services/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── limite-de-las-alcaldas.json  # CDMX borough boundaries (GeoJSON)
```

---

## Research context

This project sits at the intersection of three areas:

- **Label noise in administrative data** — crime categories in FGJ data are noisy and inconsistently labeled across years
- **Urban region representation learning** — learning embeddings that capture the functional identity of urban zones
- **Temporal dynamics analysis** — modeling how those identities evolve, persist, or shift abruptly

The methodology is transferable to any Latin American city with similar open administrative crime data.
