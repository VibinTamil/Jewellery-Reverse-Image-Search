# Jewellery Reverse Image Search

A powerful reverse image search engine for jewelry products from BlueStone. Upload an image of jewelry and find similar products instantly using AI-powered image matching.

## Overview

This project combines web scraping, machine learning, and full-stack web development to create a complete reverse image search solution. It scrapes jewelry product data and images from BlueStone, generates AI embeddings using OpenAI's CLIP model, and provides a fast similarity search engine.

Upload any jewelry image, and the system will find the most similar products in the database with similarity scores and direct product links.

## Features

- **Automated Product Scraping**: Scrape jewelry products from BlueStone.com with pagination support for hundreds of products
- **Image Download**: Download and store product images locally with validation and filtering
- **CLIP Embeddings**: Generate high-quality image embeddings using OpenAI's CLIP vision model
- **Fast Similarity Search**: Find similar products using cosine similarity on normalized embeddings
- **REST API Backend**: Built with FastAPI for high performance and easy deployment
- **Interactive Frontend**: Clean, responsive HTML/CSS UI for image upload and results display
- **SQLite Database**: Persistent storage for product metadata and search history
- **CORS Enabled**: Ready for cross-origin requests from deployed frontends

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **PyTorch**: Deep learning framework for CLIP model inference
- **Transformers**: Hugging Face library for accessing pre-trained CLIP model
- **NumPy**: Numerical computing for embeddings and similarity calculations
- **SQLite3**: Lightweight database for product storage
- **Requests & BeautifulSoup4**: Web scraping libraries for data collection
- **Pillow**: Image processing library for resizing and cropping

### Frontend
- **HTML5**: Markup structure
- **CSS3**: Responsive styling
- **JavaScript ES6**: Image upload handling and API communication

## Local Setup

### 1. Create a Virtual Environment

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Backend

From the project root:
```bash
uvicorn backend.main:app --reload
```

The API will start at `http://127.0.0.1:8000`.

### 4. Open the Frontend

1. Navigate to `frontend/index.html` in your browser
2. Or serve it with a simple HTTP server:
   ```bash
   python -m http.server 8080
   ```
   Then visit `http://127.0.0.1:8080/frontend/`

### 5. (Optional) Scrape Product Data

To populate the database with jewelry products:

```bash
cd backend
python scraper.py
```

To generate embeddings for search:

```bash
python generate_embeddings.py
```

## API Endpoints

- `GET /`: Health check endpoint
- `GET /products`: Return all products in the database
- `POST /search`: Upload an image and get similar products

### Search Request

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -F "file=@path/to/image.jpg"
```

### Search Response

```json
{
  "results": [
    {
      "id": 1,
      "name": "Diamond Ring",
      "category": "Ring",
      "image_url": "https://...",
      "product_url": "https://bluestone.com/...",
      "similarity_score": 0.92,
      "similarity_percent": 92.0,
      "match_label": "Highly Similar"
    }
  ]
}
```

## Deployment

### Render Backend

1. Create a GitHub repository with your code
2. Go to [render.com](https://render.com) and sign up
3. Click "New +" and select "Web Service"
4. Connect your GitHub repository
5. Fill in the service settings:
   - **Name**: bluestone-search-api
   - **Region**: Choose closest to your users
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables if needed
7. Click "Create Web Service"

**Important**: You need to upload `bluestone.db` and the `data/` folder to Render as persistent files or add them to your repository.

### Vercel Frontend

1. Go to [vercel.com](https://vercel.com) and sign up
2. Click "Add New..." → "Project"
3. Import your GitHub repository
4. Set the project root to `frontend/`
5. Leave build settings default (static site)
6. Before deploying, update `API_URL` in `frontend/index.html`:
   ```javascript
   const API_URL = "https://your-render-backend.onrender.com";
   ```
7. Click "Deploy"

**Result**: Your frontend will be live at a Vercel URL and will communicate with your Render backend.

## Project Structure

```
BlueStone-Reverse-Image-Search/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── database.py             # SQLite database helpers
│   ├── scraper.py              # Web scraping functions
│   ├── scrape_bluestone.py     # BlueStone dataset scraper
│   ├── search_engine.py        # Similarity search logic
│   ├── generate_embeddings.py  # CLIP embedding generation
│   ├── download_product_images.py
│   └── __pycache__/
├── frontend/
│   └── index.html              # Web UI
├── data/
│   ├── images/                 # Downloaded product images
│   ├── embeddings.npy          # CLIP embeddings
│   └── product_ids.npy         # Product IDs for embeddings
├── bluestone.db                # SQLite database
├── requirements.txt            # Python dependencies
├── Procfile                    # Render deployment config
├── README.md                   # This file
└── venv/                       # Virtual environment
```

## Configuration

### Change Backend URL (Frontend Only)

Edit `frontend/index.html` and update the `API_URL` variable:

```javascript
const API_URL = "http://your-backend-url:8000";
```

### Database Location

The database file (`bluestone.db`) is automatically resolved relative to the project root in `backend/database.py`.

### Embeddings Path

CLIP embeddings are loaded from `data/embeddings.npy` and `data/product_ids.npy` automatically.

## Future Improvements

- **FAISS Integration**: Use Facebook's FAISS library for faster approximate nearest-neighbor search
- **Expand Product Dataset**: Scrape more jewelry categories and thousands of products
- **Enhanced UI**: Improved design with filtering, sorting, and advanced search options
- **Better Ranking**: Machine learning-based re-ranking of results
- **Image Preprocessing**: Advanced image cropping and augmentation
- **Caching**: Redis caching for frequently searched images
- **Authentication**: User accounts and saved searches
- **Admin Dashboard**: Manage products, embeddings, and scraping jobs

## Troubleshooting

### CLIP model not loading
- Ensure PyTorch is installed: `pip install torch torchvision`
- First run may download ~600MB of model files

### Embeddings file not found
- Run `python backend/generate_embeddings.py` after populating products

### CORS errors on frontend
- Ensure backend CORS is enabled in `backend/main.py`
- For production, update CORS settings to match your domain

### Database locked error
- Ensure no other process is using `bluestone.db`
- Close any previous Python processes

## License

This project is for educational purposes.

## Questions?

For issues or suggestions, open a GitHub issue or contact the maintainer.
