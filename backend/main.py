from io import BytesIO

import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.database import create_database, get_all_products
from backend.search_engine import find_similar_products

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clip_model: CLIPModel | None = None
clip_processor: CLIPProcessor | None = None


@app.on_event("startup")
def startup_event():
    global clip_model, clip_processor
    create_database()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("CLIP model loaded")


def crop_jewellery_region(image: Image.Image) -> Image.Image:
    """Crop to the likely jewellery region when the image contains a person/model."""
    try:
        import face_recognition
    except ImportError:
        face_recognition = None

    if face_recognition:
        try:
            face_locations = face_recognition.face_locations(np.array(image))
            if face_locations:
                top, right, bottom, left = face_locations[0]
                width, height = image.size
                x_margin = int((right - left) * 0.8)
                y_margin = int((bottom - top) * 1.2)
                crop_left = max(0, left - x_margin)
                crop_top = max(0, top - int(y_margin * 0.5))
                crop_right = min(width, right + x_margin)
                crop_bottom = min(height, bottom + y_margin)
                cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
                print("Cropped to face/jewellery region using face detection.")
                return cropped
        except Exception:
            pass

    width, height = image.size
    # Fallback crop toward the central and upper area where jewelry is often visible.
    left = int(width * 0.1)
    right = int(width * 0.9)
    top = int(height * 0.0)
    bottom = int(height * 0.7)
    cropped = image.crop((left, top, right, bottom))
    print("Used fallback jewellery-region crop.")
    return cropped


def generate_query_embedding(image_bytes: bytes) -> np.ndarray:
    """Generate a CLIP embedding for the uploaded image bytes."""
    global clip_model, clip_processor
    if clip_model is None or clip_processor is None:
        raise RuntimeError("CLIP model is not loaded")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = crop_jewellery_region(image)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)

    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs.pooler_output
        image_features = clip_model.visual_projection(pooled_output)
        embedding = image_features / image_features.norm(dim=-1, keepdim=True)

    embedding_np = embedding.cpu().numpy().flatten()
    print("Query embedding dimension:", embedding_np.shape)
    return embedding_np


def get_match_label(score: float) -> str:
    if score >= 0.95:
        return "Exact Match"
    if score >= 0.85:
        return "Highly Similar"
    if score >= 0.75:
        return "Similar"
    return "Suggested Alternative"


@app.get("/")
def home():
    return {"message": "Jewellery Reverse Image Search API"}


@app.get("/products")
def products():
    return get_all_products()


@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        query_embedding = generate_query_embedding(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}")

    results = find_similar_products(query_embedding, top_k=5)
    products = get_all_products()
    product_map = {int(prod["id"]): prod for prod in products}

    response = []
    for product_id, score in results:
        product = product_map.get(int(product_id))
        if product:
            similarity_percent = round(score * 100, 2)
            response.append(
                {
                    "id": product["id"],
                    "name": product["name"],
                    "category": product.get("category"),
                    "image_url": product.get("image_url"),
                    "product_url": product.get("product_url"),
                    "similarity_score": score,
                    "similarity_percent": similarity_percent,
                    "match_label": get_match_label(score),
                }
            )

    return {"results": response}
