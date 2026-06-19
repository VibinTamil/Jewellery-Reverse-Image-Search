import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from search_engine import find_similar_products, load_embeddings


def generate_query_embedding(image_path: Path) -> np.ndarray:
    """Generate a CLIP image embedding from a single image file."""
    print(f"Loading image: {image_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs.pooler_output
        image_features = model.visual_projection(pooled_output)
        embedding = image_features / image_features.norm(dim=-1, keepdim=True)

    print("Embedding shape:", embedding.cpu().numpy().shape)
    return embedding.cpu().numpy().flatten()


def main() -> None:
    images_dir = Path(__file__).parent.parent / "data" / "images"
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)

    image_files = sorted(images_dir.glob("*"))
    if not image_files:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    sample_image = image_files[0]
    query_embedding = generate_query_embedding(sample_image)
    print("Query embedding dimension:", query_embedding.shape)

    embeddings, _ = load_embeddings()
    print("Stored embeddings dimension:", embeddings.shape)

    print("\nSearching for similar products...")
    results = find_similar_products(query_embedding, top_k=5)

    print("\nTop 5 similar product ids and scores:")
    for product_id, score in results:
        print(f"product_id={product_id}, score={score:.4f}")


if __name__ == "__main__":
    main()
