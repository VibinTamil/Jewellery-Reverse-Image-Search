import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def generate_embeddings() -> None:
    """Generate CLIP embeddings for all product images.

    - Reads all images from data/images.
    - Extracts product id from filename (e.g., "123.jpg" -> id 123).
    - Generates embeddings using CLIP model.
    - Saves embeddings to data/embeddings.npy.
    - Saves matching product ids to data/product_ids.npy.
    - Skips invalid images.
    - Prints progress.
    """
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Determine device (GPU if available, else CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Using device: {device}\n")

    # Get images directory.
    images_dir = Path(__file__).parent.parent / "data" / "images"
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    # Get all image files.
    image_files = sorted([f for f in images_dir.iterdir() if f.is_file()])
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} image files\n")

    embeddings_list: List[np.ndarray] = []
    product_ids_list: List[int] = []

    for idx, image_path in enumerate(image_files, 1):
        try:
            # Extract product id from filename (e.g., "123.jpg" -> 123).
            stem = image_path.stem
            product_id = int(stem.split("_")[0])    

            # Open and process image.
            image = Image.open(image_path).convert("RGB")

            # Generate embedding.
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device)
                vision_outputs = model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )

                pooled_output = vision_outputs.pooler_output
                image_features = model.visual_projection(pooled_output)

                embedding = image_features / image_features.norm(dim=-1, keepdim=True)

            embeddings_list.append(embedding.cpu().numpy().flatten())
            product_ids_list.append(product_id)

            print(f"[{idx}/{len(image_files)}] Generated embedding for product {product_id}")

        except Exception as exc:
            print(f"[SKIP] {image_path.name}: {exc}")

    print(f"\nGenerated {len(embeddings_list)} embeddings\n")

    if embeddings_list:
        # Convert lists to numpy arrays.
        embeddings_array = np.array(embeddings_list)
        product_ids_array = np.array(product_ids_list)

        # Save embeddings and product ids.
        embeddings_path = Path(__file__).parent.parent / "data" / "embeddings.npy"
        product_ids_path = Path(__file__).parent.parent / "data" / "product_ids.npy"

        np.save(embeddings_path, embeddings_array)
        np.save(product_ids_path, product_ids_array)

        print(f"Saved embeddings ({embeddings_array.shape}) to {embeddings_path}")
        print(f"Saved product ids ({product_ids_array.shape}) to {product_ids_path}")
    else:
        print("No embeddings were generated.")


if __name__ == "__main__":
    generate_embeddings()
