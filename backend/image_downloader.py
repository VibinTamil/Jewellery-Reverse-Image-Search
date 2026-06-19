import os
import urllib.parse
from typing import Dict, List

import requests

from backend.database import get_all_products


def download_images() -> None:
    """Download product images from `image_url` for all products in the DB.

    - Reads products via `get_all_products()`.
    - Saves files into `data/images/`.
    - Skips failed downloads without raising an exception.
    - Prints success or failure messages for each image.
    """
    # Ensure the images directory exists.
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")
    os.makedirs(images_dir, exist_ok=True)

    products: List[Dict[str, str]] = get_all_products()

    for prod in products:
        prod_id = prod.get("id")
        name = prod.get("name") or "<unknown>"
        image_url = prod.get("image_url")

        if not image_url:
            print(f"[SKIP] {prod_id} - {name}: no image_url")
            continue

        # Derive a safe filename from the URL. Prefix with product id to avoid collisions.
        parsed = urllib.parse.urlparse(image_url)
        basename = os.path.basename(parsed.path) or "image"
        filename = f"{prod_id}_{basename}"
        filepath = os.path.join(images_dir, filename)

        try:
            # Stream the download to avoid large memory usage.
            resp = requests.get(image_url, stream=True, timeout=10)
            resp.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"[OK]   {prod_id} - {name}: saved to {filepath}")
        except Exception as exc:
            # On any error, print a message and continue with the next product.
            print(f"[FAIL] {prod_id} - {name}: failed to download {image_url} ({exc})")


if __name__ == "__main__":
    download_images()
