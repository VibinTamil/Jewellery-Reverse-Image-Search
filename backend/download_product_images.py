import os
import urllib.parse
from typing import Dict, List

import requests

from database import get_all_products


def download_product_images() -> None:
    """Download product images from database and save locally.

    - Reads all products from database.
    - For each product with image_url:
      - Derives filename from product id and image URL extension.
      - Skips if file already exists.
      - Downloads image and saves to data/images.
      - Prints success or failure.
    """
    # Ensure the images directory exists.
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")
    os.makedirs(images_dir, exist_ok=True)

    products: List[Dict[str, str]] = get_all_products()
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    for prod in products:
        prod_id = prod.get("id")
        name = prod.get("name") or "<unknown>"
        image_url = prod.get("image_url")

        if not image_url:
            continue

        # Derive filename: use product id and extract extension from URL.
        parsed = urllib.parse.urlparse(image_url)
        path = parsed.path
        # Get file extension or default to .jpg.
        ext = os.path.splitext(path)[1] or ".jpg"
        filename = f"{prod_id}{ext}"
        filepath = os.path.join(images_dir, filename)

        # Skip if file already exists.
        if os.path.exists(filepath):
            print(f"[SKIP] {prod_id} - {name}: file already exists")
            skipped_count += 1
            continue

        try:
            # Stream the download to avoid large memory usage.
            resp = requests.get(image_url, stream=True, timeout=10)
            resp.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"[OK]   {prod_id} - {name}: saved to {filename}")
            downloaded_count += 1
        except Exception as exc:
            # On any error, print a message and continue with the next product.
            print(f"[FAIL] {prod_id} - {name}: failed to download ({exc})")
            failed_count += 1

    print(f"\nSummary:")
    print(f"  Downloaded: {downloaded_count}")
    print(f"  Skipped:    {skipped_count}")
    print(f"  Failed:     {failed_count}")
    print()


if __name__ == "__main__":
    download_product_images()
