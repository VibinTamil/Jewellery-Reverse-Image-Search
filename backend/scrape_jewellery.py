import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

try:
    from generate_embeddings import generate_embeddings
except ImportError:
    generate_embeddings = None

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
PRODUCTS_FILE = DATA_DIR / "products.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

CATEGORIES = [
    ("Earring", "https://www.bluestone.com/jewellery/earrings.html"),
    ("Ring", "https://www.bluestone.com/jewellery/rings.html"),
]

REQUEST_DELAY = (1.0, 2.5)
MAX_RETRIES = 3


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_products() -> List[Dict[str, Any]]:
    if PRODUCTS_FILE.exists():
        with PRODUCTS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_products(products: List[Dict[str, Any]]) -> None:
    with PRODUCTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)


def clean_url(url: str, base_url: str) -> str:
    absolute = urljoin(base_url, url)
    parsed = urlparse(absolute)
    clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return clean


def fetch_html(url: str) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching page ({attempt}/{MAX_RETRIES}): {url}")
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            print(f"  Request failed: {exc}")
            if attempt < MAX_RETRIES:
                delay = random.uniform(*REQUEST_DELAY)
                print(f"  Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print("  Giving up on page.")
    return None


def download_image(url: str, destination: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading image ({attempt}/{MAX_RETRIES}): {url}")
            resp = requests.get(url, headers=HEADERS, stream=True, timeout=15)
            resp.raise_for_status()
            with destination.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as exc:
            print(f"  Image request failed: {exc}")
            if attempt < MAX_RETRIES:
                delay = random.uniform(*REQUEST_DELAY)
                print(f"  Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print("  Failed to download image.")
    return False


def is_valid_product_image_url(url: str) -> bool:
    lower = url.lower()
    invalid_keywords = [
        "video-call-icon",
        "logo",
        "banner",
        "icon",
        "sprite",
        "static/resources/themes",
        "placeholder",
        "tah",
        "thumb",
        "loader",
        "spinner",
    ]
    if any(kw in lower for kw in invalid_keywords):
        return False
    if "/static/" in lower and "static/resources/themes" in lower:
        return False
    if lower.endswith(".svg"):
        return False
    return True


def get_image_url_from_tag(tag: Any, base_url: str) -> Optional[str]:
    if not tag:
        return None

    for attr in ("data-src", "data-original", "data-lazy", "data-image", "srcset", "src"):
        value = tag.get(attr)
        if not value:
            continue

        if attr == "srcset":
            # srcset contains multiple candidates separated by commas.
            value = value.split(",")[0].strip().split(" ")[0]

        url = clean_url(value, base_url)
        if is_valid_product_image_url(url):
            return url

    return None


def find_product_cards(soup: BeautifulSoup) -> List[Any]:
    selectors = [
        ".productBox",
        ".product-card",
        ".product-item",
        ".listing-item",
        ".productBoxItem",
        ".product-box",
        ".productBox__item",
        "li.product",
        "article.product",
        ".productTile",
        ".grid-item",
    ]
    for sel in selectors:
        cards = soup.select(sel)
        if cards:
            return cards
    return []


def extract_products_from_html(html: str, base_url: str, category_name: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    product_urls = set()
    products: List[Dict[str, Any]] = []
    keyword = "/rings/" if category_name == "Ring" else "/earrings/"

    cards = find_product_cards(soup)
    if not cards:
        cards = soup.find_all("a", href=True)

    for card in cards:
        # Find product link inside the card.
        anchor = card.select_one("a[href*='rings/'], a[href*='earrings/']") if card != None else None
        if not anchor:
            if card.name == "a":
                anchor = card
            else:
                continue

        href = clean_url(anchor["href"], base_url)
        if keyword not in href or not href.endswith(".html"):
            continue

        if href in product_urls:
            continue

        # Extract product name from card title selectors.
        name = None
        for title_sel in (".product-name", ".name", ".title", ".product-title", ".productTitle", "h2", "h3", "h4", ".prod-name", ".productName"):
            title_el = card.select_one(title_sel)
            if title_el:
                name = title_el.get_text(separator=" ", strip=True)
                if name:
                    break

        if not name:
            name = anchor.get_text(separator=" ", strip=True)

        if not name or len(name) < 3:
            continue

        image_url = None
        img_tag = card.select_one("img")
        if img_tag:
            image_url = get_image_url_from_tag(img_tag, base_url)

        if not image_url:
            image_url = get_image_url_from_tag(anchor.select_one("img"), base_url)

        if not image_url:
            image_url = get_image_url_from_tag(card.find_previous("img"), base_url)

        if not image_url:
            continue

        if not is_valid_product_image_url(image_url):
            print(f"Skipped invalid image URL: {image_url}")
            continue

        product_urls.add(href)
        products.append(
            {
                "name": name,
                "category": category_name,
                "product_url": href,
                "image_url": image_url,
            }
        )

    return products


def fetch_category_products(category_name: str, listing_url: str) -> List[Dict[str, Any]]:
    html = fetch_html(listing_url)
    if html is None:
        return []

    products = extract_products_from_html(html, listing_url, category_name)
    if products:
        print(f"Discovered {len(products)} products for {category_name} using static HTML.")
        return products

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Skipping JS rendering fallback.")
        return []

    print("Static HTML did not return products, trying Playwright JS rendering...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=HEADERS["User-Agent"])
            page.goto(listing_url, timeout=30000)
            time.sleep(3)
            html_js = page.content()
            browser.close()
            products = extract_products_from_html(html_js, listing_url, category_name)
            print(f"Discovered {len(products)} products for {category_name} using Playwright.")
            return products
    except Exception as exc:
        print(f"Playwright render failed: {exc}")
        return []


def build_dataset() -> None:
    ensure_directories()
    existing_products = load_existing_products()
    existing_urls = {prod["url"] for prod in existing_products}
    next_id = max((prod.get("id", 0) for prod in existing_products), default=0) + 1

    discovered_total = 0
    downloaded_total = 0
    new_products: List[Dict[str, Any]] = []

    for category_name, listing_url in CATEGORIES:
        print(f"\nScraping category: {category_name}")
        products = fetch_category_products(category_name, listing_url)
        for prod in products:
            if prod["product_url"] in existing_urls:
                continue

            discovered_total += 1
            image_url = prod["image_url"]
            product_id = next_id
            next_id += 1

            parsed = urlparse(image_url)
            ext = Path(parsed.path).suffix or ".jpg"
            image_filename = f"product_{product_id}{ext}"
            image_path = IMAGE_DIR / image_filename

            if download_image(image_url, image_path):
                downloaded_total += 1
                product_record = {
                    "id": product_id,
                    "name": prod["name"],
                    "category": category_name,
                    "image": f"images/{image_filename}",
                    "url": prod["product_url"],
                }
                new_products.append(product_record)
                existing_urls.add(prod["product_url"])
            else:
                print(f"Skipped product due to image download failure: {prod['name']}")

            delay = random.uniform(*REQUEST_DELAY)
            time.sleep(delay)

    if new_products:
        all_products = existing_products + new_products
        save_products(all_products)
        print(f"\nProducts discovered: {discovered_total}")
        print(f"Images downloaded: {downloaded_total}")
        print(f"Products saved: {len(new_products)}")
    else:
        all_products = existing_products
        print("No new products were found.")

    if generate_embeddings and new_products:
        print("\nGenerating image embeddings for the updated dataset...")
        generate_embeddings()
        print("Embedding generation complete.")
    elif not generate_embeddings:
        print("generate_embeddings.py not found, skipping embedding generation.")


if __name__ == "__main__":
    print("Starting Jewellery product dataset generation...")
    build_dataset()
