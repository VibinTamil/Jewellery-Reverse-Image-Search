import time
import urllib.parse
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

from database import insert_product, create_database, get_all_products, update_product_details


# A simple User-Agent to include with requests so servers know who is requesting.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; JewelleryScraper/0.1; +https://example.com)"
}


def fetch_html(url: str, timeout: int = 10) -> str:
    """Fetch page HTML using requests with a User-Agent header."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _extract_text(element) -> Optional[str]:
    """Helper to get text safely from a BeautifulSoup element."""
    if not element:
        return None
    return element.get_text(strip=True)


def _extract_image_url(card, base_url: str) -> Optional[str]:
    """Try common ways to get an image URL from a product card."""
    # First try an <img> tag
    img = card.select_one("img")
    if img:
        for attr in ("src", "data-src", "data-original", "data-image"):
            val = img.get(attr)
            if val:
                return urllib.parse.urljoin(base_url, val)
    # Fallback: look for an anchor that looks like an image link
    a_img = card.select_one("a img")
    if a_img:
        val = a_img.get("src") or a_img.get("data-src")
        if val:
            return urllib.parse.urljoin(base_url, val)
    return None


def _extract_product_url(card, base_url: str) -> Optional[str]:
    """Find the first anchor in the card and resolve it to an absolute URL."""
    a = card.select_one("a[href]")
    if a:
        return urllib.parse.urljoin(base_url, a.get("href"))
    return None


def _extract_price(card) -> Optional[float]:
    """Try several selectors to find a price and convert to float if possible."""
    # Common selectors for price; keep this list small and readable for beginners.
    price_selectors = [
        ".product-price", ".price", ".final-price", "[data-price]", "span.price"
    ]
    for sel in price_selectors:
        el = card.select_one(sel)
        text = _extract_text(el)
        if text:
            # Remove non-numeric characters except dot
            cleaned = "".join(ch for ch in text if ch.isdigit() or ch == ".")
            try:
                return float(cleaned)
            except ValueError:
                continue
    return None


def scrape_jewellery_category(category_name: str, listing_url: str, limit: int = 10) -> List[Dict[str, Optional[str]]]:
    """Scrape up to `limit` products from a jewellery listing page.

    - Tries a few common product-card selectors and prints debug messages when selectors fail.
    - Inserts each product using `insert_product()`.
    - Returns the list of product dicts found.
    """
    print(f"Scraping category '{category_name}' from: {listing_url} (limit {limit})")

    html = fetch_html(listing_url)
    soup = BeautifulSoup(html, "html.parser")

    # Common guesses for product card selectors; we try them in order until we find matches.
    card_selectors = [
        ".product", ".product-card", ".product-item", ".listing-item", "li.product", ".productBox",
    ]

    cards = []
    for sel in card_selectors:
        cards = soup.select(sel)
        if cards:
            print(f"Found {len(cards)} cards using selector '{sel}'")
            break

    if not cards:
        # Helpful debug: show page title and a short snippet so the developer can inspect selectors.
        title = soup.title.string.strip() if soup.title and soup.title.string else "(no title)"
        snippet = soup.get_text(separator=" ", strip=True)[:500]
        print("No product cards found with common selectors.")
        print(f"Page title: {title}")
        print(f"Page text snippet (first 500 chars): {snippet}")
        return []

    products: List[Dict[str, Optional[str]]] = []

    count = 0
    for card in cards:
        if count >= limit:
            break

        # Try a few selectors for the name; keep order readable for beginners.
        name = None
        for sel in (".product-name", ".name", "h2", "h3", ".title", ".productTitle"):
            el = card.select_one(sel)
            name = _extract_text(el)
            if name:
                break

        if not name:
            # If name is missing, print debug info and skip this card.
            print("Could not extract product name from a card; skipping this card. Inspect the card HTML to add a selector.")
            continue

        price = _extract_price(card)
        image_url = _extract_image_url(card, listing_url)
        product_url = _extract_product_url(card, listing_url)

        if not product_url:
            print(f"Warning: product '{name}' has no product link (selector may be wrong). Skipping.")
            continue

        prod = {
            "name": name,
            "category": category_name,
            "price": price,
            "image_url": image_url,
            "product_url": product_url,
        }

        # Insert into the database. insert_product will skip duplicates by URL.
        try:
            new_id = insert_product(
                name=name,
                category=category_name,
                price=price,
                image_url=image_url,
                product_url=product_url,
            )
            print(f"Inserted/Found id={new_id} for '{name}'")
        except Exception as exc:
            print(f"Failed to insert '{name}': {exc}")

        products.append(prod)
        count += 1

        # Be polite and sleep briefly to avoid hammering the server if real scraping.
        time.sleep(0.5)

    return products


def debug_links(listing_url: str) -> None:
    """Print the first 30 links containing 'rings', 'earrings', or 'jewellery' on the page.

    Useful for finding the correct URLs to scrape before running the full scraper.
    """
    print(f"Fetching page: {listing_url}")
    html = fetch_html(listing_url)
    soup = BeautifulSoup(html, "html.parser")

    # Keywords to search for in links.
    keywords = ("rings", "earrings", "jewellery")

    matching_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        text = _extract_text(link)

        if href and text:
            # Check if any keyword appears in href or text (case-insensitive).
            if any(kw.lower() in href.lower() or kw.lower() in text.lower() for kw in keywords):
                matching_links.append((href, text))

    print(f"Found {len(matching_links)} matching links. Printing first 30:\n")
    for i, (href, text) in enumerate(matching_links[:30], 1):
        print(f"{i}. href: {href}")
        print(f"   text: {text}\n")


def extract_product_links(
    listing_url: str,
    category_name: str,
    limit: int = 10,
    seen_urls: Optional[Set[str]] = None,
) -> int:
    """Extract product links from a retailer listing page.

    - Removes query parameters from each URL first.
    - Filters links by category pattern.
    - Avoids duplicate URLs.
    - Uses link text as product name.
    - Inserts into database with name, category, product_url (price and image_url are None).
    - Prints every inserted product and total count.
    - Returns the number of inserted products.
    """
    if seen_urls is None:
        seen_urls = set()

    print(f"Extracting product links for '{category_name}' from: {listing_url} (limit {limit})")

    html = fetch_html(listing_url)
    soup = BeautifulSoup(html, "html.parser")

    # Determine the pattern to match based on category.
    if category_name == "Ring":
        pattern = "/rings/"
    elif category_name == "Earring":
        pattern = "/earrings/"
    else:
        pattern = f"/{category_name.lower()}s/"

    inserted_count = 0

    # Find all anchor tags and filter by category pattern.
    for link in soup.find_all("a"):
        if inserted_count >= limit:
            break

        href = link.get("href")
        text = _extract_text(link)

        if not href or not text:
            continue

        # Resolve to absolute URL.
        full_url = urllib.parse.urljoin(listing_url, href)

        # Remove query parameters from the URL first.
        parsed = urllib.parse.urlparse(full_url)
        clean_url = urllib.parse.urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, "", "", "")
        )

        # Now check if clean_url matches the category pattern and ends with .html.
        if pattern not in clean_url or not clean_url.endswith(".html"):
            continue

        # Skip duplicates.
        if clean_url in seen_urls:
            continue

        seen_urls.add(clean_url)

        # Insert into database.
        try:
            new_id = insert_product(
                name=text,
                category=category_name,
                product_url=clean_url,
                price=None,
                image_url=None,
            )
            print(f"[OK] Inserted id={new_id}: '{text}'")
            print(f"     URL: {clean_url}")
            inserted_count += 1
        except Exception as exc:
            print(f"[FAIL] Could not insert '{text}': {exc}")

    print(f"\nTotal inserted for {category_name}: {inserted_count}\n")
    return inserted_count


def build_paginated_url(listing_url: str, page_number: int) -> str:
    """Build a page URL for paginated jewellery listings."""
    if page_number <= 1:
        return listing_url
    separator = "&" if "?" in listing_url else "?"
    return f"{listing_url}{separator}page={page_number}"


def scrape_category_pages(
    category_name: str,
    listing_url: str,
    target_count: int = 100,
    max_pages: int = 20,
) -> None:
    """Scrape multiple listing pages until the target count is reached."""
    print(f"\nStarting scrape for {category_name}, target {target_count} products.")

    existing_products = get_all_products()
    seen_urls = {prod["product_url"] for prod in existing_products if prod.get("product_url")}
    total_inserted = 0

    for page_number in range(1, max_pages + 1):
        if total_inserted >= target_count:
            break

        page_url = build_paginated_url(listing_url, page_number)
        print(f"Scraping page {page_number} for {category_name}: {page_url}")

        remaining = target_count - total_inserted
        inserted = extract_product_links(
            page_url,
            category_name,
            limit=remaining,
            seen_urls=seen_urls,
        )

        total_inserted += inserted
        print(
            f"Page {page_number}: products inserted={inserted}, total inserted={total_inserted}"
        )

        if inserted == 0 and page_number > 1:
            print("No new products found on this page, stopping early.")
            break

        time.sleep(1)

    print(f"Finished {category_name} scrape: {total_inserted} total inserted.\n")


def scrape_product_details(product_url: str) -> Tuple[Optional[str], Optional[float]]:
    """Scrape a jewellery product detail page and extract image_url and price.

    - Tries to extract image URL from: img tag src, meta og:image, meta twitter:image
    - Tries to extract price from: meta product:price:amount, visible text
    - Returns tuple (image_url, price) or (None, None) if not found
    """
    try:
        html = fetch_html(product_url)
        soup = BeautifulSoup(html, "html.parser")

        # Extract image URL.
        image_url = None

        # Try <img> tag with src attribute (look for first large image).
        img = soup.select_one("img[src]")
        if img:
            src = img.get("src")
            if src:
                image_url = urllib.parse.urljoin(product_url, src)

        # Fallback: try meta og:image.
        if not image_url:
            og_img = soup.select_one('meta[property="og:image"]')
            if og_img:
                content = og_img.get("content")
                if content:
                    image_url = urllib.parse.urljoin(product_url, content)

        # Fallback: try meta twitter:image.
        if not image_url:
            tw_img = soup.select_one('meta[name="twitter:image"]')
            if tw_img:
                content = tw_img.get("content")
                if content:
                    image_url = urllib.parse.urljoin(product_url, content)

        # Extract price.
        price = None

        # Try meta product:price:amount.
        price_meta = soup.select_one('meta[property="product:price:amount"]')
        if price_meta:
            content = price_meta.get("content")
            if content:
                try:
                    price = float(content)
                except ValueError:
                    pass

        # Fallback: search for price in visible text (common selectors).
        if not price:
            price_selectors = [
                ".product-price", ".price", ".final-price", "[data-price]", "span.price"
            ]
            for sel in price_selectors:
                el = soup.select_one(sel)
                if el:
                    text = _extract_text(el)
                    if text:
                        # Remove non-numeric characters except dot.
                        cleaned = "".join(ch for ch in text if ch.isdigit() or ch == ".")
                        try:
                            price = float(cleaned)
                            break
                        except ValueError:
                            continue

        return (image_url, price)

    except Exception as exc:
        print(f"Error scraping {product_url}: {exc}")
        return (None, None)


def update_missing_product_details(limit: int = 20) -> None:
    """Update image_url and price for products missing those details.

    - Gets all products from database.
    - For each product where image_url is missing and product_url contains the retailer domain:
      - Scrapes product detail page
      - Updates database
      - Prints result
    - Processes up to `limit` products.
    """
    print(f"Updating missing product details (limit {limit})...")

    products = get_all_products()
    updated_count = 0

    for prod in products:
        if updated_count >= limit:
            break

        prod_id = prod.get("id")
        name = prod.get("name")
        product_url = prod.get("product_url")
        image_url = prod.get("image_url")

        # Only process products with missing image_url and the retailer domain URL.
        if image_url or not product_url or "bluestone.com" not in product_url:
            continue

        print(f"Scraping details for id={prod_id}: '{name}'...")

        scraped_image, scraped_price = scrape_product_details(product_url)

        # Update database.
        try:
            update_product_details(
                product_url=product_url,
                image_url=scraped_image,
                price=scraped_price,
            )
            print(f"[OK] Updated id={prod_id}")
            if scraped_image:
                print(f"     Image: {scraped_image}")
            if scraped_price:
                print(f"     Price: {scraped_price}")
            updated_count += 1
        except Exception as exc:
            print(f"[FAIL] Could not update id={prod_id}: {exc}")

        # Be polite and sleep briefly.
        time.sleep(1)

    print(f"\nTotal updated: {updated_count}\n")
    # Ensure the database exists before inserting.
if __name__ == "__main__":
    create_database()

    rings_url = "https://www.bluestone.com/jewellery/rings.html"
    earrings_url = "https://www.bluestone.com/jewellery/earrings.html"

    scrape_category_pages("Ring", rings_url, target_count=100, max_pages=20)
    scrape_category_pages("Earring", earrings_url, target_count=100, max_pages=20)

    update_missing_product_details(limit=100)