import sqlite3
from sqlite3 import Connection
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_FILE = BASE_DIR / "bluestone.db"


def create_connection(db_file: str = DATABASE_FILE) -> Connection:
    """Create a new SQLite database connection."""
    return sqlite3.connect(db_file)


def create_database(db_file: str = DATABASE_FILE) -> None:
    """Create the products table in the SQLite database."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT,
        price REAL,
        image_url TEXT,
        product_url TEXT UNIQUE,
        local_image_path TEXT
    );
    """

    conn = create_connection(db_file)
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
    finally:
        conn.close()


def insert_product(
    name: str,
    category: Optional[str] = None,
    price: Optional[float] = None,
    image_url: Optional[str] = None,
    product_url: Optional[str] = None,
    local_image_path: Optional[str] = None,
    db_file: str = DATABASE_FILE,
) -> int:
    """Insert a product into the products table and return the new row id."""
    insert_sql = """
    INSERT INTO products (
        name,
        category,
        price,
        image_url,
        product_url,
        local_image_path
    ) VALUES (?, ?, ?, ?, ?, ?)
    """

    conn = create_connection(db_file)
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                insert_sql,
                (name, category, price, image_url, product_url, local_image_path),
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # A UNIQUE constraint failed for product_url — skip inserting duplicate.
            if product_url:
                # Return the existing product id for the duplicate URL.
                cursor.execute("SELECT id FROM products WHERE product_url = ?", (product_url,))
                row = cursor.fetchone()
                if row:
                    return row[0]
            # If product_url is None or the select didn't find a row, re-raise the error.
            raise
    finally:
        conn.close()


def get_all_products(db_file: str = DATABASE_FILE) -> List[Dict[str, Optional[str]]]:
    """Return all products stored in the products table."""
    select_sql = "SELECT * FROM products"

    conn = create_connection(db_file)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(select_sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def update_product_details(
    product_url: str,
    image_url: Optional[str] = None,
    price: Optional[float] = None,
    db_file: str = DATABASE_FILE,
) -> None:
    """Update image_url and/or price for a product using product_url."""
    update_sql = "UPDATE products SET "
    params = []

    # Build dynamic UPDATE statement for non-None fields.
    if image_url is not None:
        update_sql += "image_url = ?, "
        params.append(image_url)

    if price is not None:
        update_sql += "price = ?, "
        params.append(price)

    # Remove trailing comma and space, then add WHERE clause.
    update_sql = update_sql.rstrip(", ") + " WHERE product_url = ?"
    params.append(product_url)

    conn = create_connection(db_file)
    try:
        cursor = conn.cursor()
        cursor.execute(update_sql, params)
        conn.commit()
    finally:
        conn.close()
