"""
Quick Start Script - El Clasico Data Collection
================================================

Run this to immediately start collecting data.
No API key needed: uses only web scraping.

Usage:
    python quick_start.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


def scrape_wikipedia():
    """Scrape all El Clasico matches from Wikipedia using pandas.read_html()."""

    url = "https://en.wikipedia.org/wiki/List_of_El_Cl%C3%A1sico_matches"

    print("Fetching data from Wikipedia...")
    print(f"  URL: {url}\n")

    tables = pd.read_html(url)
    print(f"Found {len(tables)} tables on the page\n")

    for i, table in enumerate(tables):
        print(f"  Table {i}: {table.shape[0]} rows x {table.shape[1]} cols")

    # The match-list table is usually the largest
    matches_df = max(tables, key=len)
    print(f"\nSelected table with {len(matches_df)} rows")
    print(f"Columns: {matches_df.columns.tolist()}")

    return matches_df


def scrape_wikipedia_beautifulsoup():
    """Alternative: scrape with BeautifulSoup for more control."""

    url = "https://en.wikipedia.org/wiki/List_of_El_Cl%C3%A1sico_matches"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    print("\nScraping with BeautifulSoup for additional details...")

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'lxml')
    print(f"Page title: {soup.title.string}")

    # Count different elements
    wikitables = soup.find_all('table', class_='wikitable')
    print(f"Found {len(wikitables)} wikitables")

    # Extract match data with links
    all_rows = []
    for table in wikitables:
        header_row = table.find('tr')
        if not header_row:
            continue

        headers_list = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

        for row in table.find_all('tr')[1:]:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            row_data = {}
            for idx, cell in enumerate(cells):
                col = headers_list[idx] if idx < len(headers_list) else f"col_{idx}"
                row_data[col] = cell.get_text(strip=True)
            all_rows.append(row_data)

    if all_rows:
        df = pd.DataFrame(all_rows)
        print(f"Extracted {len(df)} rows via BeautifulSoup")
        return df

    return None

def main():
    print("=" * 60)
    print("El Clasico Data Collection - Quick Start")
    print("=" * 60)
    print("No API keys needed. Using web scraping only.\n")

    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Method 1: pandas.read_html (fast)
    print("-" * 40)
    print("METHOD 1: pandas.read_html()")
    print("-" * 40)

    try:
        df_pandas = scrape_wikipedia()
        output_1 = os.path.join(raw_dir, "wikipedia_el_clasico_pandas.csv")
        df_pandas.to_csv(output_1, index=False, encoding='utf-8')
        print(f"\nSaved to: {output_1}")
        print(f"\nPreview (first 5 rows):")
        print(df_pandas.head().to_string())
    except Exception as e:
        print(f"Error: {e}")
        df_pandas = None

    # Method 2: BeautifulSoup (more detail)
    print("\n" + "-" * 40)
    print("METHOD 2: BeautifulSoup")
    print("-" * 40)

    try:
        df_bs4 = scrape_wikipedia_beautifulsoup()
        if df_bs4 is not None:
            output_2 = os.path.join(raw_dir, "wikipedia_el_clasico_bs4.csv")
            df_bs4.to_csv(output_2, index=False, encoding='utf-8')
            print(f"\nSaved to: {output_2}")
    except Exception as e:
        print(f"Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    print("\nFiles saved in data/raw/:")
    for f in os.listdir(raw_dir):
        size = os.path.getsize(os.path.join(raw_dir, f)) / 1024
        print(f"  {f} ({size:.1f} KB)")

    print("\nNext steps:")
    print("  1. Run the FBref scraper for advanced stats:")
    print("     python src/scrapers/fbref_scraper.py")
    print("  2. Run the Transfermarkt scraper for lineups:")
    print("     python src/scrapers/transfermarkt_scraper.py")
    print("  3. Open notebooks/01_data_collection.ipynb for the full guide")


if __name__ == "__main__":
    main()
