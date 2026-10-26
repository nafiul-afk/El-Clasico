"""
Wikipedia El Clasico Scraper
============================
Scrapes the complete list of all El Clasico matches from Wikipedia.
No API key needed. This is the primary data source.

Source: https://en.wikipedia.org/wiki/List_of_El_Cl%C3%A1sico_matches

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import os
import time
from datetime import datetime


class WikipediaElClasicoScraper:
    """Scrapes all El Clasico match data from Wikipedia."""

    MATCHES_URL = "https://en.wikipedia.org/wiki/List_of_El_Cl%C3%A1sico_matches"
    MAIN_URL = "https://en.wikipedia.org/wiki/El_Cl%C3%A1sico"

    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.matches_df = None
        self.summary_df = None

    # ----------------------------------------------------------
    # Fetching
    # ----------------------------------------------------------

    def fetch_page(self, url):
        """Fetch HTML content from a URL and return a BeautifulSoup object."""
        print(f"Fetching: {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'lxml')

    # ----------------------------------------------------------
    # Method 1: pandas.read_html (easiest approach)
    # ----------------------------------------------------------

    def scrape_with_pandas(self):
        """
        Use pandas.read_html() to pull every HTML table from the
        Wikipedia matches page. This is the fastest and simplest way.
        """
        print("\n--- Method 1: pandas.read_html() ---")
        print(f"URL: {self.MATCHES_URL}\n")

        tables = pd.read_html(self.MATCHES_URL)
        print(f"Found {len(tables)} tables on the page\n")

        for i, table in enumerate(tables):
            print(f"  Table {i}: {table.shape[0]} rows x {table.shape[1]} cols")

        # The match-list table is usually the largest one
        self.matches_df = max(tables, key=len)
        print(f"\nSelected table with {len(self.matches_df)} rows")
        print(f"Columns: {self.matches_df.columns.tolist()}")
        return self.matches_df

    # ----------------------------------------------------------
    # Method 2: BeautifulSoup (more control)
    # ----------------------------------------------------------

    def scrape_with_beautifulsoup(self):
        """
        Parse the page manually with BeautifulSoup.
        Gives finer control over which tables and cells to extract.
        """
        print("\n--- Method 2: BeautifulSoup ---")

        soup = self.fetch_page(self.MATCHES_URL)
        wikitables = soup.find_all('table', class_='wikitable')
        print(f"Found {len(wikitables)} wikitables")

        all_matches = []

        for table in wikitables:
            header_row = table.find('tr')
            if not header_row:
                continue

            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

            for row in table.find_all('tr')[1:]:
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue

                row_data = {}
                for idx, cell in enumerate(cells):
                    col_name = headers[idx] if idx < len(headers) else f"col_{idx}"
                    row_data[col_name] = cell.get_text(strip=True)

                    # Also grab any links (useful for match report URLs)
                    link = cell.find('a', href=True)
                    if link and 'href' in link.attrs:
                        row_data[f"{col_name}_link"] = "https://en.wikipedia.org" + link['href']

                all_matches.append(row_data)

        print(f"Extracted {len(all_matches)} rows via BeautifulSoup")

        if all_matches:
            self.matches_df = pd.DataFrame(all_matches)

        return self.matches_df

    # ----------------------------------------------------------
    # Scrape overall rivalry statistics from the main article
    # ----------------------------------------------------------

    def scrape_summary_stats(self):
        """
        Scrape the summary / statistics box from the main
        El Clasico Wikipedia page (overall wins, draws, goals, etc.)
        """
        print("\n--- Scraping summary statistics ---")

        soup = self.fetch_page(self.MAIN_URL)

        # Look for the infobox / statistics table
        infobox = soup.find('table', class_='infobox')
        if infobox:
            rows = infobox.find_all('tr')
            stats = {}
            for row in rows:
                header = row.find('th')
                value = row.find('td')
                if header and value:
                    stats[header.get_text(strip=True)] = value.get_text(strip=True)

            self.summary_df = pd.DataFrame([stats])
            print("Summary stats extracted:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        else:
            print("Infobox not found on the page")

        return self.summary_df

    # ----------------------------------------------------------
    # Saving helpers
    # ----------------------------------------------------------

    def save(self, df, directory, filename_stem):
        """Save a DataFrame as both CSV and JSON."""
        os.makedirs(directory, exist_ok=True)

        csv_path = os.path.join(directory, f"{filename_stem}.csv")
        json_path = os.path.join(directory, f"{filename_stem}.json")

        df.to_csv(csv_path, index=False, encoding='utf-8')
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)

        print(f"Saved CSV : {csv_path}")
        print(f"Saved JSON: {json_path}")


# ==============================================================
# Run
# ==============================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    raw_dir = os.path.join(project_root, "data", "raw")

    scraper = WikipediaElClasicoScraper()

    # 1. Quick approach
    matches = scraper.scrape_with_pandas()
    if matches is not None:
        scraper.save(matches, raw_dir, "wikipedia_matches_pandas")
        print(f"\nPreview:\n{matches.head()}")

    # 2. Detailed approach
    matches_bs = scraper.scrape_with_beautifulsoup()
    if matches_bs is not None:
        scraper.save(matches_bs, raw_dir, "wikipedia_matches_bs4")

    # 3. Summary stats
    summary = scraper.scrape_summary_stats()
    if summary is not None:
        scraper.save(summary, raw_dir, "wikipedia_summary_stats")
