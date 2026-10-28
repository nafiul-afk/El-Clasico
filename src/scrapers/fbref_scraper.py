"""
Football-Data.co.uk El Clasico Scraper
======================================
Scrapes detailed match statistics from football-data.co.uk.
They provide FREE downloadable CSV files for each La Liga season with:
  - Match results, goals, half-time scores
  - Shots, shots on target, corners
  - Fouls, yellow cards, red cards
  - Betting odds (can be used as features)

Source: https://www.football-data.co.uk/spainm.php
Data format: CSV files (one per season)

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import time
import re
import os


class FootballDataScraper:
    """
    Scrapes La Liga match data from football-data.co.uk.
    They host CSV files with detailed match statistics.
    """

    BASE_URL = "https://www.football-data.co.uk"
    SPAIN_URL = "https://www.football-data.co.uk/spainm.php"

    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
    }

    REQUEST_DELAY = 1

    COLUMN_INFO = {
        'Div': 'Division (SP1 = La Liga)',
        'Date': 'Match date',
        'Time': 'Kick-off time',
        'HomeTeam': 'Home team name',
        'AwayTeam': 'Away team name',
        'FTHG': 'Full-time home goals',
        'FTAG': 'Full-time away goals',
        'FTR': 'Full-time result (H=Home Win, D=Draw, A=Away Win)',
        'HTHG': 'Half-time home goals',
        'HTAG': 'Half-time away goals',
        'HTR': 'Half-time result',
        'HS': 'Home shots',
        'AS': 'Away shots',
        'HST': 'Home shots on target',
        'AST': 'Away shots on target',
        'HF': 'Home fouls',
        'AF': 'Away fouls',
        'HC': 'Home corners',
        'AC': 'Away corners',
        'HY': 'Home yellow cards',
        'AY': 'Away yellow cards',
        'HR': 'Home red cards',
        'AR': 'Away red cards',
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _get(self, url):
        """Fetch content with rate limiting."""
        print(f"  Fetching: {url}")
        time.sleep(self.REQUEST_DELAY)
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp

    def get_season_links(self):
        """
        Scrape the Spain page to find all CSV download links.
        Each season has a separate CSV file.
        """
        print("\n--- Finding La Liga CSV links ---")
        resp = self._get(self.SPAIN_URL)
        soup = BeautifulSoup(resp.text, 'lxml')

        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # La Liga CSV files match pattern like mmz4281/2324/SP1.csv
            if 'SP1.csv' in href or 'SP2.csv' in href:
                if 'SP1' in href:  # SP1 = La Liga first division
                    full_url = href
                    if not href.startswith('http'):
                        full_url = f"{self.BASE_URL}/{href}"
                    links.append(full_url)

        print(f"  Found {len(links)} La Liga season CSV links")
        return links


    def download_season(self, csv_url):
        """
        Download a season CSV file and return as DataFrame.
        """
        resp = self._get(csv_url)

        try:
            df = pd.read_csv(StringIO(resp.text), on_bad_lines='skip')
            # Extract season from URL
            match = re.search(r'/(\d{4})/SP1', csv_url)
            if match:
                season_code = match.group(1)
                df['season_code'] = season_code
            return df
        except Exception as e:
            print(f"  Error parsing CSV: {e}")
            return None

    

    def filter_el_clasico(self, df):
        """Filter a La Liga DataFrame for El Clasico matches only."""
        if df is None or df.empty:
            return None

        # Find the team name columns
        home_col = None
        away_col = None
        for col in df.columns:
            if col.lower() in ('hometeam', 'home'):
                home_col = col
            elif col.lower() in ('awayteam', 'away'):
                away_col = col

        if not home_col or not away_col:
            return None

        # Real Madrid name variations
        rm_names = ['Real Madrid', 'R Madrid', 'Real']
        barca_names = ['Barcelona', 'FC Barcelona', 'Barca']

        mask = (
            (df[home_col].isin(rm_names) & df[away_col].isin(barca_names)) |
            (df[home_col].isin(barca_names) & df[away_col].isin(rm_names))
        )

        return df[mask].copy()


    def scrape_all_el_clasico(self):
        """
        Download all La Liga seasons and extract El Clasico matches.
        This is the main method to call.
        """
        print("\n" + "=" * 50)
        print("Football-Data.co.uk - El Clasico Scraper")
        print("=" * 50)

        # Step 1: Get all season CSV links
        links = self.get_season_links()

        if not links:
            print("No CSV links found!")
            return None

        # Step 2: Download each season and filter
        all_el_clasico = []
        all_seasons = []

        for i, link in enumerate(links):
            print(f"\n[{i+1}/{len(links)}] Downloading: {link}")
            df = self.download_season(link)

            if df is not None and not df.empty:
                all_seasons.append(df)

                el_clasico = self.filter_el_clasico(df)
                if el_clasico is not None and not el_clasico.empty:
                    all_el_clasico.append(el_clasico)
                    print(f"  Found {len(el_clasico)} El Clasico match(es)")
                else:
                    print(f"  No El Clasico matches in this season")

        # Step 3: Combine all El Clasico matches
        if all_el_clasico:
            combined = pd.concat(all_el_clasico, ignore_index=True)
            print(f"\n{'=' * 50}")
            print(f"Total El Clasico matches with detailed stats: {len(combined)}")
            print(f"Columns: {combined.columns.tolist()}")
            print(f"{'=' * 50}")
            return combined

        # Step 4: Also save the full La Liga dataset
        if all_seasons:
            full = pd.concat(all_seasons, ignore_index=True)
            print(f"\nFull La Liga dataset: {len(full)} matches")
            return full

        return None

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    def save(self, df, directory, filename_stem):
        os.makedirs(directory, exist_ok=True)
        csv_path = os.path.join(directory, f"{filename_stem}.csv")
        json_path = os.path.join(directory, f"{filename_stem}.json")

        df.to_csv(csv_path, index=False, encoding='utf-8')
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)

        print(f"Saved CSV:  {csv_path}")
        print(f"Saved JSON: {json_path}")

    def print_column_guide(self):
        """Print what each column in the data means."""
        print("\nColumn Guide:")
        print("-" * 50)
        for col, desc in self.COLUMN_INFO.items():
            print(f"  {col:10s} : {desc}")


# ==============================================================
# Run
# ==============================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    raw_dir = os.path.join(project_root, "data", "raw")

    scraper = FootballDataScraper()

    # Show column guide
    scraper.print_column_guide()

    # Scrape all El Clasico matches with detailed stats
    el_clasico = scraper.scrape_all_el_clasico()

    if el_clasico is not None:
        scraper.save(el_clasico, raw_dir, "football_data_el_clasico")
        print(f"\nPreview:")
        print(el_clasico.head(10).to_string())
