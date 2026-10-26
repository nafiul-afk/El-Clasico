"""
Transfermarkt El Clasico Scraper
================================
Scrapes match details, lineups, and player market values
from Transfermarkt. No API key needed.

Source: https://www.transfermarkt.com/

IMPORTANT: Transfermarkt requires a proper User-Agent header
           and blocks requests without one.

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import re
import os


class TransfermarktScraper:
    """Scrapes match and player data from Transfermarkt."""

    BASE_URL = "https://www.transfermarkt.com"

    # Transfermarkt requires this specific header pattern
    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    REQUEST_DELAY = 3

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _get(self, url):
        """Fetch page with rate limiting."""
        print(f"  Fetching: {url}")
        time.sleep(self.REQUEST_DELAY)
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, 'lxml')

    # ----------------------------------------------------------
    # Head-to-head matchup page
    # ----------------------------------------------------------

    def scrape_head_to_head(self):
        """
        Scrape the Real Madrid vs Barcelona head-to-head page.
        URL pattern: /real-madrid/bilanz/verein/131/
        Real Madrid ID = 418, Barcelona ID = 131
        """
        url = f"{self.BASE_URL}/real-madrid/bilanz/verein/131/wettbewerb//saison_id//"
        
        print("\n--- Transfermarkt Head-to-Head ---")
        soup = self._get(url)

        # Find all match rows in the results tables
        matches = []
        tables = soup.find_all('table', class_='items')

        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 3:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    matches.append(row_data)

        if matches:
            # Try to extract from the page tables with pandas
            try:
                dfs = pd.read_html(str(soup), flavor='lxml')
                print(f"Found {len(dfs)} tables")
                for i, df in enumerate(dfs):
                    print(f"  Table {i}: {df.shape}")
                return dfs
            except Exception as e:
                print(f"pandas.read_html failed: {e}")
                return pd.DataFrame(matches)
        
        return None

    # ----------------------------------------------------------
    # Match detail page
    # ----------------------------------------------------------

    def scrape_match_detail(self, match_url):
        """
        Scrape a specific match page for lineups, goals, cards.
        """
        print(f"\n--- Transfermarkt Match Detail ---")
        soup = self._get(match_url)

        match_data = {}

        # Score
        score_box = soup.find('div', class_='sb-endresult')
        if score_box:
            match_data['score'] = score_box.get_text(strip=True)

        # Goals
        goals = []
        goal_rows = soup.find_all('div', class_='sb-aktion-spielstand')
        for goal in goal_rows:
            goals.append(goal.get_text(strip=True))
        match_data['goals'] = goals

        # Cards
        cards = []
        card_rows = soup.find_all('span', class_='sb-sprite-gelbe-karte')
        match_data['yellow_cards'] = len(card_rows)

        red_cards = soup.find_all('span', class_='sb-sprite-rote-karte')
        match_data['red_cards'] = len(red_cards)

        # Lineups
        lineup_tables = soup.find_all('div', class_='large-6')
        for lineup in lineup_tables:
            players = lineup.find_all('a', class_='wichtig')
            player_names = [p.get_text(strip=True) for p in players]
            if player_names:
                if 'home_lineup' not in match_data:
                    match_data['home_lineup'] = player_names
                else:
                    match_data['away_lineup'] = player_names

        # Stadium and attendance
        info_box = soup.find('p', class_='sb-zusatzinfos')
        if info_box:
            match_data['venue_info'] = info_box.get_text(strip=True)

        return match_data

    # ----------------------------------------------------------
    # Season-by-season results
    # ----------------------------------------------------------

    def scrape_season_results(self, season_id):
        """
        Get all matches between Real Madrid and Barcelona in a given season.
        season_id is like 2023 for the 2023-24 season.
        """
        url = (
            f"{self.BASE_URL}/real-madrid/bilanz/verein/131/"
            f"wettbewerb//saison_id/{season_id}"
        )

        print(f"\n--- Transfermarkt Season {season_id}/{season_id+1} ---")
        soup = self._get(url)

        try:
            tables = pd.read_html(str(soup), flavor='lxml')
            if tables:
                result = max(tables, key=len)
                print(f"Found {len(result)} rows")
                return result
        except Exception as e:
            print(f"Error: {e}")

        return None

    # ----------------------------------------------------------
    # Player market values for a given season
    # ----------------------------------------------------------

    def scrape_squad_values(self, team_slug, team_id, season_id):
        """
        Scrape player market values for a team in a season.
        Example: team_slug='real-madrid', team_id=418, season_id=2023
        """
        url = (
            f"{self.BASE_URL}/{team_slug}/kader/verein/{team_id}/"
            f"saison_id/{season_id}"
        )

        print(f"\n--- Squad Values: {team_slug} {season_id} ---")
        soup = self._get(url)

        try:
            tables = pd.read_html(str(soup), flavor='lxml')
            if tables:
                squad = max(tables, key=len)
                print(f"Found {len(squad)} players")
                return squad
        except Exception as e:
            print(f"Error: {e}")

        return None

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    def save(self, df, directory, filename_stem):
        os.makedirs(directory, exist_ok=True)
        csv_path = os.path.join(directory, f"{filename_stem}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved: {csv_path}")


# ==============================================================
# Run
# ==============================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    raw_dir = os.path.join(project_root, "data", "raw")

    scraper = TransfermarktScraper()

    # 1. Head-to-head overview
    h2h = scraper.scrape_head_to_head()
    if h2h is not None:
        if isinstance(h2h, list):
            for i, df in enumerate(h2h):
                scraper.save(df, raw_dir, f"transfermarkt_h2h_table_{i}")
        else:
            scraper.save(h2h, raw_dir, "transfermarkt_h2h")

    # 2. Squad market values (example)
    # Real Madrid 2023-24
    rm_squad = scraper.scrape_squad_values("real-madrid", 418, 2023)
    if rm_squad is not None:
        scraper.save(rm_squad, raw_dir, "transfermarkt_rm_squad_2023")

    # Barcelona 2023-24
    barca_squad = scraper.scrape_squad_values("fc-barcelona", 131, 2023)
    if barca_squad is not None:
        scraper.save(barca_squad, raw_dir, "transfermarkt_barca_squad_2023")
