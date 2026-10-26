"""
Selenium Scraper for Dynamic Websites
=====================================
For websites that load data via JavaScript (Sofascore, WhoScored, etc.)
No API key needed: Selenium drives a real browser.

Setup:
    pip install selenium webdriver-manager
    (Chrome browser must be installed)
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os


class SeleniumScraper:
    """Base class for Selenium-based scraping."""

    def __init__(self, headless=True):
        self.options = Options()
        if headless:
            self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--window-size=1920,1080')
        self.options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
        self.driver = None

    def start(self):
        """Start the Chrome WebDriver."""
        print("Starting Chrome WebDriver...")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=self.options)
        print("WebDriver ready.")

    def stop(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            print("WebDriver closed.")

    def load_page(self, url, wait_seconds=5):
        """Navigate to a URL and wait for content to render."""
        print(f"Loading: {url}")
        self.driver.get(url)
        time.sleep(wait_seconds)

    def wait_for(self, css_selector, timeout=10):
        """Wait until an element matching the CSS selector appears."""
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))

    def get_tables(self):
        """Parse all tables from the current page source."""
        try:
            return pd.read_html(self.driver.page_source)
        except ValueError:
            print("No tables found on the page.")
            return []

    def save(self, df, directory, filename_stem):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{filename_stem}.csv")
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"Saved: {path}")


# ------------------------------------------------------------------
# FBref via Selenium (fallback when requests are blocked)
# ------------------------------------------------------------------

class FBrefSeleniumScraper(SeleniumScraper):
    """Use Selenium to scrape FBref when plain requests get blocked."""

    REAL_MADRID_ID = "53a199f4"
    BARCELONA_ID = "206d90db"

    def scrape_head_to_head(self):
        """Get head-to-head results from FBref."""
        url = (
            f"https://fbref.com/en/stathead/matchup/teams/"
            f"{self.REAL_MADRID_ID}/{self.BARCELONA_ID}/"
        )

        self.start()
        try:
            self.load_page(url, wait_seconds=6)
            tables = self.get_tables()

            if tables:
                print(f"Found {len(tables)} tables")
                largest = max(tables, key=len)
                print(f"Main table: {largest.shape}")
                return largest
            return None
        finally:
            self.stop()

    def scrape_match_report(self, match_url):
        """Scrape a single FBref match report."""
        self.start()
        try:
            self.load_page(match_url, wait_seconds=6)
            tables = self.get_tables()
            print(f"Found {len(tables)} tables in match report")
            return tables
        finally:
            self.stop()


# ------------------------------------------------------------------
# ESPN FC via Selenium
# ------------------------------------------------------------------

class ESPNScraper(SeleniumScraper):
    """Scrape match data from ESPN FC."""

    def scrape_el_clasico_results(self):
        """
        Scrape El Clasico results from ESPN's fixture pages.
        ESPN renders content dynamically so Selenium is needed.
        """
        # ESPN search for Real Madrid vs Barcelona
        url = "https://www.espn.com/soccer/matchstats/_/gameId/"

        # Alternative: team fixture page
        # Real Madrid fixtures for a specific season
        url_fixtures = "https://www.espn.com/soccer/team/fixtures/_/id/86/real-madrid"

        self.start()
        try:
            self.load_page(url_fixtures, wait_seconds=5)

            # Scroll down to load more fixtures
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

            tables = self.get_tables()
            print(f"Found {len(tables)} tables")

            if tables:
                # Find the fixtures table
                fixtures = max(tables, key=len)

                # Filter for Barcelona matches
                mask = fixtures.apply(
                    lambda r: r.astype(str).str.contains('Barcelona', case=False).any(),
                    axis=1
                )
                el_clasico = fixtures[mask]
                print(f"El Clasico matches found: {len(el_clasico)}")
                return el_clasico

            return None
        finally:
            self.stop()

    def scrape_match_stats(self, game_id):
        """
        Scrape detailed match statistics from an ESPN match page.
        game_id example: 123456
        """
        url = f"https://www.espn.com/soccer/matchstats/_/gameId/{game_id}"

        self.start()
        try:
            self.load_page(url, wait_seconds=5)

            # Wait for the stats section
            try:
                self.wait_for('.stat-list', timeout=10)
            except Exception:
                pass

            tables = self.get_tables()
            print(f"Found {len(tables)} stat tables")

            return tables
        finally:
            self.stop()


# ------------------------------------------------------------------
# Sofascore via Selenium
# ------------------------------------------------------------------

class SofascoreScraper(SeleniumScraper):
    """Scrape match data from Sofascore."""

    def scrape_head_to_head(self):
        """Get Real Madrid vs Barcelona head-to-head from Sofascore."""
        url = "https://www.sofascore.com/team/football/real-madrid/2829"

        self.start()
        try:
            self.load_page(url, wait_seconds=5)

            # Scroll to load content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

            tables = self.get_tables()
            print(f"Found {len(tables)} tables")

            return tables if tables else None
        finally:
            self.stop()


# ==============================================================
# Run
# ==============================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    raw_dir = os.path.join(project_root, "data", "raw")

    print("=" * 60)
    print("Selenium Scraper for El Clasico Data")
    print("=" * 60)
    print()
    print("Available scrapers:")
    print("  1. FBrefSeleniumScraper  - Advanced stats (xG, possession)")
    print("  2. ESPNScraper           - Match events, scores")
    print("  3. SofascoreScraper      - Ratings, live data")
    print()
    print("Requirements:")
    print("  - Chrome browser installed")
    print("  - pip install selenium webdriver-manager")
    print()

    # Example: FBref head-to-head
    print("Running FBref Selenium scraper...")
    fbref = FBrefSeleniumScraper(headless=True)
    h2h = fbref.scrape_head_to_head()

    if h2h is not None:
        fbref_scraper = SeleniumScraper()
        csv_path = os.path.join(raw_dir, "fbref_selenium_h2h.csv")
        os.makedirs(raw_dir, exist_ok=True)
        h2h.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        print(f"\nPreview:\n{h2h.head()}")
