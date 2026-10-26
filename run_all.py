"""
Run All: Complete El Clasico Data Pipeline
=============================================
Runs the entire project end-to-end:
  1. Scrape data from Wikipedia and Football-Data.co.uk
  2. Clean and merge into final dataset
  3. Launch the Streamlit dashboard

Usage:
    python run_all.py           # full pipeline + dashboard
    python run_all.py --scrape  # scrape only
    python run_all.py --clean   # clean only
    python run_all.py --dash    # dashboard only (needs data/final/)
"""

import subprocess
import sys
import os
import argparse


ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
RAW_DIR = os.path.join(ROOT, "data", "raw")
FINAL_DIR = os.path.join(ROOT, "data", "final")
FINAL_CSV = os.path.join(FINAL_DIR, "el_clasico_final.csv")


def run_script(path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run([PYTHON, path], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  [WARNING] {label} exited with code {result.returncode}")
    return result.returncode


def step_scrape():
    """Step 1: Scrape data from web sources."""
    print("\n" + "#"*60)
    print("#  STEP 1: DATA COLLECTION")
    print("#"*60)

    wiki_csv = os.path.join(RAW_DIR, "wikipedia_el_clasico_bs4.csv")
    fd_csv = os.path.join(RAW_DIR, "football_data_el_clasico.csv")

    if not os.path.exists(wiki_csv):
        run_script(os.path.join(ROOT, "quick_start.py"), "Wikipedia Scraper")
    else:
        print(f"\n  Wikipedia data already exists: {wiki_csv}")

    if not os.path.exists(fd_csv):
        run_script(
            os.path.join(ROOT, "src", "scrapers", "fbref_scraper.py"),
            "Football-Data.co.uk Scraper",
        )
    else:
        print(f"  Football-Data already exists: {fd_csv}")


def step_clean():
    """Step 2: Clean and merge datasets."""
    print("\n" + "#"*60)
    print("#  STEP 2: DATA CLEANING & MERGING")
    print("#"*60)

    run_script(
        os.path.join(ROOT, "src", "cleaning", "clean_and_merge.py"),
        "Clean & Merge Pipeline",
    )


def step_dashboard():
    """Step 3: Launch Streamlit dashboard."""
    print("\n" + "#"*60)
    print("#  STEP 3: LAUNCHING DASHBOARD")
    print("#"*60)

    app_path = os.path.join(ROOT, "app.py")
    if not os.path.exists(app_path):
        print("  app.py not found!")
        return

    if not os.path.exists(FINAL_CSV):
        print(f"  Final dataset not found: {FINAL_CSV}")
        print("  Run the pipeline first: python run_all.py --scrape --clean")
        return

    print(f"\n  Starting Streamlit dashboard...")
    print(f"  Open http://localhost:8501 in your browser\n")
    subprocess.run([PYTHON, "-m", "streamlit", "run", app_path], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="El Clasico Data Pipeline")
    parser.add_argument("--scrape", action="store_true", help="Run scraping only")
    parser.add_argument("--clean", action="store_true", help="Run cleaning only")
    parser.add_argument("--dash", action="store_true", help="Launch dashboard only")
    args = parser.parse_args()

    # If no flags, run everything
    run_everything = not (args.scrape or args.clean or args.dash)

    print("\n" + "#"*60)
    print("#  EL CLASICO DATA ANALYSIS | FULL PIPELINE")
    print("#"*60)

    if run_everything or args.scrape:
        step_scrape()

    if run_everything or args.clean:
        step_clean()

    if run_everything or args.dash:
        step_dashboard()


if __name__ == "__main__":
    main()
