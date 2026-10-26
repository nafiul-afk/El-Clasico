# El Clasico All-Time Data Analysis

Real Madrid vs FC Barcelona: 120+ years of rivalry data, scraped, cleaned, analyzed, and predicted.

**Final Dataset:** 202 matches (1929-2026) | 68 features | Shots, fouls, cards, corners, goals, scorers

## Overview

This project scrapes historical El Clasico match data from multiple web sources, cleans and merges it into a single dataset, performs exploratory data analysis with 20 visualizations, trains 10 machine learning models, and presents everything through an interactive Streamlit dashboard.

## Quick Start

```bash
git clone https://github.com/nafiul-afk/El-Clasico.git
cd El-Clasico

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

# Run the full pipeline
python run_all.py

# Or run individual steps
python run_all.py --scrape    # Scrape data only
python run_all.py --clean     # Clean and merge only
python run_all.py --dash      # Launch dashboard only
```

## Project Structure

```
El-Clasico/
|
|-- run_all.py                       # Single entry point for entire pipeline
|-- app.py                           # Streamlit interactive dashboard
|-- quick_start.py                   # Quick Wikipedia scraper
|-- requirements.txt
|
|-- src/
|   |-- scrapers/
|   |   |-- wikipedia_scraper.py     # BeautifulSoup
|   |   |-- fbref_scraper.py         # Football-Data.co.uk CSV downloader
|   |   |-- transfermarkt_scraper.py # BeautifulSoup
|   |   |-- selenium_scraper.py      # Selenium for dynamic sites
|   |
|   |-- cleaning/
|       |-- clean_and_merge.py       # 8-step cleaning and merging pipeline
|
|-- data/
|   |-- raw/                         # Scraped data
|   |-- processed/                   # Cleaned intermediate files
|   |-- final/                       # Final merged dataset
|
|-- notebooks/
    |-- 01_data_collection.ipynb     # Scraping with BeautifulSoup and Selenium
    |-- 02_data_cleaning.ipynb       # Cleaning, imputation, transformation
    |-- 03_eda.ipynb                 # 20 visualizations
    |-- 04_ml_models.ipynb           # 10 ML models with prediction
```

## Data Sources

All data collected through web scraping. No API keys required.

| Source | Method | Data | Coverage |
|--------|--------|------|----------|
| Wikipedia | BeautifulSoup | Match list, scores, goalscorers | 1929-2026 |
| Football-Data.co.uk | CSV download | Shots, fouls, cards, corners | 1993-2026 |

## Data Pipeline

| Step | Description |
|------|------------|
| Scraping | Wikipedia + Football-Data.co.uk |
| Cleaning | Fix encoding, parse dates, remove junk columns, deduplicate |
| Imputation | Median, KNN, MICE for missing values |
| Merging | Left join on date and teams |
| Encoding | Label encoding for categorical features |
| Outlier Detection | Z-Score, IQR, K-Means clustering |
| Normalization | Min-Max scaling, Z-Score standardization |
| Feature Engineering | Goal categories, decade, venue, rolling averages |

## EDA: 20 Visualizations

| # | Analysis | Chart Type |
|---|---------|------------|
| 1 | Dataset overview | Summary table |
| 2 | All-time win record | Pie chart |
| 3 | Goals per match by decade | Grouped bar |
| 4 | Total goals distribution | Histogram with KDE |
| 5 | Home vs away goals | Box plot |
| 6 | RM goals vs Barca goals | Scatter plot |
| 7 | Win rate by decade | Stacked bar |
| 8 | Home advantage | Grouped bar |
| 9 | Monthly match distribution | Bar chart |
| 10 | Shots vs goals correlation | Scatter with regression |
| 11 | Fouls and cards over time | Dual-axis line |
| 12 | Goal category breakdown | Donut chart |
| 13 | Top 10 biggest wins | Horizontal bar |
| 14 | Correlation matrix | Heatmap |
| 15 | Corners comparison | Violin plot |
| 16 | Half-time vs full-time results | Grouped bar |
| 17 | Shots on target efficiency | Multi-bar |
| 18 | Outlier matches | Annotated scatter |
| 19 | Cumulative wins over time | Line with fill |
| 20 | K-Means clusters | Cluster scatter |

## ML Models

### Classification (8 models)

| Model | Target |
|-------|--------|
| Logistic Regression | Match result (H/D/A) |
| Random Forest | Match result (H/D/A) |
| K-Nearest Neighbors | RM Win / Loss / Draw |
| Support Vector Machine | RM Win / Loss / Draw |
| Decision Tree | Goal category |
| XGBoost | Match result (H/D/A) |
| Ensemble Voting | Match result (H/D/A) |
| Tuned RF (GridSearchCV) | Match result (H/D/A) |

### Regression (2 models)

| Model | Target |
|-------|--------|
| Linear Regression | Total goals |
| Random Forest Regressor | Total goals |

### Features

- Rolling 5-match averages (RM goals, Barca goals, total goals)
- Rolling 5-match win rates (RM, Barca, draw)
- Cumulative all-time RM win rate
- Home/away indicator
- Year

## Dashboard

Launch with `streamlit run app.py`

| Page | Content |
|------|---------|
| Overview | Metrics, win record, goals distribution, recent matches |
| Head to Head | Goals by decade, win rate trends, home advantage, cumulative wins |
| Match Stats | Correlations, corners, goal categories, biggest wins |
| Trends | Goals over time, fouls and cards, monthly distribution |
| Outliers | Z-Score outliers, K-Means clusters |
| Predict | Select home team and year, get prediction with probabilities |

## Requirements

- Python 3.10+
- Chrome browser (for Selenium scraping only)
- All packages listed in `requirements.txt`

## Author

**nafiul-afk** | [GitHub](https://github.com/nafiul-afk)
