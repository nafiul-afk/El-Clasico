"""
Data Cleaning and Merging Pipeline
===================================
Cleans both scraped datasets and merges them into one final CSV.

Source 1: Wikipedia (420 rows, 1929-2026, scores + goalscorers)
Source 2: Football-Data.co.uk (64 rows, 1993-2026, detailed match stats)

"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans


# ==============================================================
# STEP 1: Clean Wikipedia Data
# ==============================================================

def clean_wikipedia(filepath):
    """
    Clean the raw Wikipedia El Clasico data.

    Problems fixed:
      - Remove junk columns (scraped from summary tables)
      - Fix score encoding (dash character issues)
      - Parse dates (missing space before year)
      - Extract home/away goals from score
      - Remove non-match rows
      - Deduplicate
    """
    print("=" * 60)
    print("STEP 1: Cleaning Wikipedia Data")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"  Raw: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---- 1a. Keep only useful columns ----
    keep_cols = ['No.', 'Date', 'Matchweek', 'Home team', 'Away team',
                 'Score', 'Goals (home)', 'Goals (away)',
                 'Season', 'Round', 'Stadium', 'Tournament', 'Season[36]']
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()
    print(f"  After column selection: {len(df.columns)} columns")

    # ---- 1b. Remove rows that are not actual matches ----
    # A valid match row must have Home team and Away team
    df = df.dropna(subset=['Home team', 'Away team'], how='any')
    print(f"  After dropping rows without teams: {len(df)} rows")

    # Keep only rows where both teams are Real Madrid or Barcelona
    valid_teams = ['Real Madrid', 'Barcelona', 'FC Barcelona',
                   'Real Madrid CF', 'Real Madrid Castilla']
    mask = (
        df['Home team'].isin(valid_teams) &
        df['Away team'].isin(valid_teams)
    )
    df = df[mask].copy()
    print(f"  After filtering valid El Clasico teams: {len(df)} rows")

    # ---- 1c. Fix score encoding ----
    # The dash in scores gets corrupted: "1-2" -> "1\u2013 2" or similar
    def fix_score(score):
        if pd.isna(score):
            return score
        score = str(score)
        # Replace various dash characters with a standard hyphen
        score = score.replace('\u2013', '-')  # en-dash
        score = score.replace('\u2014', '-')  # em-dash
        score = score.replace('\ufffd', '-')  # replacement character
        score = re.sub(r'[^\d\-]', '-', score)  # anything non-digit becomes dash
        # Clean up multiple dashes
        score = re.sub(r'-+', '-', score)
        score = score.strip('-')
        return score

    df['Score'] = df['Score'].apply(fix_score)
    print(f"  Fixed score encoding")

    # ---- 1d. Extract home and away goals from Score ----
    def extract_goals(score):
        if pd.isna(score):
            return None, None
        match = re.match(r'(\d+)\s*-\s*(\d+)', str(score))
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    goals = df['Score'].apply(lambda s: pd.Series(extract_goals(s), index=['home_goals', 'away_goals']))
    df['home_goals'] = goals['home_goals']
    df['away_goals'] = goals['away_goals']
    print(f"  Extracted goals: {df['home_goals'].notna().sum()} matches with valid scores")

    # ---- 1e. Fix dates ----
    # Dates look like "17 February1929" (missing space before year)
    def fix_date(date_str):
        if pd.isna(date_str):
            return None
        date_str = str(date_str).strip()
        # Insert space before 4-digit year if missing
        date_str = re.sub(r'(\D)(\d{4})', r'\1 \2', date_str)

        formats = [
            '%d %B %Y',      # 17 February 1929
            '%B %d, %Y',     # February 17, 1929
            '%d %b %Y',      # 17 Feb 1929
            '%Y-%m-%d',      # 1929-02-17
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    df['date_parsed'] = df['Date'].apply(fix_date)
    parsed_count = df['date_parsed'].notna().sum()
    print(f"  Parsed dates: {parsed_count}/{len(df)}")

    # Extract year, month, day
    df['year'] = df['date_parsed'].apply(lambda d: d.year if pd.notna(d) and d is not None else None)
    df['month'] = df['date_parsed'].apply(lambda d: d.month if pd.notna(d) and d is not None else None)
    df['day'] = df['date_parsed'].apply(lambda d: d.day if pd.notna(d) and d is not None else None)
    df['date_str'] = df['date_parsed'].apply(
        lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) and d is not None else None
    )

    # ---- 1f. Standardize team names ----
    team_map = {
        'FC Barcelona': 'Barcelona',
        'Real Madrid CF': 'Real Madrid',
        'Real Madrid Castilla': 'Real Madrid',
    }
    df['Home team'] = df['Home team'].replace(team_map)
    df['Away team'] = df['Away team'].replace(team_map)

    # ---- 1g. Determine result ----
    def get_result(row):
        h, a = row['home_goals'], row['away_goals']
        if pd.isna(h) or pd.isna(a):
            return None
        if h > a:
            return 'H'
        elif h < a:
            return 'A'
        return 'D'

    df['result'] = df.apply(get_result, axis=1)

    # ---- 1h. Determine winner ----
    def get_winner(row):
        if row['result'] == 'H':
            return row['Home team']
        elif row['result'] == 'A':
            return row['Away team']
        elif row['result'] == 'D':
            return 'Draw'
        return None

    df['winner'] = df.apply(get_winner, axis=1)

    # ---- 1i. Total goals ----
    df['total_goals'] = df['home_goals'] + df['away_goals']
    df['goal_diff'] = df['home_goals'] - df['away_goals']

    # ---- 1j. Deduplicate ----
    before = len(df)
    df = df.drop_duplicates(subset=['date_str', 'Home team', 'Away team'], keep='first')
    print(f"  Deduplication: {before} -> {len(df)} rows")

    # ---- 1k. Clean goalscorer text ----
    def clean_goalscorer_text(text):
        if pd.isna(text):
            return None
        text = str(text)
        # Remove reference markers like [8], [9]
        text = re.sub(r'\[\d+\]', '', text)
        # Fix encoding
        text = text.replace('\ufffd', 'e')
        return text.strip() if text.strip() else None

    if 'Goals (home)' in df.columns:
        df['Goals (home)'] = df['Goals (home)'].apply(clean_goalscorer_text)
    if 'Goals (away)' in df.columns:
        df['Goals (away)'] = df['Goals (away)'].apply(clean_goalscorer_text)

    # ---- 1l. Count goalscorers ----
    def count_scorers(text):
        if pd.isna(text) or not text:
            return 0
        # Count commas + 1 gives number of scoring events roughly
        return len(re.findall(r'\(', str(text)))

    df['home_scorer_count'] = df['Goals (home)'].apply(count_scorers)
    df['away_scorer_count'] = df['Goals (away)'].apply(count_scorers)

    # ---- Final column rename ----
    df = df.rename(columns={
        'Home team': 'home_team',
        'Away team': 'away_team',
        'Goals (home)': 'home_scorers',
        'Goals (away)': 'away_scorers',
        'No.': 'match_number',
        'Matchweek': 'matchweek',
    })

    # Select and order final columns
    final_cols = [
        'date_str', 'year', 'month', 'day',
        'home_team', 'away_team',
        'home_goals', 'away_goals', 'total_goals', 'goal_diff',
        'result', 'winner',
        'home_scorers', 'away_scorers',
        'home_scorer_count', 'away_scorer_count',
        'matchweek', 'match_number',
        'Season', 'Round', 'Stadium', 'Tournament', 'Season[36]',
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].copy()

    print(f"\n  Final Wikipedia data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ==============================================================
# STEP 2: Clean Football-Data.co.uk Data
# ==============================================================

def clean_football_data(filepath):
    """
    Clean the raw Football-Data.co.uk El Clasico data.

    Problems fixed:
      - Remove 200+ betting/odds columns (keep match stats only)
      - Parse dates into standard format
      - Handle missing stats for older seasons (22 matches)
      - Standardize column names
    """
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning Football-Data.co.uk Data")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"  Raw: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---- 2a. Keep only match stat columns ----
    keep_cols = {
        'Date': 'date_raw',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'FTR': 'result',
        'HTHG': 'ht_home_goals',
        'HTAG': 'ht_away_goals',
        'HTR': 'ht_result',
        'HS': 'home_shots',
        'AS': 'away_shots',
        'HST': 'home_shots_on_target',
        'AST': 'away_shots_on_target',
        'HF': 'home_fouls',
        'AF': 'away_fouls',
        'HC': 'home_corners',
        'AC': 'away_corners',
        'HY': 'home_yellows',
        'AY': 'away_yellows',
        'HR': 'home_reds',
        'AR': 'away_reds',
        'season_code': 'season_code',
    }

    available = {k: v for k, v in keep_cols.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)
    print(f"  After column selection: {len(df.columns)} columns")

    # ---- 2b. Parse dates ----
    def parse_fd_date(date_str):
        if pd.isna(date_str):
            return None
        date_str = str(date_str).strip()
        formats = [
            '%d/%m/%Y',   # 26/10/2025
            '%d/%m/%y',   # 08/05/94
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    df['date_parsed'] = df['date_raw'].apply(parse_fd_date)
    df['date_str'] = df['date_parsed'].apply(
        lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) and d is not None else None
    )
    df['year'] = df['date_parsed'].apply(lambda d: d.year if pd.notna(d) and d is not None else None)
    df['month'] = df['date_parsed'].apply(lambda d: d.month if pd.notna(d) and d is not None else None)
    df['day'] = df['date_parsed'].apply(lambda d: d.day if pd.notna(d) and d is not None else None)

    parsed_count = df['date_parsed'].notna().sum()
    print(f"  Parsed dates: {parsed_count}/{len(df)}")

    # ---- 2c. Convert goals to int ----
    for col in ['home_goals', 'away_goals', 'ht_home_goals', 'ht_away_goals']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # ---- 2d. Derived features ----
    df['total_goals'] = df['home_goals'] + df['away_goals']
    df['goal_diff'] = df['home_goals'] - df['away_goals']
    df['total_shots'] = df['home_shots'] + df['away_shots']
    df['total_fouls'] = df['home_fouls'] + df['away_fouls']
    df['total_cards'] = df['home_yellows'] + df['away_yellows'] + df['home_reds'] + df['away_reds']

    # ---- 2e. Season string ----
    def season_code_to_string(code):
        if pd.isna(code):
            return None
        code = str(int(code)) if isinstance(code, float) else str(code)
        if len(code) == 4:
            start = int(code[:2])
            end = int(code[2:])
            if start > 90:
                return f"19{start:02d}-{start+1:02d}"
            else:
                return f"20{start:02d}-{end:02d}"
        return code

    df['season'] = df['season_code'].apply(season_code_to_string)

    # Drop intermediate columns
    df = df.drop(columns=['date_raw', 'date_parsed', 'season_code'], errors='ignore')

    # ---- 2f. Report missing values ----
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n  Missing values:")
        for col, count in missing_cols.items():
            print(f"    {col}: {count}/{len(df)}")

    print(f"\n  Final Football-Data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ==============================================================
# STEP 3: Merge Datasets
# ==============================================================

def merge_datasets(wiki_df, fd_df):
    """
    Merge Wikipedia and Football-Data on date + teams.
    Wikipedia has historical data (1929+).
    Football-Data has detailed stats (1993+).
    The merged dataset will have stats where available.

    """
    print("\n" + "=" * 60)
    print("STEP 3: Merging Datasets")
    print("=" * 60)

    print(f"  Wikipedia:      {len(wiki_df)} rows")
    print(f"  Football-Data:  {len(fd_df)} rows")

    # Standardize team names for matching
    wiki_df = wiki_df.copy()
    fd_df = fd_df.copy()

    # Create merge keys
    wiki_df['merge_key'] = (
        wiki_df['date_str'].astype(str) + '_' +
        wiki_df['home_team'].astype(str) + '_' +
        wiki_df['away_team'].astype(str)
    )

    fd_df['merge_key'] = (
        fd_df['date_str'].astype(str) + '_' +
        fd_df['home_team'].astype(str) + '_' +
        fd_df['away_team'].astype(str)
    )

    # Columns from Football-Data to add (stats not in Wikipedia)
    fd_stat_cols = [
        'ht_home_goals', 'ht_away_goals', 'ht_result',
        'home_shots', 'away_shots',
        'home_shots_on_target', 'away_shots_on_target',
        'home_fouls', 'away_fouls',
        'home_corners', 'away_corners',
        'home_yellows', 'away_yellows',
        'home_reds', 'away_reds',
        'total_shots', 'total_fouls', 'total_cards',
        'season',
    ]
    fd_stat_cols = [c for c in fd_stat_cols if c in fd_df.columns]

    fd_for_merge = fd_df[['merge_key'] + fd_stat_cols].copy()

    # Left join: keep all Wikipedia rows, add stats where available
    merged = wiki_df.merge(fd_for_merge, on='merge_key', how='left', suffixes=('', '_fd'))
    merged = merged.drop(columns=['merge_key'], errors='ignore')

    matched = merged['home_shots'].notna().sum()
    print(f"  Matched rows (with detailed stats): {matched}")
    print(f"  Unmatched rows (Wikipedia only): {len(merged) - matched}")
    print(f"  Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")

    return merged


# ==============================================================
# STEP 4: Imputation
# ==============================================================

def impute_missing(df):
    """
    Handle missing values using multiple imputation strategies.

    """
    print("\n" + "=" * 60)
    print("STEP 4: Imputation")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Report before
    total_missing = df[numeric_cols].isnull().sum().sum()
    print(f"  Total missing values (numeric): {total_missing}")

    # ---- 4a. Simple Imputation ----
    # For stats columns, older matches simply don't have data.
    # We'll use median imputation for numeric stats.
    stat_cols = ['home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
                 'home_fouls', 'away_fouls', 'home_corners', 'away_corners',
                 'home_yellows', 'away_yellows', 'home_reds', 'away_reds',
                 'total_shots', 'total_fouls', 'total_cards',
                 'ht_home_goals', 'ht_away_goals']
    stat_cols = [c for c in stat_cols if c in df.columns]

    if stat_cols:
        available_stats = df[stat_cols].dropna(how='all')
        if len(available_stats) > 0:
            # Simple median imputation
            simple_imputer = SimpleImputer(strategy='median')
            df[stat_cols] = simple_imputer.fit_transform(df[stat_cols])
            print(f"  Applied median imputation to {len(stat_cols)} stat columns")

    # ---- 4b. KNN Imputation (for demonstration) ----
    # KNN uses similar matches to estimate missing values
    knn_cols = ['home_goals', 'away_goals', 'total_goals']
    knn_cols = [c for c in knn_cols if c in df.columns and df[c].isnull().any()]

    if knn_cols:
        knn_imputer = KNNImputer(n_neighbors=5)
        df[knn_cols] = knn_imputer.fit_transform(df[knn_cols])
        print(f"  Applied KNN imputation to: {knn_cols}")

    # Report after
    total_after = df[numeric_cols].isnull().sum().sum()
    print(f"  Missing values remaining (numeric): {total_after}")

    return df


# ==============================================================
# STEP 5: Feature Engineering
# ==============================================================

def engineer_features(df):
    """
    Create new features for analysis and ML.

    """
    print("\n" + "=" * 60)
    print("STEP 5: Feature Engineering")
    print("=" * 60)

    # ---- 5a. Continuous to Categorical ----
    def goal_category(total):
        if pd.isna(total):
            return 'Unknown'
        total = int(total)
        if total == 0:
            return 'Goalless'
        elif total <= 2:
            return 'Low'
        elif total <= 4:
            return 'Medium'
        else:
            return 'High'

    df['goal_category'] = df['total_goals'].apply(goal_category)
    print(f"  Created goal_category: {df['goal_category'].value_counts().to_dict()}")

    # ---- 5b. Decade feature ----
    df['decade'] = df['year'].apply(
        lambda y: f"{int(y)//10*10}s" if pd.notna(y) else None
    )

    # ---- 5c. Venue feature (who is home) ----
    def get_venue_for_rm(row):
        if row['home_team'] == 'Real Madrid':
            return 'Home'
        elif row['away_team'] == 'Real Madrid':
            return 'Away'
        return 'Neutral'

    df['rm_venue'] = df.apply(get_venue_for_rm, axis=1)

    # ---- 5d. RM result ----
    def rm_result(row):
        if pd.isna(row['result']):
            return None
        if row['result'] == 'D':
            return 'Draw'
        if row['home_team'] == 'Real Madrid':
            return 'RM Win' if row['result'] == 'H' else 'RM Loss'
        else:
            return 'RM Win' if row['result'] == 'A' else 'RM Loss'

    df['rm_result'] = df.apply(rm_result, axis=1)

    # ---- 5e. RM and Barca goals regardless of home/away ----
    def get_rm_goals(row):
        if row['home_team'] == 'Real Madrid':
            return row['home_goals']
        return row['away_goals']

    def get_barca_goals(row):
        if row['home_team'] == 'Barcelona':
            return row['home_goals']
        return row['away_goals']

    df['rm_goals'] = df.apply(get_rm_goals, axis=1)
    df['barca_goals'] = df.apply(get_barca_goals, axis=1)

    # ---- 5f. Encoding categorical variables ----
    le_result = LabelEncoder()
    df['result_encoded'] = le_result.fit_transform(df['result'].fillna('Unknown'))

    le_venue = LabelEncoder()
    df['rm_venue_encoded'] = le_venue.fit_transform(df['rm_venue'])

    print(f"  Created {6} new features")
    print(f"  Encoded result: {dict(zip(le_result.classes_, le_result.transform(le_result.classes_)))}")

    return df


# ==============================================================
# STEP 6: Outlier Detection
# ==============================================================

def detect_outliers(df):
    """
    Detect outlier matches using multiple methods.

    """
    print("\n" + "=" * 60)
    print("STEP 6: Outlier Detection")
    print("=" * 60)

    # ---- 6a. Z-Score ----
    if df['total_goals'].notna().sum() > 0:
        z_scores = np.abs(stats.zscore(df['total_goals'].dropna()))
        z_outliers = (z_scores > 2.5).sum()
        df['is_outlier_zscore'] = False
        valid_mask = df['total_goals'].notna()
        df.loc[valid_mask, 'is_outlier_zscore'] = np.abs(
            stats.zscore(df.loc[valid_mask, 'total_goals'])
        ) > 2.5
        print(f"  Z-Score outliers (total_goals > 2.5 std): {df['is_outlier_zscore'].sum()}")

        outlier_matches = df[df['is_outlier_zscore']][['date_str', 'home_team', 'away_team', 'home_goals', 'away_goals', 'total_goals']]
        if len(outlier_matches) > 0:
            print(f"  Outlier matches:")
            for _, row in outlier_matches.iterrows():
                print(f"    {row['date_str']}: {row['home_team']} {int(row['home_goals'])}-{int(row['away_goals'])} {row['away_team']}")

    # ---- 6b. IQR ----
    Q1 = df['total_goals'].quantile(0.25)
    Q3 = df['total_goals'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df['is_outlier_iqr'] = (df['total_goals'] < lower) | (df['total_goals'] > upper)
    print(f"\n  IQR method (Q1={Q1}, Q3={Q3}, IQR={IQR}):")
    print(f"    Bounds: [{lower:.1f}, {upper:.1f}]")
    print(f"    Outliers: {df['is_outlier_iqr'].sum()}")

    # ---- 6c. K-Means Clustering ----
    cluster_cols = ['home_goals', 'away_goals']
    valid = df[cluster_cols].dropna()

    if len(valid) >= 3:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(valid)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Distance to nearest cluster center
        distances = kmeans.transform(X_scaled).min(axis=1)
        threshold = np.percentile(distances, 95)

        df['kmeans_cluster'] = np.nan
        df.loc[valid.index, 'kmeans_cluster'] = clusters

        df['is_outlier_kmeans'] = False
        df.loc[valid.index, 'is_outlier_kmeans'] = distances > threshold

        print(f"\n  K-Means clustering (k=3):")
        print(f"    Cluster sizes: {pd.Series(clusters).value_counts().to_dict()}")
        print(f"    Outliers (95th percentile distance): {df['is_outlier_kmeans'].sum()}")

    return df


# ==============================================================
# STEP 7: Normalization & Standardization
# ==============================================================

def normalize_and_standardize(df):
    """
    Apply normalization and standardization to numeric columns.

    """
    print("\n" + "=" * 60)
    print("STEP 7: Normalization & Standardization")
    print("=" * 60)

    cols_to_scale = ['home_goals', 'away_goals', 'total_goals',
                     'home_shots', 'away_shots',
                     'home_fouls', 'away_fouls']
    cols_to_scale = [c for c in cols_to_scale if c in df.columns]

    # Min-Max Normalization (0 to 1)
    min_max = MinMaxScaler()
    for col in cols_to_scale:
        valid = df[col].notna()
        if valid.sum() > 0:
            df[f'{col}_normalized'] = np.nan
            df.loc[valid, f'{col}_normalized'] = min_max.fit_transform(
                df.loc[valid, [col]]
            ).flatten()

    # Z-Score Standardization (mean=0, std=1)
    std_scaler = StandardScaler()
    for col in cols_to_scale:
        valid = df[col].notna()
        if valid.sum() > 0:
            df[f'{col}_standardized'] = np.nan
            df.loc[valid, f'{col}_standardized'] = std_scaler.fit_transform(
                df.loc[valid, [col]]
            ).flatten()

    print(f"  Normalized and standardized {len(cols_to_scale)} columns")
    print(f"  Added {len(cols_to_scale) * 2} new columns")

    return df


# ==============================================================
# STEP 8: Data Characteristics
# ==============================================================

def compute_statistics(df):
    """
    Compute mean, variance, skewness, kurtosis for key columns.

    """
    print("\n" + "=" * 60)
    print("STEP 8: Data Characteristics")
    print("=" * 60)

    key_cols = ['total_goals', 'home_goals', 'away_goals', 'goal_diff',
                'rm_goals', 'barca_goals']
    key_cols = [c for c in key_cols if c in df.columns]

    stats_data = []
    for col in key_cols:
        s = df[col].dropna()
        stats_data.append({
            'column': col,
            'count': len(s),
            'mean': round(s.mean(), 3),
            'median': round(s.median(), 3),
            'std': round(s.std(), 3),
            'variance': round(s.var(), 3),
            'skewness': round(s.skew(), 3),
            'kurtosis': round(s.kurtosis(), 3),
            'min': s.min(),
            'max': s.max(),
        })

    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))

    return stats_df


# ==============================================================
# MAIN PIPELINE
# ==============================================================

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    final_dir = os.path.join(project_root, 'data', 'final')

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    print("\n" + "#" * 60)
    print("# EL CLASICO DATA CLEANING & MERGING PIPELINE")
    print("#" * 60)

    # Step 1: Clean Wikipedia
    wiki_path = os.path.join(raw_dir, 'wikipedia_el_clasico_bs4.csv')
    wiki_clean = clean_wikipedia(wiki_path)
    wiki_clean.to_csv(os.path.join(processed_dir, 'wikipedia_cleaned.csv'), index=False)

    # Step 2: Clean Football-Data
    fd_path = os.path.join(raw_dir, 'football_data_el_clasico.csv')
    fd_clean = clean_football_data(fd_path)
    fd_clean.to_csv(os.path.join(processed_dir, 'football_data_cleaned.csv'), index=False)

    # Step 3: Merge
    merged = merge_datasets(wiki_clean, fd_clean)
    merged.to_csv(os.path.join(processed_dir, 'merged_raw.csv'), index=False)

    # Step 4: Imputation
    merged = impute_missing(merged)

    # Step 5: Feature Engineering
    merged = engineer_features(merged)

    # Step 6: Outlier Detection
    merged = detect_outliers(merged)

    # Step 7: Normalization
    merged = normalize_and_standardize(merged)

    # Step 8: Statistics
    stats_df = compute_statistics(merged)
    stats_df.to_csv(os.path.join(final_dir, 'data_statistics.csv'), index=False)

    # Save final dataset
    final_path = os.path.join(final_dir, 'el_clasico_final.csv')
    merged.to_csv(final_path, index=False, encoding='utf-8')

    # Also save as JSON
    json_path = os.path.join(final_dir, 'el_clasico_final.json')
    merged.to_json(json_path, orient='records', indent=2, force_ascii=False)

    print("\n" + "#" * 60)
    print("# PIPELINE COMPLETE")
    print("#" * 60)
    print(f"\n  Final dataset: {merged.shape[0]} rows x {merged.shape[1]} columns")
    print(f"\n  Files saved:")
    print(f"    {os.path.join(processed_dir, 'wikipedia_cleaned.csv')}")
    print(f"    {os.path.join(processed_dir, 'football_data_cleaned.csv')}")
    print(f"    {os.path.join(processed_dir, 'merged_raw.csv')}")
    print(f"    {final_path}")
    print(f"    {json_path}")
    print(f"    {os.path.join(final_dir, 'data_statistics.csv')}")

    print(f"\n  Column list:")
    for i, col in enumerate(merged.columns):
        print(f"    {i+1:2d}. {col}")

    return merged


if __name__ == '__main__':
    df = main()
