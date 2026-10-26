"""
El Clasico All-Time Dashboard
==============================
Interactive Streamlit dashboard for El Clasico data analysis.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="El Clasico Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

RM_COLOR = "#FEBE10"
BARCA_COLOR = "#A50044"
DRAW_COLOR = "#808080"


@st.cache_data
def load_data():
    df = pd.read_csv("data/final/el_clasico_final.csv")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


df = load_data()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

st.sidebar.title("El Clasico Analysis")
st.sidebar.markdown("Real Madrid vs Barcelona")
st.sidebar.markdown(f"**{len(df)}** matches | **{df['year'].min():.0f} - {df['year'].max():.0f}**")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Head to Head", "Match Stats", "Trends", "Outliers", "Predict"],
)

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

year_range = st.sidebar.slider(
    "Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max())),
)

venue_filter = st.sidebar.multiselect(
    "RM Venue", ["Home", "Away"], default=["Home", "Away"]
)

# Apply filters
mask = (
    (df["year"] >= year_range[0])
    & (df["year"] <= year_range[1])
    & (df["rm_venue"].isin(venue_filter))
)
filtered = df[mask].copy()

st.sidebar.markdown(f"Showing **{len(filtered)}** / {len(df)} matches")


# ──────────────────────────────────────────────
# Page: Overview
# ──────────────────────────────────────────────

if page == "Overview":
    st.title("El Clasico: All-Time Overview")

    rm_wins = (filtered["rm_result"] == "RM Win").sum()
    barca_wins = (filtered["rm_result"] == "RM Loss").sum()
    draws = (filtered["rm_result"] == "Draw").sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Matches", len(filtered))
    col2.metric("RM Wins", rm_wins)
    col3.metric("Draws", draws)
    col4.metric("Barca Wins", barca_wins)
    col5.metric("Total Goals", int(filtered["total_goals"].sum()))

    col6, col7, col8, col9 = st.columns(4)
    col6.metric("Avg Goals/Match", f"{filtered['total_goals'].mean():.2f}")
    col7.metric("RM Avg Goals", f"{filtered['rm_goals'].mean():.2f}")
    col8.metric("Barca Avg Goals", f"{filtered['barca_goals'].mean():.2f}")
    col9.metric("Highest Scoring", f"{filtered['total_goals'].max():.0f} goals")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("All-Time Win Record")
        fig, ax = plt.subplots(figsize=(6, 6))
        sizes = [rm_wins, draws, barca_wins]
        labels = [f"RM Win\n{rm_wins}", f"Draw\n{draws}", f"Barca Win\n{barca_wins}"]
        colors = [RM_COLOR, DRAW_COLOR, BARCA_COLOR]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 12, "fontweight": "bold"},
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        st.pyplot(fig)

    with right:
        st.subheader("Goals Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered["total_goals"], bins=range(0, 12), kde=True,
                     color="#2196F3", edgecolor="black", alpha=0.7, ax=ax)
        ax.axvline(filtered["total_goals"].mean(), color="red", linestyle="--",
                   linewidth=2, label=f"Mean = {filtered['total_goals'].mean():.2f}")
        ax.set_xlabel("Total Goals")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Recent Matches")
    recent = filtered.dropna(subset=["date_str"]).sort_values("date_str", ascending=False).head(15)
    display_cols = ["date_str", "home_team", "home_goals", "away_goals", "away_team", "winner", "total_goals"]
    display_cols = [c for c in display_cols if c in recent.columns]
    st.dataframe(recent[display_cols].reset_index(drop=True), use_container_width=True)


# ──────────────────────────────────────────────
# Page: Head to Head
# ──────────────────────────────────────────────

elif page == "Head to Head":
    st.title("Head to Head Comparison")

    left, right = st.columns(2)

    with left:
        st.subheader("Goals by Decade")
        decade_df = filtered.dropna(subset=["decade"])
        decade_stats = decade_df.groupby("decade").agg(
            rm_avg=("rm_goals", "mean"),
            barca_avg=("barca_goals", "mean"),
            count=("rm_goals", "count"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(decade_stats))
        w = 0.35
        ax.bar(x - w / 2, decade_stats["rm_avg"], w, label="Real Madrid", color=RM_COLOR, edgecolor="black")
        ax.bar(x + w / 2, decade_stats["barca_avg"], w, label="Barcelona", color=BARCA_COLOR, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(decade_stats["decade"], rotation=45)
        ax.set_ylabel("Avg Goals")
        ax.legend()
        for i, c in enumerate(decade_stats["count"]):
            ax.text(i, max(decade_stats["rm_avg"].iloc[i], decade_stats["barca_avg"].iloc[i]) + 0.1,
                    f"n={c}", ha="center", fontsize=8, color="gray")
        st.pyplot(fig)

    with right:
        st.subheader("Win Rate by Decade")
        decade_results = filtered.dropna(subset=["decade", "rm_result"]).groupby(
            ["decade", "rm_result"]
        ).size().unstack(fill_value=0)
        decade_pct = decade_results.div(decade_results.sum(axis=1), axis=0) * 100
        col_order = [c for c in ["RM Win", "Draw", "RM Loss"] if c in decade_pct.columns]
        decade_pct = decade_pct[col_order]

        fig, ax = plt.subplots(figsize=(10, 5))
        decade_pct.plot(kind="bar", stacked=True, color=[RM_COLOR, DRAW_COLOR, BARCA_COLOR], ax=ax, edgecolor="black")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(fontsize=9)
        st.pyplot(fig)

    st.markdown("---")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Home Advantage: Real Madrid")
        rm_home = filtered[filtered["home_team"] == "Real Madrid"]
        rm_away = filtered[filtered["away_team"] == "Real Madrid"]
        home_w = len(rm_home[rm_home["result"] == "H"])
        home_d = len(rm_home[rm_home["result"] == "D"])
        home_l = len(rm_home[rm_home["result"] == "A"])
        away_w = len(rm_away[rm_away["result"] == "A"])
        away_d = len(rm_away[rm_away["result"] == "D"])
        away_l = len(rm_away[rm_away["result"] == "H"])

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(3)
        ax.bar(x - 0.175, [home_w, home_d, home_l], 0.35, label="Home", color=RM_COLOR, edgecolor="black")
        ax.bar(x + 0.175, [away_w, away_d, away_l], 0.35, label="Away", color=RM_COLOR, alpha=0.4, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["Win", "Draw", "Loss"])
        ax.legend()
        for c in ax.containers:
            ax.bar_label(c, fontsize=10, fontweight="bold")
        st.pyplot(fig)

    with right2:
        st.subheader("Home Advantage: Barcelona")
        bc_home = filtered[filtered["home_team"] == "Barcelona"]
        bc_away = filtered[filtered["away_team"] == "Barcelona"]
        bh_w = len(bc_home[bc_home["result"] == "H"])
        bh_d = len(bc_home[bc_home["result"] == "D"])
        bh_l = len(bc_home[bc_home["result"] == "A"])
        ba_w = len(bc_away[bc_away["result"] == "A"])
        ba_d = len(bc_away[bc_away["result"] == "D"])
        ba_l = len(bc_away[bc_away["result"] == "H"])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.175, [bh_w, bh_d, bh_l], 0.35, label="Home", color=BARCA_COLOR, edgecolor="black")
        ax.bar(x + 0.175, [ba_w, ba_d, ba_l], 0.35, label="Away", color=BARCA_COLOR, alpha=0.4, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["Win", "Draw", "Loss"])
        ax.legend()
        for c in ax.containers:
            ax.bar_label(c, fontsize=10, fontweight="bold")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Cumulative Wins Over Time")
    timeline = filtered.dropna(subset=["year", "rm_result"]).sort_values("year")
    timeline["rm_cum"] = (timeline["rm_result"] == "RM Win").cumsum()
    timeline["barca_cum"] = (timeline["rm_result"] == "RM Loss").cumsum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timeline["year"], timeline["rm_cum"], linewidth=2.5, color=RM_COLOR, label="Real Madrid")
    ax.plot(timeline["year"], timeline["barca_cum"], linewidth=2.5, color=BARCA_COLOR, label="Barcelona")
    ax.fill_between(timeline["year"], timeline["rm_cum"], timeline["barca_cum"],
                    where=timeline["rm_cum"] >= timeline["barca_cum"], alpha=0.15, color=RM_COLOR)
    ax.fill_between(timeline["year"], timeline["rm_cum"], timeline["barca_cum"],
                    where=timeline["rm_cum"] < timeline["barca_cum"], alpha=0.15, color=BARCA_COLOR)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Wins")
    ax.legend(fontsize=12)
    st.pyplot(fig)


# ──────────────────────────────────────────────
# Page: Match Stats
# ──────────────────────────────────────────────

elif page == "Match Stats":
    st.title("Match Statistics")

    left, right = st.columns(2)

    with left:
        st.subheader("Shots vs Goals Correlation")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(x="total_shots", y="total_goals", data=filtered, ax=ax,
                    scatter_kws={"alpha": 0.5, "color": "#FF9800"}, line_kws={"color": "red"})
        corr = filtered["total_shots"].corr(filtered["total_goals"])
        ax.set_title(f"r = {corr:.3f}")
        ax.set_xlabel("Total Shots")
        ax.set_ylabel("Total Goals")
        st.pyplot(fig)

    with right:
        st.subheader("Correlation Heatmap")
        corr_cols = ["home_goals", "away_goals", "total_goals",
                     "home_shots", "away_shots", "home_fouls", "away_fouls",
                     "home_corners", "away_corners", "home_yellows", "away_yellows"]
        corr_matrix = filtered[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 7))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax, annot_kws={"fontsize": 7})
        st.pyplot(fig)

    st.markdown("---")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Corners: Home vs Away")
        fig, ax = plt.subplots(figsize=(8, 4))
        corner_melt = filtered[["home_corners", "away_corners"]].melt(var_name="Type", value_name="Corners")
        corner_melt["Type"] = corner_melt["Type"].map({"home_corners": "Home", "away_corners": "Away"})
        sns.violinplot(data=corner_melt, x="Type", y="Corners", hue="Type",
                       palette=["#4CAF50", "#FF5722"], inner="quart", legend=False, ax=ax)
        ax.set_ylabel("Corners")
        st.pyplot(fig)

    with right2:
        st.subheader("Goal Category Breakdown")
        cat_counts = filtered["goal_category"].value_counts()
        cat_order = ["Goalless", "Low", "Medium", "High"]
        cat_counts = cat_counts.reindex(cat_order).dropna()
        colors_cat = ["#9E9E9E", "#4CAF50", "#2196F3", "#F44336"]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(cat_counts.values, labels=cat_counts.index, colors=colors_cat,
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.4})
        ax.text(0, 0, f"{len(filtered)}\nMatches", ha="center", va="center", fontsize=14, fontweight="bold")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top 10 Biggest Wins")
    top = filtered.dropna(subset=["date_str"]).copy()
    top["abs_diff"] = top["goal_diff"].abs()
    top10 = top.nlargest(10, "abs_diff")[["date_str", "home_team", "home_goals", "away_goals", "away_team", "winner", "abs_diff"]]
    top10.columns = ["Date", "Home", "HG", "AG", "Away", "Winner", "Diff"]
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)


# ──────────────────────────────────────────────
# Page: Trends
# ──────────────────────────────────────────────

elif page == "Trends":
    st.title("Trends Over Time")

    st.subheader("Goals Per Match Over Time")
    decade_goals = filtered.dropna(subset=["decade"]).groupby("decade").agg(
        avg_goals=("total_goals", "mean"),
        avg_rm=("rm_goals", "mean"),
        avg_barca=("barca_goals", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(decade_goals["decade"], decade_goals["avg_goals"], "o-", color="#2196F3", linewidth=2, markersize=8, label="Total")
    ax.plot(decade_goals["decade"], decade_goals["avg_rm"], "s--", color=RM_COLOR, linewidth=1.5, markersize=6, label="Real Madrid")
    ax.plot(decade_goals["decade"], decade_goals["avg_barca"], "^--", color=BARCA_COLOR, linewidth=1.5, markersize=6, label="Barcelona")
    ax.set_ylabel("Avg Goals/Match")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Fouls and Cards Over Time")
    decade_disc = filtered.dropna(subset=["decade"]).groupby("decade").agg(
        avg_fouls=("total_fouls", "mean"),
        avg_cards=("total_cards", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(decade_disc["decade"], decade_disc["avg_fouls"], "o-", color="#F44336", linewidth=2, markersize=8, label="Fouls")
    ax1.set_ylabel("Avg Fouls", color="#F44336")
    ax2 = ax1.twinx()
    ax2.plot(decade_disc["decade"], decade_disc["avg_cards"], "s--", color="#FF9800", linewidth=2, markersize=8, label="Cards")
    ax2.set_ylabel("Avg Cards", color="#FF9800")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Monthly Distribution")
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    month_counts = filtered["month"].dropna().astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar([month_names.get(m, m) for m in month_counts.index],
                  month_counts.values, color="#2196F3", edgecolor="black")
    ax.bar_label(bars, fontweight="bold")
    ax.set_ylabel("Matches")
    st.pyplot(fig)


# ──────────────────────────────────────────────
# Page: Outliers
# ──────────────────────────────────────────────

elif page == "Outliers":
    st.title("Outlier Detection")

    st.subheader("Z-Score Outliers (Total Goals > 2.5 std)")
    outliers = filtered[filtered["is_outlier_zscore"] == True]
    normal = filtered[filtered["is_outlier_zscore"] == False]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(normal["year"], normal["total_goals"], c="#90CAF9", s=50, alpha=0.6, edgecolors="gray", label="Normal")
    ax.scatter(outliers["year"], outliers["total_goals"], c="#F44336", s=120, alpha=0.9,
               edgecolors="black", linewidth=1.5, zorder=5, label="Outlier")
    for _, row in outliers.iterrows():
        ax.annotate(f"{int(row['home_goals'])}-{int(row['away_goals'])}",
                    xy=(row["year"], row["total_goals"]),
                    xytext=(5, 10), textcoords="offset points",
                    fontsize=10, fontweight="bold", color="#F44336")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Goals")
    ax.legend()
    st.pyplot(fig)

    if len(outliers) > 0:
        st.subheader("Outlier Matches")
        out_cols = ["date_str", "home_team", "home_goals", "away_goals", "away_team", "total_goals"]
        out_cols = [c for c in out_cols if c in outliers.columns]
        st.dataframe(outliers[out_cols].reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    st.subheader("K-Means Clusters")
    cluster_df = filtered.dropna(subset=["kmeans_cluster"]).copy()
    cluster_df["kmeans_cluster"] = cluster_df["kmeans_cluster"].astype(int)

    cluster_means = cluster_df.groupby("kmeans_cluster")["total_goals"].mean().sort_values()
    rename = {cluster_means.index[i]: n for i, n in enumerate(["Low-scoring", "Balanced", "High-scoring"])}
    cluster_df["cluster_name"] = cluster_df["kmeans_cluster"].map(rename)

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, group in cluster_df.groupby("cluster_name"):
        ax.scatter(group["home_goals"], group["away_goals"], s=80, alpha=0.7,
                   edgecolors="black", linewidth=0.5, label=f"{name} (n={len(group)})")
    km_out = cluster_df[cluster_df["is_outlier_kmeans"]]
    if len(km_out) > 0:
        ax.scatter(km_out["home_goals"], km_out["away_goals"],
                   s=200, facecolors="none", edgecolors="red", linewidths=2, label="Outlier")
    ax.plot([0, 8], [0, 8], "k--", alpha=0.2)
    ax.set_xlabel("Home Goals")
    ax.set_ylabel("Away Goals")
    ax.set_aspect("equal")
    ax.legend()
    st.pyplot(fig)


# ──────────────────────────────────────────────
# Page: Predict
# ──────────────────────────────────────────────

elif page == "Predict":
    st.title("Predict the Next El Clasico")

    st.markdown("Uses a **Random Forest** model trained on historical rolling features.")

    @st.cache_resource
    def train_model():
        ml_df = df.dropna(subset=["year", "result"]).sort_values("year").reset_index(drop=True).copy()

        window = 5
        for col in ["rm_goals", "barca_goals", "total_goals", "home_goals", "away_goals"]:
            ml_df[f"{col}_roll{window}"] = ml_df[col].rolling(window=window, min_periods=1).mean().shift(1)

        ml_df["rm_win_flag"] = (ml_df["rm_result"] == "RM Win").astype(int)
        ml_df["barca_win_flag"] = (ml_df["rm_result"] == "RM Loss").astype(int)
        ml_df["draw_flag"] = (ml_df["rm_result"] == "Draw").astype(int)
        ml_df["rm_winrate_roll5"] = ml_df["rm_win_flag"].rolling(window=window, min_periods=1).mean().shift(1)
        ml_df["barca_winrate_roll5"] = ml_df["barca_win_flag"].rolling(window=window, min_periods=1).mean().shift(1)
        ml_df["draw_rate_roll5"] = ml_df["draw_flag"].rolling(window=window, min_periods=1).mean().shift(1)
        ml_df["rm_cum_wins"] = ml_df["rm_win_flag"].cumsum().shift(1).fillna(0)
        ml_df["barca_cum_wins"] = ml_df["barca_win_flag"].cumsum().shift(1).fillna(0)
        ml_df["match_index"] = range(len(ml_df))
        ml_df["rm_cum_winrate"] = (ml_df["rm_cum_wins"] / (ml_df["match_index"] + 1)).fillna(0)
        ml_df["is_rm_home"] = (ml_df["home_team"] == "Real Madrid").astype(int)

        features = [
            "year", "is_rm_home",
            "rm_goals_roll5", "barca_goals_roll5", "total_goals_roll5",
            "home_goals_roll5", "away_goals_roll5",
            "rm_winrate_roll5", "barca_winrate_roll5", "draw_rate_roll5",
            "rm_cum_winrate",
        ]

        ready = ml_df.dropna(subset=features).copy()
        X = ready[features].values
        le = LabelEncoder()
        y = le.fit_transform(ready["result"].values)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_scaled, y)

        latest = ready.iloc[-1]
        return model, scaler, le, features, latest, ready

    model, scaler, le, features, latest, ready = train_model()

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        is_rm_home = st.selectbox("Who is Home?", ["Real Madrid", "Barcelona"])
    with col2:
        year_input = st.number_input("Year", value=2026, min_value=2024, max_value=2030)

    # Build prediction input from latest rolling stats
    X_pred = latest[features].values.copy().astype(float)
    X_pred[0] = year_input
    X_pred[1] = 1 if is_rm_home == "Real Madrid" else 0

    X_pred_scaled = scaler.transform(X_pred.reshape(1, -1))
    prediction = le.inverse_transform(model.predict(X_pred_scaled))[0]
    probabilities = model.predict_proba(X_pred_scaled)[0]

    result_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    st.markdown("---")

    st.subheader("Prediction Result")
    pred_col1, pred_col2, pred_col3 = st.columns(3)

    for i, (cls, prob) in enumerate(zip(le.classes_, probabilities)):
        label = result_map.get(cls, cls)
        col = [pred_col1, pred_col2, pred_col3][i]
        highlight = "**" if cls == prediction else ""
        col.metric(f"{highlight}{label}{highlight}", f"{prob*100:.1f}%")

    predicted_label = result_map.get(prediction, prediction)
    if is_rm_home == "Real Madrid":
        if prediction == "H":
            winner = "Real Madrid"
        elif prediction == "A":
            winner = "Barcelona"
        else:
            winner = "Draw"
    else:
        if prediction == "H":
            winner = "Barcelona"
        elif prediction == "A":
            winner = "Real Madrid"
        else:
            winner = "Draw"

    st.success(f"Predicted outcome: **{predicted_label}** | **{winner}**")

    st.markdown("---")
    st.subheader("Feature Importance")
    fi = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(fi["Feature"], fi["Importance"], color="#4CAF50", edgecolor="black")
    ax.set_xlabel("Importance")
    st.pyplot(fig)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("El Clasico Analysis Project")
