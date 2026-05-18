import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path("./data_processed")
HISTORICAL_FILE = Path("./world_cup_last_30_years.csv")
FORECAST_FILE = DATA_DIR / "tournament_forecast_2026.csv"

st.set_page_config(
    page_title="T20 World Cup Forecast",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 T20 World Cup 2026 Forecast Explorer")
st.write(
    "Explore historical cricket data, model forecasts, and tournament probability insights "
    "from the existing pipeline."
)


@st.cache_data
def load_data():
    data = None
    forecast = None

    if HISTORICAL_FILE.exists():
        data = pd.read_csv(HISTORICAL_FILE, parse_dates=["date"])
    if FORECAST_FILE.exists():
        forecast = pd.read_csv(FORECAST_FILE)

    return data, forecast


data, forecast = load_data()

@st.cache_data
def get_latest_team_elo(df):
    df_sorted = df.sort_values("date")
    ratings = {}
    for _, row in df_sorted.iterrows():
        ratings[row["team1"]] = row.get("elo_team1", 1500.0)
        ratings[row["team2"]] = row.get("elo_team2", 1500.0)
    return ratings

@st.cache_resource
def train_match_predictor(df):
    df_model = df.copy()
    df_model = df_model[df_model["match_result"] == "completed"].copy()
    df_model["team1_won"] = (df_model["winner"] == df_model["team1"]).astype(int)
    df_model["toss_won_by_team1"] = (df_model["toss_winner"] == df_model["team1"]).astype(int)
    df_model["toss_bat_first"] = (df_model["toss_decision"] == "bat").astype(int)

    feature_cols = [
        "elo_team1", "elo_team2", "elo_diff",
        "team1_form_5", "team2_form_5",
        "team1_form_10", "team2_form_10",
        "h2h_win_pct", "toss_won_by_team1", "toss_bat_first"
    ]
    df_model = df_model.dropna(subset=feature_cols + ["team1_won"])

    model = LogisticRegression(max_iter=1000)
    model.fit(df_model[feature_cols], df_model["team1_won"])
    return model


def team_recent_form(df, team, window):
    team_matches = df[(df["team1"] == team) | (df["team2"] == team)].sort_values("date")
    if team_matches.empty:
        return 0.5
    last_matches = team_matches.tail(window)
    wins = ((last_matches["winner"] == team).astype(int)).sum()
    return wins / max(1, len(last_matches))


def team_head_to_head(df, team1, team2):
    h2h = df[((df["team1"] == team1) & (df["team2"] == team2)) | ((df["team1"] == team2) & (df["team2"] == team1))]
    if h2h.empty:
        return 0.5
    wins_for_team1 = ((h2h["winner"] == team1).astype(int)).sum()
    return wins_for_team1 / max(1, len(h2h))


def prepare_match_features(team1, team2, toss_winner, toss_decision, data, ratings):
    elo1 = ratings.get(team1, 1500.0)
    elo2 = ratings.get(team2, 1500.0)
    elo_diff = elo1 - elo2
    team1_form_5 = team_recent_form(data, team1, 5)
    team2_form_5 = team_recent_form(data, team2, 5)
    team1_form_10 = team_recent_form(data, team1, 10)
    team2_form_10 = team_recent_form(data, team2, 10)
    h2h_pct = team_head_to_head(data, team1, team2)
    toss_won_by_team1 = 0.5
    if toss_winner == team1:
        toss_won_by_team1 = 1
    elif toss_winner == team2:
        toss_won_by_team1 = 0
    toss_bat_first = 0.5
    if toss_decision == "bat":
        toss_bat_first = 1
    elif toss_decision == "field":
        toss_bat_first = 0

    return np.array([[
        elo1, elo2, elo_diff,
        team1_form_5, team2_form_5,
        team1_form_10, team2_form_10,
        h2h_pct, toss_won_by_team1, toss_bat_first
    ]])


page = st.sidebar.selectbox("Select page", ["Overview", "Historical Data", "Forecast", "Match Predictor", "About"])

if page == "Overview":
    st.header("Project overview")
    st.markdown(
        "This app presents a Streamlit interface for your T20 World Cup prediction project. "
        "It loads processed historical match data and the tournament forecast produced by the notebook."
    )

    if data is None:
        st.warning("Historical dataset not found. Run the notebook first or place `world_cup_last_30_years.csv` in the project root.")
    else:
        st.subheader("Historical dataset summary")
        st.write(f"Rows: {len(data):,}")
        st.write(f"Columns: {len(data.columns)}")
        st.write(data.describe(include="all"))

        st.markdown("---")
        st.subheader("Key statistics")
        st.write(
            data[["season", "team1", "team2", "winner", "venue", "city"]]
            .head(10)
        )

    if forecast is None:
        st.warning("Forecast dataset not found. Run the notebook first or place `data_processed/tournament_forecast_2026.csv` in the data_processed folder.")
    else:
        st.subheader("Top forecast probabilities")
        st.bar_chart(forecast.set_index("Team")["Title_Probability"].head(10))
        st.write(forecast.head(10))

elif page == "Historical Data":
    st.header("Historical match data")
    if data is None:
        st.warning("Historical dataset not found.")
    else:
        st.write(data.head(20))
        st.subheader("Filter by season")
        seasons = sorted(data["season"].unique())
        selected_season = st.selectbox("Season", seasons)
        filtered = data[data["season"] == selected_season]
        st.write(filtered)

        st.subheader("Match count by season")
        counts = data["season"].value_counts().sort_index()
        st.bar_chart(counts)

        st.subheader("Top teams by appearances")
        team_counts = pd.concat([data["team1"], data["team2"]]).value_counts().head(15)
        st.bar_chart(team_counts)

elif page == "Forecast":
    st.header("2026 World Cup Forecast")
    if forecast is None:
        st.warning("Forecast dataset not found.")
    else:
        st.write(forecast)
        st.subheader("Championship probability chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        chart_data = forecast.head(10).sort_values("Title_Probability")
        ax.barh(chart_data["Team"], chart_data["Title_Probability"], color="#f4c542")
        ax.set_xlabel("Title probability (%)")
        ax.set_title("Top 10 teams by championship probability")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Forecast details")
        st.write(
            "This forecast is generated from your model pipeline and Monte Carlo simulation. "
            "Use the notebook to retrain the model or update tournament fixtures."
        )

elif page == "Match Predictor":
    st.header("Match Predictor")
    if data is None:
        st.warning("Historical dataset not found.")
    else:
        teams = sorted(pd.unique(pd.concat([data["team1"], data["team2"]])))
        team1 = st.selectbox("Select Team 1", teams, index=teams.index("India") if "India" in teams else 0)
        team2 = st.selectbox("Select Team 2", [team for team in teams if team != team1])
        toss_winner = st.selectbox("Toss winner", [team1, team2, "Unknown"])
        toss_decision = st.selectbox("Toss decision", ["bat", "field", "Unknown"])

        st.markdown("---")
        st.subheader("Prediction inputs")
        st.write(f"Team 1: **{team1}**")
        st.write(f"Team 2: **{team2}**")
        st.write(f"Toss winner: **{toss_winner}**")
        st.write(f"Toss decision: **{toss_decision}**")

        if st.button("Predict match winner"):
            if team1 == team2:
                st.warning("Please select two different teams.")
            else:
                model = train_match_predictor(data)
                ratings = get_latest_team_elo(data)
                features = prepare_match_features(team1, team2, toss_winner, toss_decision, data, ratings)
                prob_team1 = model.predict_proba(features)[0, 1]
                prob_team2 = 1 - prob_team1
                predicted_winner = team1 if prob_team1 >= 0.5 else team2

                st.success(f"Predicted winner: **{predicted_winner}**")
                st.write(f"Probability {team1} wins: **{prob_team1 * 100:.1f}%**")
                st.write(f"Probability {team2} wins: **{prob_team2 * 100:.1f}%**")

                st.markdown("---")
                st.subheader("Computed features")
                st.write({
                    "Team 1 ELO": ratings.get(team1, 1500.0),
                    "Team 2 ELO": ratings.get(team2, 1500.0),
                    "ELO diff (team1 - team2)": ratings.get(team1, 1500.0) - ratings.get(team2, 1500.0),
                    "Team 1 recent win pct (last 5)": f"{team_recent_form(data, team1, 5) * 100:.1f}%",
                    "Team 2 recent win pct (last 5)": f"{team_recent_form(data, team2, 5) * 100:.1f}%",
                    "Team 1 recent win pct (last 10)": f"{team_recent_form(data, team1, 10) * 100:.1f}%",
                    "Team 2 recent win pct (last 10)": f"{team_recent_form(data, team2, 10) * 100:.1f}%",
                    "Head-to-head win pct for Team 1": f"{team_head_to_head(data, team1, team2) * 100:.1f}%"
                })

elif page == "About":
    st.header("How to use this Streamlit app")
    st.markdown(
        "1. Install dependencies with `pip install -r requirements.txt`\n"
        "2. Run `streamlit run streamlit_app.py`\n"
        "3. Open the local URL that Streamlit prints in the terminal\n"
        "4. If data is missing, run `train.ipynb` first to produce the CSV files."
    )
    st.markdown("### Deployment options")
    st.markdown(
        "- Streamlit Cloud: easiest for public sharing\n"
        "- Render: can host a Streamlit app from GitHub\n"
        "- Railway: another fast deploy option\n"
    )
    st.markdown("### Notes")
    st.write(
        "This app currently reads the saved CSV outputs from the notebook. "
        "For a production deployment, the notebook logic should be refactored into reusable Python modules."
    )
