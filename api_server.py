from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson

app = Flask(__name__)
CORS(app)

# Load data and models
df = pd.read_csv("Football_Data_Cleaned_2026.csv")
home_model = joblib.load("home_goals_model.pkl")
away_model = joblib.load("away_goals_model.pkl")
feature_cols = joblib.load("model_features.pkl")

# Build team stats
team_stats = {}
for team in pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique():
    home_games = df[df["HomeTeam"] == team]
    away_games = df[df["AwayTeam"] == team]
    all_games = pd.concat([home_games, away_games])
    if len(all_games) == 0:
        continue
    team_stats[team] = {
        "elo": float(
            pd.concat([home_games["HomeElo"], away_games["AwayElo"]]).mean()
        ),
        "form3": float(
            pd.concat([home_games["Form3Home"], away_games["Form3Away"]]).mean()
        ),
        "form5": float(
            pd.concat([home_games["Form5Home"], away_games["Form5Away"]]).mean()
        ),
        "attack": float(home_games["Home_Attack_Strength"].mean())
        if len(home_games) > 0
        else 1.0,
        "defense": float(away_games["Away_Defense_Weakness"].mean())
        if len(away_games) > 0
        else 1.0,
        "games": len(all_games),
        "division": str(all_games["Division"].mode()[0])
        if len(all_games) > 0
        else "Unknown",
    }

# League mapping
LEAGUE_MAP = {
    # England
    "E0": {"name": "Premier League", "country": "England", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "E1": {"name": "Championship", "country": "England", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "E2": {"name": "League One", "country": "England", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "E3": {"name": "League Two", "country": "England", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    # Spain
    "SP1": {"name": "La Liga", "country": "Spain", "flag": "🇪🇸"},
    "SP2": {"name": "Segunda División", "country": "Spain", "flag": "🇪🇸"},
    # Italy
    "I1": {"name": "Serie A", "country": "Italy", "flag": "🇮🇹"},
    "I2": {"name": "Serie B", "country": "Italy", "flag": "🇮🇹"},
    # France
    "F1": {"name": "Ligue 1", "country": "France", "flag": "🇫🇷"},
    "F2": {"name": "Ligue 2", "country": "France", "flag": "🇫🇷"},
    # Germany
    "D1": {"name": "Bundesliga", "country": "Germany", "flag": "🇩🇪"},
    "D2": {"name": "2. Bundesliga", "country": "Germany", "flag": "🇩🇪"},
    # Portugal
    "P1": {"name": "Primeira Liga", "country": "Portugal", "flag": "🇵🇹"},
    # Belgium
    "B1": {"name": "Pro League", "country": "Belgium", "flag": "🇧🇪"},
    # Netherlands
    "N1": {"name": "Eredivisie", "country": "Netherlands", "flag": "🇳🇱"},
    # Turkey
    "T1": {"name": "Süper Lig", "country": "Turkey", "flag": "🇹🇷"},
    # Scotland
    "SC0": {"name": "Scottish Premiership", "country": "Scotland", "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    "SC1": {"name": "Scottish Championship", "country": "Scotland", "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    "SC2": {"name": "Scottish League One", "country": "Scotland", "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    "SC3": {"name": "Scottish League Two", "country": "Scotland", "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    # Other Europe
    "RUS": {"name": "Russian Premier League", "country": "Russia", "flag": "🇷🇺"},
    "ROM": {"name": "Liga 1", "country": "Romania", "flag": "🇷🇴"},
    "POL": {"name": "Ekstraklasa", "country": "Poland", "flag": "🇵🇱"},
    "SWE": {"name": "Allsvenskan", "country": "Sweden", "flag": "🇸🇪"},
    "NOR": {"name": "Eliteserien", "country": "Norway", "flag": "🇳🇴"},
    "DEN": {"name": "Superliga", "country": "Denmark", "flag": "🇩🇰"},
    "FIN": {"name": "Veikkausliiga", "country": "Finland", "flag": "🇫🇮"},
    "AUT": {"name": "Austrian Bundesliga", "country": "Austria", "flag": "🇦🇹"},
    "SUI": {"name": "Super League", "country": "Switzerland", "flag": "🇨🇭"},
    "IRL": {"name": "League of Ireland", "country": "Ireland", "flag": "🇮🇪"},
    "G1": {"name": "Super League", "country": "Greece", "flag": "🇬🇷"},
    # Europe Competition
    "EC": {"name": "European Competition", "country": "Europe", "flag": "🇪🇺"},
    # Americas
    "USA": {"name": "MLS", "country": "USA", "flag": "🇺🇸"},
    "ARG": {"name": "Primera División", "country": "Argentina", "flag": "🇦🇷"},
    "BRA": {"name": "Série A", "country": "Brazil", "flag": "🇧🇷"},
    "MEX": {"name": "Liga MX", "country": "Mexico", "flag": "🇲🇽"},
    # Asia
    "JAP": {"name": "J-League", "country": "Japan", "flag": "🇯🇵"},
    "CHN": {"name": "Super League", "country": "China", "flag": "🇨🇳"},
}


@app.route("/health")
def health():
    return jsonify({"status": "ok", "teams_loaded": len(team_stats)})


@app.route("/teams")
def teams():
    return jsonify({"teams": sorted(team_stats.keys())})


@app.route("/team/<name>")
def team(name):
    if name not in team_stats:
        return jsonify({"error": "Team not found"}), 404
    return jsonify({"team": name, "stats": team_stats[name]})


@app.route("/leagues")
def leagues():
    result = []
    divisions_in_data = df["Division"].unique()

    for div_code in sorted(divisions_in_data):
        # Get teams in this division
        teams_in_div = df[df["Division"] == div_code]["HomeTeam"].unique().tolist()
        teams_in_div += df[df["Division"] == div_code]["AwayTeam"].unique().tolist()
        teams_in_div = sorted(list(set(teams_in_div)))

        info = LEAGUE_MAP.get(
            div_code,
            {"name": div_code, "country": "Unknown", "flag": "🌍"},
        )

        result.append(
            {
                "code": div_code,
                "name": info["name"],
                "country": info["country"],
                "flag": info["flag"],
                "teams": teams_in_div,
                "teams_count": len(teams_in_div),
            }
        )

    return jsonify({"leagues": result, "total": len(result)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        home_team = data["home_team"]
        away_team = data["away_team"]

        if home_team not in team_stats or away_team not in team_stats:
            return jsonify({"error": "Team not found"}), 404

        hs = team_stats[home_team]
        as_ = team_stats[away_team]

        features = {
            "HomeElo": hs["elo"],
            "AwayElo": as_["elo"],
            "Form3Home": hs["form3"],
            "Form5Home": hs["form5"],
            "Form3Away": as_["form3"],
            "Form5Away": as_["form5"],
            "OddHome": 2.0,
            "OddDraw": 3.2,
            "OddAway": 3.5,
            "Over25": 0.55,
            "Under25": 0.45,
            "HandiSize": 0.0,
            "Elo_Diff": hs["elo"] - as_["elo"],
            "Form3_Diff": hs["form3"] - as_["form3"],
            "Form5_Diff": hs["form5"] - as_["form5"],
            "Month": 5,
            "DayOfWeek": 5,
            "Home_Attack_Strength": hs["attack"],
            "Away_Defense_Weakness": as_["defense"],
        }

        X = pd.DataFrame([[features.get(f, 0) for f in feature_cols]], columns=feature_cols)
        xg_home = float(max(0.1, home_model.predict(X)[0]))
        xg_away = float(max(0.1, away_model.predict(X)[0]))

        # Poisson matrix
        max_goals = 6
        home_probs = [poisson.pmf(i, xg_home) for i in range(max_goals + 1)]
        away_probs = [poisson.pmf(i, xg_away) for i in range(max_goals + 1)]

        matrix = []
        win_home = draw = win_away = 0.0
        top_scores = []

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = home_probs[i] * away_probs[j]
                matrix.append({"home": i, "away": j, "prob": round(p * 100, 2)})
                top_scores.append((i, j, p))
                if i > j:
                    win_home += p
                elif i == j:
                    draw += p
                else:
                    win_away += p

        top_scores.sort(key=lambda x: -x[2])
        top_5 = [{"score": f"{s[0]}-{s[1]}", "prob": round(s[2] * 100, 2)} for s in top_scores[:5]]

        return jsonify({
            "home_team": home_team,
            "away_team": away_team,
            "xg_home": round(xg_home, 2),
            "xg_away": round(xg_away, 2),
            "win_prob": {
                "home": round(win_home * 100, 1),
                "draw": round(draw * 100, 1),
                "away": round(win_away * 100, 1),
            },
            "top_scores": top_5,
            "poisson_matrix": matrix,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
