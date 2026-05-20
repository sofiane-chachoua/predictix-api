from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson

app = Flask(__name__)
CORS(app)

# ── Load data and models ──────────────────────────────────────────
df           = pd.read_csv("Football_Data_Cleaned_2026.csv")
home_model   = joblib.load("home_goals_model.pkl")
away_model   = joblib.load("away_goals_model.pkl")
feature_cols = joblib.load("model_features.pkl")

# ── Global averages (fallback values) ────────────────────────────
GLOBAL = {
    "OddHome":  float(df["OddHome"].mean()),
    "OddDraw":  float(df["OddDraw"].mean()),
    "OddAway":  float(df["OddAway"].mean()),
    "Over25":   float(df["Over25"].mean()),
    "Under25":  float(df["Under25"].mean()),
    "HandiSize":float(df["HandiSize"].mean()),
}

# ── Build per-team stats ──────────────────────────────────────────
team_stats = {}
for team in pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique():
    hg = df[df["HomeTeam"] == team]
    ag = df[df["AwayTeam"]  == team]
    all_g = pd.concat([hg, ag])
    if len(all_g) == 0:
        continue

    team_stats[team] = {
        "elo":      float(pd.concat([hg["HomeElo"],    ag["AwayElo"]]).mean()),
        "form3":    float(pd.concat([hg["Form3Home"],  ag["Form3Away"]]).mean()),
        "form5":    float(pd.concat([hg["Form5Home"],  ag["Form5Away"]]).mean()),
        "attack":   float(hg["Home_Attack_Strength"].mean()) if len(hg) > 0 else 1.0,
        "defense":  float(ag["Away_Defense_Weakness"].mean()) if len(ag) > 0 else 1.0,
        # Real historical odds for this team when playing at home
        "odd_home": float(hg["OddHome"].mean()) if len(hg) > 0 else GLOBAL["OddHome"],
        "odd_draw": float(hg["OddDraw"].mean()) if len(hg) > 0 else GLOBAL["OddDraw"],
        "odd_away": float(hg["OddAway"].mean()) if len(hg) > 0 else GLOBAL["OddAway"],
        "over25":   float(all_g["Over25"].mean()),
        "under25":  float(all_g["Under25"].mean()),
        "handi":    float(all_g["HandiSize"].mean()),
        "games":    len(all_g),
        "division": str(all_g["Division"].mode()[0]),
    }

# ── League mapping ────────────────────────────────────────────────
LEAGUE_MAP = {
    "E0":  {"name": "Premier League",         "country": "England",     "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "E1":  {"name": "Championship",            "country": "England",     "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "E2":  {"name": "League One",              "country": "England",     "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "E3":  {"name": "League Two",              "country": "England",     "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "SP1": {"name": "La Liga",                 "country": "Spain",       "flag": "🇪🇸"},
    "SP2": {"name": "Segunda División",        "country": "Spain",       "flag": "🇪🇸"},
    "I1":  {"name": "Serie A",                 "country": "Italy",       "flag": "🇮🇹"},
    "I2":  {"name": "Serie B",                 "country": "Italy",       "flag": "🇮🇹"},
    "F1":  {"name": "Ligue 1",                 "country": "France",      "flag": "🇫🇷"},
    "F2":  {"name": "Ligue 2",                 "country": "France",      "flag": "🇫🇷"},
    "D1":  {"name": "Bundesliga",              "country": "Germany",     "flag": "🇩🇪"},
    "D2":  {"name": "2. Bundesliga",           "country": "Germany",     "flag": "🇩🇪"},
    "P1":  {"name": "Primeira Liga",           "country": "Portugal",    "flag": "🇵🇹"},
    "B1":  {"name": "Pro League",              "country": "Belgium",     "flag": "🇧🇪"},
    "N1":  {"name": "Eredivisie",              "country": "Netherlands", "flag": "🇳🇱"},
    "T1":  {"name": "Süper Lig",               "country": "Turkey",      "flag": "🇹🇷"},
    "SC0": {"name": "Scottish Premiership",    "country": "Scotland",    "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    "SC1": {"name": "Scottish Championship",   "country": "Scotland",    "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    "SC2": {"name": "Scottish League One",     "country": "Scotland",    "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    "SC3": {"name": "Scottish League Two",     "country": "Scotland",    "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    "RUS": {"name": "Russian Premier League",  "country": "Russia",      "flag": "🇷🇺"},
    "ROM": {"name": "Liga 1",                  "country": "Romania",     "flag": "🇷🇴"},
    "POL": {"name": "Ekstraklasa",             "country": "Poland",      "flag": "🇵🇱"},
    "SWE": {"name": "Allsvenskan",             "country": "Sweden",      "flag": "🇸🇪"},
    "NOR": {"name": "Eliteserien",             "country": "Norway",      "flag": "🇳🇴"},
    "DEN": {"name": "Superliga",               "country": "Denmark",     "flag": "🇩🇰"},
    "FIN": {"name": "Veikkausliiga",           "country": "Finland",     "flag": "🇫🇮"},
    "AUT": {"name": "Austrian Bundesliga",     "country": "Austria",     "flag": "🇦🇹"},
    "SUI": {"name": "Super League",            "country": "Switzerland", "flag": "🇨🇭"},
    "IRL": {"name": "League of Ireland",       "country": "Ireland",     "flag": "🇮🇪"},
    "G1":  {"name": "Super League",            "country": "Greece",      "flag": "🇬🇷"},
    "EC":  {"name": "European Competition",    "country": "Europe",      "flag": "🇪🇺"},
    "USA": {"name": "MLS",                     "country": "USA",         "flag": "🇺🇸"},
    "ARG": {"name": "Primera División",        "country": "Argentina",   "flag": "🇦🇷"},
    "BRA": {"name": "Série A",                 "country": "Brazil",      "flag": "🇧🇷"},
    "MEX": {"name": "Liga MX",                 "country": "Mexico",      "flag": "🇲🇽"},
    "JAP": {"name": "J-League",                "country": "Japan",       "flag": "🇯🇵"},
    "CHN": {"name": "Super League",            "country": "China",       "flag": "🇨🇳"},
}

# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

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
    for div_code in sorted(df["Division"].unique()):
        teams_in_div = sorted(set(
            df[df["Division"] == div_code]["HomeTeam"].tolist() +
            df[df["Division"] == div_code]["AwayTeam"].tolist()
        ))
        info = LEAGUE_MAP.get(div_code,
                              {"name": div_code, "country": "Unknown", "flag": "🌍"})
        result.append({
            "code":        div_code,
            "name":        info["name"],
            "country":     info["country"],
            "flag":        info["flag"],
            "teams":       teams_in_div,
            "teams_count": len(teams_in_div),
        })
    return jsonify({"leagues": result, "total": len(result)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data      = request.get_json()
        home_team = data["home_team"]
        away_team = data["away_team"]

        if home_team not in team_stats or away_team not in team_stats:
            return jsonify({"error": "Team not found"}), 404

        hs  = team_stats[home_team]
        as_ = team_stats[away_team]

        # ── Feature vector using REAL per-team historical averages ──
        features = {
            "HomeElo":               hs["elo"],
            "AwayElo":               as_["elo"],
            "Form3Home":             hs["form3"],
            "Form5Home":             hs["form5"],
            "Form3Away":             as_["form3"],
            "Form5Away":             as_["form5"],
            # Blend: home team's home odds vs away team's away odds
            "OddHome":               (hs["odd_home"] + as_["odd_away"]) / 2,
            "OddDraw":               (hs["odd_draw"] + as_["odd_draw"]) / 2,
            "OddAway":               (hs["odd_away"] + as_["odd_home"]) / 2,
            "Over25":                (hs["over25"]   + as_["over25"])   / 2,
            "Under25":               (hs["under25"]  + as_["under25"])  / 2,
            "HandiSize":              hs["handi"]    - as_["handi"],
            "Elo_Diff":               hs["elo"]      - as_["elo"],
            "Form3_Diff":             hs["form3"]    - as_["form3"],
            "Form5_Diff":             hs["form5"]    - as_["form5"],
            "Month":                  5,
            "DayOfWeek":              5,
            "Home_Attack_Strength":   hs["attack"],
            "Away_Defense_Weakness":  as_["defense"],
        }

        X = pd.DataFrame(
            [[features.get(f, 0) for f in feature_cols]],
            columns=feature_cols
        )

        xg_home = float(max(0.1, home_model.predict(X)[0]))
        xg_away = float(max(0.1, away_model.predict(X)[0]))

        # ── Poisson matrix ────────────────────────────────────────
        max_goals = 6
        hp = [poisson.pmf(i, xg_home) for i in range(max_goals + 1)]
        ap = [poisson.pmf(i, xg_away) for i in range(max_goals + 1)]

        matrix     = []
        win_home   = draw = win_away = 0.0
        top_scores = []

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = hp[i] * ap[j]
                matrix.append({"home": i, "away": j, "prob": round(p * 100, 2)})
                top_scores.append((i, j, p))
                if i > j:    win_home += p
                elif i == j: draw     += p
                else:        win_away += p

        top_scores.sort(key=lambda x: -x[2])
        top_5 = [
            {"score": f"{s[0]}-{s[1]}", "prob": round(s[2] * 100, 2)}
            for s in top_scores[:5]
        ]

        return jsonify({
            "home_team":      home_team,
            "away_team":      away_team,
            "xg_home":        round(xg_home, 2),
            "xg_away":        round(xg_away, 2),
            "win_prob": {
                "home": round(win_home  * 100, 1),
                "draw": round(draw      * 100, 1),
                "away": round(win_away  * 100, 1),
            },
            "top_scores":     top_5,
            "poisson_matrix": matrix,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
