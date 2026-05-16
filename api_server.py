from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from scipy.stats import poisson

app = Flask(__name__)
CORS(app)  # يسمح لـ Flutter بالاتصال

# ── تحميل النماذج عند بدء السيرفر ──────────────────────────────
home_model   = joblib.load("home_goals_model.pkl")
away_model   = joblib.load("away_goals_model.pkl")
feature_names = joblib.load("model_features.pkl")

# ── تحميل إحصائيات الفرق من الـ CSV ────────────────────────────
df = pd.read_csv("Football_Data_Cleaned_2026.csv")

teams_stats = df.groupby("HomeTeam").agg(
    HomeElo=("HomeElo", "last"),
    Home_Attack_Strength=("Home_Attack_Strength", "mean"),
    Away_Defense_Weakness=("Away_Defense_Weakness", "mean"),
    OddHome=("OddHome", "mean"),
).to_dict("index")

ALL_TEAMS = sorted(teams_stats.keys())


# ══════════════════════════════════════════════════════════════════
#  GET /teams  →  قائمة كل الفرق المتاحة
# ══════════════════════════════════════════════════════════════════
@app.route("/teams", methods=["GET"])
def get_teams():
    return jsonify({"teams": ALL_TEAMS})


# ══════════════════════════════════════════════════════════════════
#  POST /predict  →  توقع نتيجة المباراة
#
#  Request JSON:
#    { "home_team": "Arsenal", "away_team": "Chelsea" }
#
#  Response JSON:
#    {
#      "home_team": "Arsenal",
#      "away_team": "Chelsea",
#      "home_xg": 2.1,
#      "away_xg": 1.3,
#      "win_prob":  { "home": 58.2, "draw": 24.1, "away": 17.7 },
#      "top_scores": [
#          { "score": "2-1", "probability": 22.8 },
#          { "score": "1-0", "probability": 14.3 },
#          { "score": "2-0", "probability": 11.6 }
#      ],
#      "poisson_matrix": [
#          { "score": "0-0", "probability": 9.2, "is_predicted": false },
#          ...
#      ]
#    }
# ══════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")

    # ── التحقق من صحة الفرق ──
    if home_team not in teams_stats:
        return jsonify({"error": f"الفريق '{home_team}' غير موجود"}), 400
    if away_team not in teams_stats:
        return jsonify({"error": f"الفريق '{away_team}' غير موجود"}), 400
    if home_team == away_team:
        return jsonify({"error": "لا يمكن أن يلعب الفريق ضد نفسه"}), 400

    h = teams_stats[home_team]
    a = teams_stats[away_team]

    # ── تجهيز مدخلات النموذج ──
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    input_df["HomeElo"]              = h["HomeElo"]
    input_df["AwayElo"]              = a["HomeElo"]
    input_df["Elo_Diff"]             = h["HomeElo"] - a["HomeElo"]
    input_df["Home_Attack_Strength"] = h["Home_Attack_Strength"]
    input_df["Away_Defense_Weakness"]= a["Away_Defense_Weakness"]
    input_df["OddHome"]              = h["OddHome"]

    # ── توقع الأهداف ──
    home_xg = float(home_model.predict(input_df)[0])
    away_xg = float(away_model.predict(input_df)[0])

    # ── حساب توزيع Poisson ──
    max_g = 6
    h_probs = poisson.pmf(np.arange(max_g), home_xg)
    a_probs = poisson.pmf(np.arange(max_g), away_xg)
    matrix  = np.outer(h_probs, a_probs)

    # ── احتمالات الفوز ──
    home_win = float(np.sum(np.tril(matrix, -1)) * 100)
    draw     = float(np.sum(np.diag(matrix))     * 100)
    away_win = float(np.sum(np.triu(matrix, 1))  * 100)

    # ── أفضل 9 نتائج لشاشة Poisson ──
    poisson_list = []
    for h_g in range(max_g):
        for a_g in range(max_g):
            poisson_list.append({
                "score":        f"{h_g}-{a_g}",
                "probability":  round(float(matrix[h_g, a_g]) * 100, 2),
                "is_home_win":  h_g > a_g,
                "is_predicted": False,
            })

    poisson_list.sort(key=lambda x: x["probability"], reverse=True)

    # ── تحديد النتيجة المتوقعة ──
    if poisson_list:
        poisson_list[0]["is_predicted"] = True

    # ── أفضل 3 نتائج للعرض السريع ──
    top_scores = [
        {"score": p["score"], "probability": p["probability"]}
        for p in poisson_list[:3]
    ]

    return jsonify({
        "home_team":      home_team,
        "away_team":      away_team,
        "home_xg":        round(home_xg, 2),
        "away_xg":        round(away_xg, 2),
        "win_prob": {
            "home":  round(home_win, 1),
            "draw":  round(draw,     1),
            "away":  round(away_win, 1),
        },
        "top_scores":     top_scores,
        "poisson_matrix": poisson_list[:9],
    })


# ══════════════════════════════════════════════════════════════════
#  GET /team/<name>  →  إحصائيات فريق معين
# ══════════════════════════════════════════════════════════════════
@app.route("/team/<name>", methods=["GET"])
def get_team(name):
    if name not in teams_stats:
        return jsonify({"error": "الفريق غير موجود"}), 404

    t = teams_stats[name]
    return jsonify({
        "name":                  name,
        "elo":                   round(t["HomeElo"], 1),
        "attack_strength":       round(t["Home_Attack_Strength"], 3),
        "defense_weakness":      round(t["Away_Defense_Weakness"], 3),
        "avg_odd":               round(t["OddHome"], 2),
    })


# ══════════════════════════════════════════════════════════════════
#  GET /health  →  التحقق من أن السيرفر يعمل
# ══════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "teams_loaded": len(ALL_TEAMS)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
