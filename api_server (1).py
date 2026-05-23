# =================================================================
#  PREDICTIX API SERVER — Flask + Rate Limiting
#  + UptimeRobot ping endpoint
# =================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
import os
import logging
from datetime import datetime

# =================================================================
#  LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# =================================================================
#  APP INIT
# =================================================================
app = Flask(__name__)
CORS(app)

# =================================================================
#  RATE LIMITER
#  - Global: 200 requests/day, 50/hour per IP
#  - /predict: max 30 predictions/hour per IP (أغلى endpoint)
#  - /ping:    بدون حد (UptimeRobot يضرب كل 14 دقيقة)
# =================================================================
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",   # ← غيّره إلى Redis URI إذا أضفت Redis لاحقاً
)

# =================================================================
#  LOAD MODELS & DATA
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    home_model    = joblib.load(os.path.join(BASE_DIR, 'home_goals_model.pkl'))
    away_model    = joblib.load(os.path.join(BASE_DIR, 'away_goals_model.pkl'))
    features_list = joblib.load(os.path.join(BASE_DIR, 'model_features.pkl'))
    df            = pd.read_csv(os.path.join(BASE_DIR, 'Football_Data_Cleaned_2026.csv'))
    logger.info(f"✅ Models loaded. Dataset: {len(df):,} rows")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    home_model = away_model = features_list = df = None

# =================================================================
#  LEAGUE MAP
# =================================================================
LEAGUE_MAP = {
    'E0':  ('Premier League',      'England',     '🏴󠁧󠁢󠁥󠁮󠁧󠁿'),
    'E1':  ('Championship',        'England',     '🏴󠁧󠁢󠁥󠁮󠁧󠁿'),
    'E2':  ('League One',          'England',     '🏴󠁧󠁢󠁥󠁮󠁧󠁿'),
    'E3':  ('League Two',          'England',     '🏴󠁧󠁢󠁥󠁮󠁧󠁿'),
    'SP1': ('La Liga',             'Spain',       '🇪🇸'),
    'SP2': ('Segunda División',    'Spain',       '🇪🇸'),
    'I1':  ('Serie A',             'Italy',       '🇮🇹'),
    'I2':  ('Serie B',             'Italy',       '🇮🇹'),
    'F1':  ('Ligue 1',             'France',      '🇫🇷'),
    'F2':  ('Ligue 2',             'France',      '🇫🇷'),
    'D1':  ('Bundesliga',          'Germany',     '🇩🇪'),
    'D2':  ('2. Bundesliga',       'Germany',     '🇩🇪'),
    'P1':  ('Primeira Liga',       'Portugal',    '🇵🇹'),
    'B1':  ('Pro League',          'Belgium',     '🇧🇪'),
    'N1':  ('Eredivisie',          'Netherlands', '🇳🇱'),
    'T1':  ('Süper Lig',           'Turkey',      '🇹🇷'),
    'SC0': ('Scottish Premiership','Scotland',    '🏴󠁧󠁢󠁳󠁣󠁴󠁿'),
    'RUS': ('Premier League',      'Russia',      '🇷🇺'),
    'ARG': ('Primera División',    'Argentina',   '🇦🇷'),
    'BRA': ('Série A',             'Brazil',      '🇧🇷'),
    'USA': ('MLS',                 'USA',         '🇺🇸'),
    'MEX': ('Liga MX',             'Mexico',      '🇲🇽'),
    'JAP': ('J-League',            'Japan',       '🇯🇵'),
    'CHN': ('Super League',        'China',       '🇨🇳'),
}

# =================================================================
#  HELPER: team stats
# =================================================================
def get_team_stats(team_name: str) -> dict | None:
    if df is None:
        return None
    hg   = df[df['HomeTeam'] == team_name]
    ag   = df[df['AwayTeam'] == team_name]
    all_g = pd.concat([hg, ag])
    if all_g.empty:
        return None
    wins   = len(hg[hg['FTResult'] == 'H']) + len(ag[ag['FTResult'] == 'A'])
    draws  = len(all_g[all_g['FTResult'] == 'D'])
    losses = len(all_g) - wins - draws
    return {
        'team':      team_name,
        'matches':   len(all_g),
        'wins':      wins,
        'draws':     draws,
        'losses':    losses,
        'home_elo':  float(hg['HomeElo'].mean()) if not hg.empty else 1500.0,
        'away_elo':  float(ag['AwayElo'].mean()) if not ag.empty else 1500.0,
        'form3':     float(all_g['Form3Home'].mean()) if 'Form3Home' in df.columns else 0.5,
        'form5':     float(all_g['Form5Home'].mean()) if 'Form5Home' in df.columns else 0.5,
        'odd_home':  float(hg['OddHome'].mean())  if not hg.empty else 2.0,
        'odd_draw':  float(hg['OddDraw'].mean())  if not hg.empty else 3.2,
        'odd_away':  float(hg['OddAway'].mean())  if not hg.empty else 3.5,
        'odd_away_ag': float(ag['OddAway'].mean()) if not ag.empty else 3.5,
        'odd_home_ag': float(ag['OddHome'].mean()) if not ag.empty else 2.0,
        'over25':    float(all_g['Over25'].mean()) if 'Over25' in df.columns else 0.55,
        'under25':   float(all_g['Under25'].mean()) if 'Under25' in df.columns else 0.45,
        'handi':     float(all_g['HandiSize'].mean()) if 'HandiSize' in df.columns else 0.0,
        'atk_str':   float(all_g['Home_Attack_Strength'].mean()) if 'Home_Attack_Strength' in df.columns else 1.0,
        'def_wk':    float(all_g['Away_Defense_Weakness'].mean()) if 'Away_Defense_Weakness' in df.columns else 1.0,
    }

# =================================================================
#  ENDPOINTS
# =================================================================

# ── /ping — UptimeRobot keep-alive (بدون rate limit) ────────────
@app.route('/ping', methods=['GET'])
@limiter.exempt
def ping():
    """
    UptimeRobot يضرب هذا الـ endpoint كل 14 دقيقة.
    لا يحسب ضد حد الطلبات.
    """
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'predictix-api'
    }), 200


# ── /health ──────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'ok',
        'models':  home_model is not None,
        'teams':   len(df['HomeTeam'].unique()) if df is not None else 0,
        'rows':    len(df) if df is not None else 0,
    }), 200


# ── /teams ───────────────────────────────────────────────────────
@app.route('/teams', methods=['GET'])
def get_teams():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    return jsonify({'teams': teams, 'count': len(teams)}), 200


# ── /team/<name> ─────────────────────────────────────────────────
@app.route('/team/<name>', methods=['GET'])
def get_team(name):
    stats = get_team_stats(name)
    if not stats:
        return jsonify({'error': f'Team "{name}" not found'}), 404
    return jsonify(stats), 200


# ── /leagues ─────────────────────────────────────────────────────
@app.route('/leagues', methods=['GET'])
def get_leagues():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    leagues = []
    for code, (name, country, flag) in LEAGUE_MAP.items():
        teams_in_div = df[df['Division'] == code]['HomeTeam'].unique().tolist()
        if teams_in_div:
            leagues.append({
                'code':    code,
                'name':    name,
                'country': country,
                'flag':    flag,
                'teams':   sorted(teams_in_div),
                'count':   len(teams_in_div),
            })
    return jsonify({'leagues': leagues, 'total': len(leagues)}), 200


# ── /predict — rate limited: 30/hour per IP ──────────────────────
@app.route('/predict', methods=['POST'])
@limiter.limit("30 per hour")
def predict():
    if home_model is None:
        return jsonify({'error': 'Models not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    home_team = data.get('home_team', '')
    away_team = data.get('away_team', '')

    if not home_team or not away_team:
        return jsonify({'error': 'home_team and away_team are required'}), 400

    hs = get_team_stats(home_team)
    as_ = get_team_stats(away_team)

    if not hs:
        return jsonify({'error': f'Home team "{home_team}" not found'}), 404
    if not as_:
        return jsonify({'error': f'Away team "{away_team}" not found'}), 404

    # -- Feature Vector --
    feature_values = {
        'HomeElo':               hs['home_elo'],
        'AwayElo':               as_['away_elo'],
        'Form3Home':             hs['form3'],
        'Form5Home':             hs['form5'],
        'Form3Away':             as_['form3'],
        'Form5Away':             as_['form5'],
        'OddHome':               (hs['odd_home'] + as_['odd_away_ag']) / 2,
        'OddDraw':               (hs['odd_draw'] + as_['odd_draw']) / 2,
        'OddAway':               (hs['odd_away'] + as_['odd_home_ag']) / 2,
        'Over25':                (hs['over25'] + as_['over25']) / 2,
        'Under25':               (hs['under25'] + as_['under25']) / 2,
        'HandiSize':             (hs['handi'] + as_['handi']) / 2,
        'Elo_Diff':              hs['home_elo'] - as_['away_elo'],
        'Form3_Diff':            hs['form3'] - as_['form3'],
        'Form5_Diff':            hs['form5'] - as_['form5'],
        'Month':                 datetime.utcnow().month,
        'DayOfWeek':             datetime.utcnow().weekday(),
        'Home_Attack_Strength':  hs['atk_str'],
        'Away_Defense_Weakness': as_['def_wk'],
    }

    try:
        X = np.array([[feature_values.get(f, 0) for f in features_list]])
        home_xg = float(home_model.predict(X)[0])
        away_xg = float(away_model.predict(X)[0])
        home_xg = max(0.1, home_xg)
        away_xg = max(0.1, away_xg)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # -- Poisson Matrix (0-5 goals) --
    max_goals = 6
    poisson_matrix = []
    home_win_prob = draw_prob = away_win_prob = 0.0

    for h in range(max_goals):
        for a in range(max_goals):
            prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg) * 100
            poisson_matrix.append({'home': h, 'away': a, 'probability': round(prob, 2)})
            if h > a:
                home_win_prob += prob
            elif h == a:
                draw_prob += prob
            else:
                away_win_prob += prob

    total = home_win_prob + draw_prob + away_win_prob
    if total > 0:
        home_win_prob = round(home_win_prob / total * 100, 1)
        draw_prob     = round(draw_prob     / total * 100, 1)
        away_win_prob = round(away_win_prob / total * 100, 1)

    # -- Top Scores --
    sorted_scores = sorted(poisson_matrix, key=lambda x: x['probability'], reverse=True)
    top_scores = [{'score': f"{s['home']}-{s['away']}",
                   'probability': round(s['probability'], 1)}
                  for s in sorted_scores[:5]]

    logger.info(f"Predicted: {home_team} vs {away_team} → {top_scores[0]['score']}")

    return jsonify({
        'home_team':      home_team,
        'away_team':      away_team,
        'xg_home':        round(home_xg, 2),
        'xg_away':        round(away_xg, 2),
        'win_probability': {
            'home': home_win_prob,
            'draw': draw_prob,
            'away': away_win_prob,
        },
        'poisson_matrix': poisson_matrix,
        'top_scores':     top_scores,
    }), 200


# ── Error Handlers ───────────────────────────────────────────────
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Too many requests',
        'message': str(e.description),
        'retry_after': '1 hour'
    }), 429

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# =================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
