# =================================================================
#  PREDICTIX API SERVER — Flask + Rate Limiting + Redis Cache
#  + Home Bias Fix + Prediction Calibration
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
import json
import hashlib
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
#  REDIS CACHE SETUP
#  - Uses Redis if REDIS_URL env var is set (production)
#  - Falls back to in-memory dict if Redis unavailable (free tier)
# =================================================================
REDIS_URL = os.environ.get('REDIS_URL', None)
_memory_cache = {}   # fallback in-memory cache

try:
    if REDIS_URL:
        import redis
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
        CACHE_BACKEND = 'redis'
        logger.info("✅ Redis cache connected")
    else:
        _redis = None
        CACHE_BACKEND = 'memory'
        logger.info("ℹ️  Using in-memory cache (set REDIS_URL for Redis)")
except Exception as e:
    _redis = None
    CACHE_BACKEND = 'memory'
    logger.warning(f"⚠️  Redis unavailable, using memory cache: {e}")

CACHE_TTL = 3600  # 1 hour

def cache_get(key: str):
    try:
        if CACHE_BACKEND == 'redis' and _redis:
            val = _redis.get(key)
            return json.loads(val) if val else None
        else:
            entry = _memory_cache.get(key)
            if entry and (datetime.utcnow().timestamp() - entry['ts']) < CACHE_TTL:
                return entry['data']
            return None
    except Exception:
        return None

def cache_set(key: str, value: dict):
    try:
        if CACHE_BACKEND == 'redis' and _redis:
            _redis.setex(key, CACHE_TTL, json.dumps(value))
        else:
            _memory_cache[key] = {
                'data': value,
                'ts': datetime.utcnow().timestamp()
            }
            # Keep memory cache small (max 200 entries)
            if len(_memory_cache) > 200:
                oldest = sorted(_memory_cache.items(), key=lambda x: x[1]['ts'])
                for k, _ in oldest[:50]:
                    del _memory_cache[k]
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")

def make_cache_key(prefix: str, *args) -> str:
    raw = f"{prefix}:{'|'.join(str(a).lower() for a in args)}"
    return hashlib.md5(raw.encode()).hexdigest()

# =================================================================
#  RATE LIMITER
# =================================================================
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=REDIS_URL if REDIS_URL else "memory://",
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

    # ── Compute global home advantage factor from real data ──────
    # نحسب نسبة فوز المضيف الحقيقية من البيانات
    total_matches  = len(df)
    actual_home_wr = len(df[df['FTResult'] == 'H']) / total_matches  # ~0.46 عادةً
    actual_draw_r  = len(df[df['FTResult'] == 'D']) / total_matches  # ~0.26
    actual_away_wr = len(df[df['FTResult'] == 'A']) / total_matches  # ~0.28
    logger.info(f"📊 Historical: Home {actual_home_wr:.1%} | Draw {actual_draw_r:.1%} | Away {actual_away_wr:.1%}")

    # ── Home advantage baseline: أهداف إضافية للمضيف تاريخياً ──
    avg_home_goals = df['FTHome'].mean()  # متوسط أهداف المضيف
    avg_away_goals = df['FTAway'].mean()  # متوسط أهداف الضيف
    HOME_BIAS_CORRECTION = avg_home_goals - avg_away_goals  # ~0.35 عادةً
    logger.info(f"⚖️  Home bias correction factor: {HOME_BIAS_CORRECTION:.3f}")

except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    home_model = away_model = features_list = df = None
    actual_home_wr = 0.46
    actual_draw_r  = 0.26
    actual_away_wr = 0.28
    HOME_BIAS_CORRECTION = 0.35

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
#  HELPER: team stats  (مع cache)
# =================================================================
def get_team_stats(team_name: str) -> dict | None:
    if df is None:
        return None

    cache_key = make_cache_key('team_stats', team_name)
    cached = cache_get(cache_key)
    if cached:
        return cached

    hg    = df[df['HomeTeam'] == team_name]
    ag    = df[df['AwayTeam'] == team_name]
    all_g = pd.concat([hg, ag])
    if all_g.empty:
        return None

    wins   = len(hg[hg['FTResult'] == 'H']) + len(ag[ag['FTResult'] == 'A'])
    draws  = len(all_g[all_g['FTResult'] == 'D'])
    losses = len(all_g) - wins - draws

    # ── Real win rates للفريق (المضيف والضيف منفصلين) ──────────
    home_wr = len(hg[hg['FTResult'] == 'H']) / len(hg) if len(hg) > 0 else 0.4
    away_wr = len(ag[ag['FTResult'] == 'A']) / len(ag) if len(ag) > 0 else 0.25
    home_goals_avg = float(hg['FTHome'].mean()) if not hg.empty else 1.5
    away_goals_avg = float(ag['FTAway'].mean()) if not ag.empty else 1.0

    stats = {
        'team':         team_name,
        'matches':      len(all_g),
        'wins':         wins,
        'draws':        draws,
        'losses':       losses,
        'home_wr':      round(home_wr, 3),
        'away_wr':      round(away_wr, 3),
        'home_goals_avg': round(home_goals_avg, 3),
        'away_goals_avg': round(away_goals_avg, 3),
        'home_elo':     float(hg['HomeElo'].mean()) if not hg.empty else 1500.0,
        'away_elo':     float(ag['AwayElo'].mean()) if not ag.empty else 1500.0,
        'form3':        float(all_g['Form3Home'].mean()) if 'Form3Home' in df.columns else 0.5,
        'form5':        float(all_g['Form5Home'].mean()) if 'Form5Home' in df.columns else 0.5,
        'odd_home':     float(hg['OddHome'].mean())   if not hg.empty else 2.0,
        'odd_draw':     float(hg['OddDraw'].mean())   if not hg.empty else 3.2,
        'odd_away':     float(hg['OddAway'].mean())   if not hg.empty else 3.5,
        'odd_away_ag':  float(ag['OddAway'].mean())   if not ag.empty else 3.5,
        'odd_home_ag':  float(ag['OddHome'].mean())   if not ag.empty else 2.0,
        'over25':       float(all_g['Over25'].mean())  if 'Over25'  in df.columns else 0.55,
        'under25':      float(all_g['Under25'].mean()) if 'Under25' in df.columns else 0.45,
        'handi':        float(all_g['HandiSize'].mean()) if 'HandiSize' in df.columns else 0.0,
        'atk_str':      float(all_g['Home_Attack_Strength'].mean())  if 'Home_Attack_Strength'  in df.columns else 1.0,
        'def_wk':       float(all_g['Away_Defense_Weakness'].mean()) if 'Away_Defense_Weakness' in df.columns else 1.0,
    }

    cache_set(cache_key, stats)
    return stats


# =================================================================
#  HELPER: calibrate xG to fix home bias
#
#  المشكلة: النموذج يعطي المضيف xG أعلى دائماً حتى لو الضيف أقوى
#  الحل:
#    1. نحسب نسبة قوة كل فريق من بياناته الحقيقية
#    2. نطبق strength_ratio لتعديل xG بناءً على قوة الفريقين
#    3. نطرح جزءاً من HOME_BIAS_CORRECTION من xG المضيف
#       إذا كان الضيف أقوى تاريخياً
# =================================================================
def calibrate_xg(
    raw_home_xg: float,
    raw_away_xg: float,
    hs: dict,
    as_: dict,
) -> tuple[float, float]:

    # ── 1. حساب نسبة قوة الهجوم التاريخية ──────────────────────
    home_attack  = hs.get('home_goals_avg', 1.5)   # أهداف المضيف وهو يلعب في ملعبه
    away_attack  = as_.get('away_goals_avg', 1.0)  # أهداف الضيف وهو يلعب بعيداً
    global_avg_h = avg_home_goals if df is not None else 1.5
    global_avg_a = avg_away_goals if df is not None else 1.1

    # Attack strength مقارنةً بالمتوسط العالمي
    home_atk_factor = home_attack / global_avg_h if global_avg_h > 0 else 1.0
    away_atk_factor = away_attack / global_avg_a if global_avg_a > 0 else 1.0

    # ── 2. تعديل xG الخام بعوامل القوة الحقيقية ────────────────
    calibrated_home = raw_home_xg * home_atk_factor
    calibrated_away = raw_away_xg * away_atk_factor

    # ── 3. إزالة Home Bias الزائد ───────────────────────────────
    # إذا كان ELO الضيف أعلى — نخفض ميزة المضيف
    home_elo = hs.get('home_elo', 1500)
    away_elo = as_.get('away_elo', 1500)
    elo_diff = home_elo - away_elo  # موجب = المضيف أقوى، سالب = الضيف أقوى

    # نحول فرق ELO إلى تعديل خطي (كل 100 نقطة ELO ≈ 0.1 هدف)
    elo_adjustment = np.clip(elo_diff / 1000.0, -0.3, 0.3)

    # تطبيق التعديل: إذا الضيف أقوى — نرفع xG-ه ونخفض xG المضيف قليلاً
    calibrated_home = calibrated_home + elo_adjustment * 0.15
    calibrated_away = calibrated_away - elo_adjustment * 0.15

    # ── 4. تصحيح Home Bias الهيكلي ──────────────────────────────
    # النموذج مدرب على features تعطي المضيف ميزة دائماً
    # نطبق تصحيحاً نسبياً بسيطاً
    bias_factor = HOME_BIAS_CORRECTION * 0.25  # 25% من الـ bias الحقيقي
    calibrated_home = calibrated_home - bias_factor
    calibrated_away = calibrated_away + bias_factor * 0.5

    # ── 5. Bounds: لا يقل عن 0.3 ────────────────────────────────
    calibrated_home = max(0.3, calibrated_home)
    calibrated_away = max(0.3, calibrated_away)

    return round(calibrated_home, 3), round(calibrated_away, 3)


# =================================================================
#  ENDPOINTS
# =================================================================

# ── /ping ────────────────────────────────────────────────────────
@app.route('/ping', methods=['GET'])
@limiter.exempt
def ping():
    return jsonify({
        'status':    'alive',
        'timestamp': datetime.utcnow().isoformat(),
        'service':   'predictix-api',
        'cache':     CACHE_BACKEND,
    }), 200


# ── /health ──────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'models':       home_model is not None,
        'teams':        len(df['HomeTeam'].unique()) if df is not None else 0,
        'rows':         len(df) if df is not None else 0,
        'cache':        CACHE_BACKEND,
        'home_bias_fix': round(HOME_BIAS_CORRECTION, 3),
    }), 200


# ── /teams ───────────────────────────────────────────────────────
@app.route('/teams', methods=['GET'])
def get_teams():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500

    cache_key = make_cache_key('teams_list')
    cached = cache_get(cache_key)
    if cached:
        return jsonify(cached), 200

    teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    result = {'teams': teams, 'count': len(teams)}
    cache_set(cache_key, result)
    return jsonify(result), 200


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

    cache_key = make_cache_key('leagues_list')
    cached = cache_get(cache_key)
    if cached:
        return jsonify(cached), 200

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

    result = {'leagues': leagues, 'total': len(leagues)}
    cache_set(cache_key, result)
    return jsonify(result), 200


# ── /predict ─────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
@limiter.limit("30 per hour")
def predict():
    if home_model is None:
        return jsonify({'error': 'Models not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    home_team = data.get('home_team', '').strip()
    away_team = data.get('away_team', '').strip()

    if not home_team or not away_team:
        return jsonify({'error': 'home_team and away_team are required'}), 400

    # ── Cache check ──────────────────────────────────────────────
    cache_key = make_cache_key('predict', home_team, away_team)
    cached = cache_get(cache_key)
    if cached:
        cached['cached'] = True
        logger.info(f"⚡ Cache hit: {home_team} vs {away_team}")
        return jsonify(cached), 200

    hs  = get_team_stats(home_team)
    as_ = get_team_stats(away_team)

    if not hs:
        return jsonify({'error': f'Home team "{home_team}" not found'}), 404
    if not as_:
        return jsonify({'error': f'Away team "{away_team}" not found'}), 404

    # ── Feature Vector ───────────────────────────────────────────
    feature_values = {
        'HomeElo':               hs['home_elo'],
        'AwayElo':               as_['away_elo'],
        'Form3Home':             hs['form3'],
        'Form5Home':             hs['form5'],
        'Form3Away':             as_['form3'],
        'Form5Away':             as_['form5'],
        'OddHome':               (hs['odd_home'] + as_['odd_away_ag']) / 2,
        'OddDraw':               (hs['odd_draw'] + as_['odd_draw'])    / 2,
        'OddAway':               (hs['odd_away'] + as_['odd_home_ag']) / 2,
        'Over25':                (hs['over25']   + as_['over25'])      / 2,
        'Under25':               (hs['under25']  + as_['under25'])     / 2,
        'HandiSize':             (hs['handi']    + as_['handi'])       / 2,
        'Elo_Diff':              hs['home_elo']  - as_['away_elo'],
        'Form3_Diff':            hs['form3']     - as_['form3'],
        'Form5_Diff':            hs['form5']     - as_['form5'],
        'Month':                 datetime.utcnow().month,
        'DayOfWeek':             datetime.utcnow().weekday(),
        'Home_Attack_Strength':  hs['atk_str'],
        'Away_Defense_Weakness': as_['def_wk'],
    }

    try:
        X = np.array([[feature_values.get(f, 0) for f in features_list]])
        raw_home_xg = float(home_model.predict(X)[0])
        raw_away_xg = float(away_model.predict(X)[0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # ── Apply Home Bias Calibration ──────────────────────────────
    home_xg, away_xg = calibrate_xg(raw_home_xg, raw_away_xg, hs, as_)

    logger.info(
        f"🔢 xG raw: {raw_home_xg:.2f}-{raw_away_xg:.2f} "
        f"→ calibrated: {home_xg:.2f}-{away_xg:.2f}"
    )

    # ── Poisson Matrix (0-5 goals) ───────────────────────────────
    max_goals = 6
    poisson_matrix = []
    home_win_prob = draw_prob = away_win_prob = 0.0

    for h in range(max_goals):
        for a in range(max_goals):
            prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg) * 100
            poisson_matrix.append({
                'home': h, 'away': a,
                'probability': round(prob, 2)
            })
            if h > a:   home_win_prob  += prob
            elif h == a: draw_prob     += prob
            else:        away_win_prob += prob

    total = home_win_prob + draw_prob + away_win_prob
    if total > 0:
        home_win_prob = round(home_win_prob / total * 100, 1)
        draw_prob     = round(draw_prob     / total * 100, 1)
        away_win_prob = round(away_win_prob / total * 100, 1)

    sorted_scores = sorted(
        poisson_matrix, key=lambda x: x['probability'], reverse=True)
    top_scores = [
        {'score': f"{s['home']}-{s['away']}",
         'probability': round(s['probability'], 1)}
        for s in sorted_scores[:5]
    ]

    logger.info(
        f"✅ {home_team} vs {away_team} → {top_scores[0]['score']} "
        f"| H:{home_win_prob}% D:{draw_prob}% A:{away_win_prob}%"
    )

    result = {
        'home_team':       home_team,
        'away_team':       away_team,
        'xg_home':         home_xg,
        'xg_away':         away_xg,
        'xg_raw': {
            'home': round(raw_home_xg, 2),
            'away': round(raw_away_xg, 2),
        },
        'win_probability': {
            'home': home_win_prob,
            'draw': draw_prob,
            'away': away_win_prob,
        },
        'poisson_matrix':  poisson_matrix,
        'top_scores':      top_scores,
        'cached':          False,
    }

    cache_set(cache_key, result)
    return jsonify(result), 200


# ── /cache/clear — لإعادة ضبط الـ cache يدوياً ──────────────────
@app.route('/cache/clear', methods=['POST'])
@limiter.exempt
def clear_cache():
    """
    Secret key مطلوب في الـ body: {"secret": "your_secret"}
    يُستخدم بعد رفع نماذج جديدة لمسح النتائج القديمة.
    """
    secret = os.environ.get('CACHE_CLEAR_SECRET', 'predictix2026')
    body   = request.get_json() or {}
    if body.get('secret') != secret:
        return jsonify({'error': 'Unauthorized'}), 403

    global _memory_cache
    if CACHE_BACKEND == 'redis' and _redis:
        _redis.flushdb()
        msg = 'Redis cache cleared'
    else:
        _memory_cache = {}
        msg = f'Memory cache cleared'

    logger.info(f"🗑️  {msg}")
    return jsonify({'status': 'ok', 'message': msg}), 200


# ── Error Handlers ───────────────────────────────────────────────
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error':       'Too many requests',
        'message':     str(e.description),
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
