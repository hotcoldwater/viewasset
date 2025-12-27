import json
import hashlib
import io
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf


# ======================
# Altair theme (light)
# ======================
def _altair_theme():
    return {
        "config": {
            "background": "#ffffff",
            "view": {"stroke": "#c8ddf5", "fill": "#ffffff"},
            "axis": {
                "labelColor": "#0b1c2c",
                "titleColor": "#0b1c2c",
                "gridColor": "#d7e2ef",
                "domainColor": "#8aa4bf",
                "tickColor": "#8aa4bf",
            },
            "legend": {"labelColor": "#0b1c2c", "titleColor": "#0b1c2c"},
            "title": {"color": "#0b1c2c"},
        }
    }


if "va_light" not in alt.themes.names():
    alt.themes.register("va_light", _altair_theme)
alt.themes.enable("va_light")


# ======================
# 공통 설정
# ======================
TICKER_LIST = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY", "IWD", "GLD", "QQQ", "BIL"]

# ✅ 사용자가 직접 보유 입력하는 티커(=BIL 입력칸 제거)
INPUT_TICKERS = [t for t in TICKER_LIST if t != "BIL"]

# ✅ 전략 티커/룰 설정 (여기만 수정하면 전체 반영)
VAA_SIGNAL = ["SPY", "EFA", "EEM"]
VAA_RISK_ON = ["SPY", "EFA", "EEM", "AGG"]
VAA_DEFENSIVE = ["LQD", "IEF", "SHY"]
LAA_TICKERS = ["IWD", "GLD", "IEF", "QQQ", "SHY"]
TDM_TICKERS = ["SPY", "EFA", "AGG", "SHY", "IEF"]

# ✅ VAA 모멘텀 표기(7개) + 선택도 7개 중에서
VAA_UNIVERSE = VAA_RISK_ON + VAA_DEFENSIVE

# ✅ 전략 비중/밴드
STRATEGY_WEIGHTS = {"VAA": 0.30, "LAA": 0.30, "TDM": 0.40}
REBALANCE_BAND = 0.10  # 10%p
STRATEGY_TICKERS = {
    "VAA": VAA_UNIVERSE,
    "LAA": LAA_TICKERS,
    "TDM": TDM_TICKERS,
}

st.set_page_config(page_title="Rebalance (Private)", layout="wide")
st.markdown(
    """
<style>
:root {
  --va-blue-900: #0b2a4a;
  --va-blue-700: #134b7f;
  --va-blue-500: #1e6fbf;
  --va-blue-200: #d7e9fb;
  --va-blue-50: #f5faff;
  --va-white: #ffffff;
  --va-text: #0b1c2c;
  --va-text-invert: #ffffff;
  --va-border: #c8ddf5;
}

.stApp {
  background: linear-gradient(180deg, var(--va-blue-50) 0%, var(--va-white) 60%);
  color: var(--va-text);
}

section.main, section.main * {
  color: var(--va-text);
}

section.main a {
  color: var(--va-blue-700);
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stToolbar"] * {
  color: var(--va-text) !important;
  fill: var(--va-text) !important;
}

[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a {
  color: var(--va-text) !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--va-blue-900) 0%, var(--va-blue-700) 100%);
}

[data-testid="stSidebar"] * {
  color: var(--va-text-invert);
}

section.main [data-testid="stExpander"] details {
  background: var(--va-white);
  border: 1px solid var(--va-border);
  border-radius: 10px;
}

section.main [data-testid="stExpander"] {
  background: var(--va-white);
  border: 1px solid var(--va-border);
  border-radius: 10px;
}

section.main [data-testid="stExpander"] summary {
  background: var(--va-white) !important;
  color: var(--va-text) !important;
}

section.main [data-testid="stExpander"] summary:focus,
section.main [data-testid="stExpander"] summary:active {
  background: var(--va-white) !important;
}

section.main [data-testid="stExpander"] details[open] summary {
  background: var(--va-white) !important;
}

section.main [data-testid="stExpander"] details > div {
  background: var(--va-white);
  color: var(--va-text);
}

.stButton > button {
  border: 1px solid var(--va-blue-500);
  background: var(--va-white);
  color: var(--va-blue-700);
  font-weight: 600;
}

.stButton > button[kind="primary"] {
  background: var(--va-blue-500);
  color: var(--va-text-invert);
  border-color: var(--va-blue-500);
}

.stButton > button:hover {
  border-color: var(--va-blue-700);
  color: var(--va-blue-900);
}

.stButton > button[kind="primary"]:hover {
  background: var(--va-blue-700);
  color: var(--va-text-invert);
}

[data-testid="stSidebar"] .stButton > button {
  border-color: var(--va-blue-200);
  background: var(--va-blue-200);
  color: var(--va-text);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--va-blue-200);
  color: var(--va-text);
  border-color: var(--va-blue-200);
}

[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] .stButton > button[kind="primary"] * {
  color: var(--va-text);
}

[data-testid="stSidebar"] .stButton > button:hover {
  color: var(--va-blue-900);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: var(--va-blue-50);
  color: var(--va-blue-900);
}

[data-testid="stSidebar"] .stButton:last-of-type > button {
  background: var(--va-white);
  border-color: var(--va-blue-200);
  color: var(--va-text);
}

[data-testid="stSidebar"] .stButton:last-of-type > button * {
  color: var(--va-text);
}

[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
  background: var(--va-white);
  color: var(--va-blue-900);
}

section.main .vega-embed,
section.main .vega-embed canvas,
section.main .vega-embed svg {
  background: var(--va-white) !important;
}

input, textarea, select {
  color: var(--va-text) !important;
  background: var(--va-white) !important;
}

[data-testid="stTextInput"] label,
[data-testid="stFileUploader"] label {
  color: var(--va-text) !important;
  background: transparent !important;
}

.stDownloadButton > button {
  border: 1px solid var(--va-blue-500);
  background: var(--va-blue-500);
  color: var(--va-text-invert) !important;
  font-weight: 600;
}

.stDownloadButton > button:hover {
  background: var(--va-blue-700);
  border-color: var(--va-blue-700);
  color: var(--va-text-invert) !important;
}

.stDownloadButton > button * {
  color: var(--va-text-invert) !important;
}

div[data-testid="stMetric"] {
  background: var(--va-white);
  border: 1px solid var(--va-border);
  border-radius: 10px;
  padding: 12px;
}

div[data-testid="stMetric"] * {
  color: var(--va-text);
}

div[data-testid="stDataFrame"] {
  background: var(--va-white);
}

div[data-testid="stAlert"] {
  background: #e6f1ff;
  color: var(--va-text);
  border-color: var(--va-blue-200);
}
</style>
""",
    unsafe_allow_html=True,
)
st.sidebar.title("AssetView")
st.sidebar.subheader("자산 관리")


# ======================
# 모드 버튼 (사이드바)
# ======================
def _set_mode(m: str):
    st.session_state["mode"] = m


if "mode" not in st.session_state:
    st.session_state["mode"] = "Monthly"

mode = st.session_state["mode"]
if mode not in ("Monthly", "Annual"):
    st.session_state["mode"] = "Monthly"
    mode = "Monthly"

st.sidebar.button(
    "자산배분 포트폴리오",
    type="primary" if mode == "Monthly" else "secondary",
    use_container_width=True,
    on_click=_set_mode,
    args=("Monthly",),
)
st.sidebar.button(
    "연금저축 포트폴리오",
    type="primary" if mode == "Annual" else "secondary",
    use_container_width=True,
    on_click=_set_mode,
    args=("Annual",),
)

st.divider()


# ======================
# 숫자 입력(콤마 표기) 유틸
# ======================
def parse_money(text: str, allow_decimal: bool) -> float:
    if text is None:
        return 0.0
    s = str(text).strip()
    if s == "":
        return 0.0
    s = s.replace(",", "").replace("₩", "").replace("$", "").replace(" ", "")
    v = float(s)
    return float(v) if allow_decimal else float(int(v))


def format_money(value: float, allow_decimal: bool) -> str:
    if allow_decimal:
        s = f"{float(value):,.2f}"
        s = s.rstrip("0").rstrip(".")
        return s
    return f"{int(value):,}"


def money_input(
    label: str,
    key: str,
    default: float = 0.0,
    allow_decimal: bool = False,
    help_text: str = "",
) -> float:
    if key not in st.session_state:
        st.session_state[key] = format_money(default, allow_decimal)

    def _fmt():
        try:
            v = parse_money(st.session_state.get(key, ""), allow_decimal)
            st.session_state[key] = format_money(v, allow_decimal)
        except Exception:
            pass

    st.text_input(label, key=key, help=help_text, on_change=_fmt)

    try:
        return parse_money(st.session_state.get(key, ""), allow_decimal)
    except Exception:
        st.error(f"'{label}' 숫자 입력이 이상해. 예: 1,000,000 / 1000 / 1,000.50")
        st.stop()


# ======================
# Robust FRED CSV loader (BOM/whitespace/HTML guard)
# ======================
def _fred_csv(url: str) -> pd.DataFrame:
    """
    Robust CSV loader for FRED.
    - Strips BOM/whitespace in headers
    - Detects HTML response (blocked) early
    """
    text = None
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
        text = raw.decode("utf-8", errors="ignore")
        if "<html" in (text or "").lower():
            raise RuntimeError("FRED returned HTML instead of CSV.")
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        df = pd.read_csv(url)

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip().lstrip("\ufeff").lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    for c in df.columns:
        cl = str(c).lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    return None


# ======================
# yfinance / FRED (캐시)
# ======================
@st.cache_data(ttl=900, show_spinner=False)
def _download_hist_one(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} 가격 데이터를 못 가져옴 (history empty).")

    df = df.copy()

    # ✅ MultiIndex 방어
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    # tz 제거(있으면)
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    return df


@st.cache_data(ttl=300, show_spinner=False)
def last_adj_close(ticker: str) -> float:
    df = yf.download(ticker, period="7d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} 최근 가격을 못 가져옴 (history empty).")

    df = df.copy()

    # ✅ MultiIndex 방어
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    v = df[col].iloc[-1]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return float(v)


@st.cache_data(ttl=300, show_spinner=False)
def fx_usdkrw() -> float:
    ticker = "USDKRW=X"
    df = yf.download(ticker, period="7d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("USDKRW=X 환율을 못 가져옴 (history empty).")

    df = df.copy()

    # ✅ MultiIndex 방어
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    v = df[col].iloc[-1]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return float(v)


@st.cache_data(ttl=3600, show_spinner=False)
def _unrate_info(today: datetime):
    """
    ✅ FRED(UNRATE) robust fetch (BOM/whitespace/HTML guard)
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    df = _fred_csv(url)

    date_col = _pick_col(df, ["DATE", "observation_date"])
    val_col = _pick_col(df, ["UNRATE"])
    if not date_col or not val_col:
        raise RuntimeError("UNRATE columns not found")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col, val_col])

    start = today - timedelta(days=400)
    df = df[(df[date_col] >= start) & (df[date_col] <= today)].copy()
    if df.empty:
        raise RuntimeError("UNRATE 데이터가 비어있음")

    unrate_now = float(df[val_col].iloc[-1])
    unrate_ma = float(df[val_col].tail(12).mean())  # 최근 12개월 평균
    return unrate_now, unrate_ma


def price_asof_or_before(df: pd.DataFrame, dt: datetime) -> float:
    s = df.loc[df.index <= dt, "Adj Close"]
    if s.empty:
        return float(df["Adj Close"].iloc[0])
    v = s.iloc[-1]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return float(v)


# ======================
# 전략 로직
# ======================
def momentum_score(t: str, prices: dict, d_1m, d_3m, d_6m, d_12m) -> float:
    """가중합 모멘텀 (Adj Close): 1m*12 + 3m*6 + 6m*3 + 12m*1"""
    try:
        hist = _download_hist_one(t, period="2y")
        p = float(prices[t])

        p1 = price_asof_or_before(hist, d_1m)
        p3 = price_asof_or_before(hist, d_3m)
        p6 = price_asof_or_before(hist, d_6m)
        p12 = price_asof_or_before(hist, d_12m)

        return (p / p1 - 1) * 12 + (p / p3 - 1) * 6 + (p / p6 - 1) * 3 + (p / p12 - 1) * 1
    except Exception:
        return -9999


def strategy_value_holdings(holdings: dict, price_map: dict) -> float:
    total = 0.0
    for t, q in holdings.items():
        if t not in price_map:
            continue
        total += float(price_map[t]) * int(q)
    return float(total)


def holdings_weights(holdings: dict, price_map: dict) -> dict:
    total = strategy_value_holdings(holdings, price_map)
    if total <= 0:
        return {}
    return {t: (float(price_map[t]) * int(q)) / total for t, q in holdings.items() if t in price_map}


def needs_rebalance(current_weights: dict, target_weights: dict, band: float) -> bool:
    for t in set(current_weights.keys()) | set(target_weights.keys()):
        cur = float(current_weights.get(t, 0.0))
        tgt = float(target_weights.get(t, 0.0))
        if abs(cur - tgt) >= band:
            return True
    return False


def buy_weighted_min_cash(target_weights: dict, budget_usd: float, prices: dict):
    holdings = {}
    cash = float(budget_usd)
    for t, w in target_weights.items():
        if w <= 0:
            continue
        p = float(prices[t])
        alloc = float(budget_usd) * float(w)
        q = int(alloc // p)
        holdings[t] = int(q)
        cash -= float(q) * p

    if holdings:
        cheapest = min(target_weights.keys(), key=lambda a: float(prices[a]))
        p = float(prices[cheapest])
        if cash >= p:
            q = int(cash // p)
            holdings[cheapest] = holdings.get(cheapest, 0) + int(q)
            cash -= float(q) * p

    holdings = {t: int(q) for t, q in holdings.items() if int(q) != 0}
    return holdings, float(cash)


def add_cash_by_weights(holdings: dict, add_cash: float, target_weights: dict, prices: dict):
    if add_cash <= 0:
        return holdings, 0.0
    add_hold, cash_left = buy_weighted_min_cash(target_weights, add_cash, prices)
    merged = merge_holdings(holdings, add_hold)
    return merged, float(cash_left)


def buy_all_in(asset: str, budget_usd: float, prices: dict):
    p = float(prices[asset])
    q = int(float(budget_usd) // p)
    cash = float(float(budget_usd) - q * p)
    return {asset: q}, cash


def buy_all_in_if_affordable(asset: str, budget_usd: float, prices: dict):
    p = float(prices[asset])
    if float(budget_usd) < p:
        return {}, float(budget_usd)
    return buy_all_in(asset, budget_usd, prices)


def buy_equal_split_round(assets: list, budget_usd: float, prices: dict):
    n = len(assets)
    each = float(budget_usd) / n
    holdings = {a: 0 for a in assets}
    spent = 0.0

    for a in assets:
        p = float(prices[a])
        q = int(each // p)
        holdings[a] += q
        spent += q * p

    cash = float(float(budget_usd) - spent)
    return holdings, cash


def buy_equal_split_min_cash(assets: list, budget_usd: float, prices: dict):
    total_hold = {a: 0 for a in assets}
    cash = float(budget_usd)

    while True:
        round_hold, new_cash = buy_equal_split_round(assets, cash, prices)

        bought_any = any(q > 0 for q in round_hold.values())
        for a, q in round_hold.items():
            total_hold[a] += int(q)

        cash = float(new_cash)

        if not bought_any:
            break
        if cash < min(float(prices[a]) for a in assets):
            break

    cheapest = min(assets, key=lambda a: float(prices[a]))
    if cash >= float(prices[cheapest]):
        q = int(cash // float(prices[cheapest]))
        total_hold[cheapest] += q
        cash = float(cash - q * float(prices[cheapest]))

    total_hold = {a: int(q) for a, q in total_hold.items() if int(q) != 0}
    return total_hold, float(cash)


def safe_laa_asset(today: datetime, prices: dict) -> str:
    try:
        unrate_now, unrate_ma = _unrate_info(today)
    except Exception:
        return "QQQ"

    spy_hist = _download_hist_one("SPY", period="2y")
    spy_200ma = spy_hist["Adj Close"].rolling(200).mean().iloc[-1]
    if spy_200ma != spy_200ma:
        return "QQQ"

    risk_off = (float(prices["SPY"]) < float(spy_200ma)) or (unrate_now > unrate_ma)
    return "SHY" if risk_off else "QQQ"


def r12_return(prices: dict, t: str, d_12m: datetime) -> float:
    hist = _download_hist_one(t, period="2y")
    p0 = price_asof_or_before(hist, d_12m)
    return float(prices[t]) / float(p0) - 1


def infer_prev_pick(holdings: dict, candidates: list[str]) -> str | None:
    for t in candidates:
        if int(holdings.get(t, 0)) > 0:
            return t
    return None


def merge_holdings(*holding_dicts):
    merged = {}
    for h in holding_dicts:
        for t, q in h.items():
            merged[t] = merged.get(t, 0) + int(q)
    return merged


def vaa_scores_df(vaa: dict) -> pd.DataFrame:
    scores = vaa.get("scores", {})
    rows = [{"Ticker": t, "Momentum Score": float(scores.get(t, -9999))} for t in VAA_UNIVERSE]
    df = pd.DataFrame(rows).sort_values("Momentum Score", ascending=False, ignore_index=True)
    return df


def vaa_target_weights(scores: dict) -> tuple[dict, dict]:
    risk_on = all(float(scores.get(t, -9999)) >= 0 for t in VAA_SIGNAL)
    if risk_on:
        ranked = sorted(VAA_RISK_ON, key=lambda t: float(scores.get(t, -9999)), reverse=True)
        weights = {ranked[0]: 0.70, ranked[1]: 0.30}
        meta = {"picked": [ranked[0], ranked[1]], "risk": "on"}
        return weights, meta
    ranked = sorted(VAA_DEFENSIVE, key=lambda t: float(scores.get(t, -9999)), reverse=True)
    weights = {ranked[0]: 1.0}
    meta = {"picked": [ranked[0]], "risk": "off"}
    return weights, meta


def tdm_target_weights(prices: dict, d_12m: datetime, prev_pick: str | None) -> tuple[dict, dict]:
    r_spy = r12_return(prices, "SPY", d_12m)
    r_efa = r12_return(prices, "EFA", d_12m)
    r_bil = r12_return(prices, "BIL", d_12m)

    if r_spy >= r_bil:
        top = "SPY" if r_spy >= r_efa else "EFA"
        if prev_pick in ["SPY", "EFA"] and top != prev_pick and abs(r_spy - r_efa) < 0.05:
            top = prev_pick
        weights = {top: 1.0}
        meta = {"picked": [top], "mode": "attack", "r12": {"SPY": r_spy, "EFA": r_efa, "BIL": r_bil}}
        return weights, meta

    r_agg = r12_return(prices, "AGG", d_12m)
    r_shy = r12_return(prices, "SHY", d_12m)
    r_ief = r12_return(prices, "IEF", d_12m)
    ranked = sorted(
        [("AGG", r_agg), ("SHY", r_shy), ("IEF", r_ief)],
        key=lambda x: float(x[1]),
        reverse=True,
    )
    top = ranked[0][0]
    weights = {top: 1.0}
    meta = {
        "picked": [top],
        "mode": "defense",
        "r12": {"AGG": r_agg, "SHY": r_shy, "IEF": r_ief, "BIL": r_bil, "SPY": r_spy},
    }
    return weights, meta


def laa_target_weights(today: datetime, prices: dict, prev_pick: str | None) -> tuple[dict, dict]:
    laa_safe = safe_laa_asset(today, prices)
    weights = {"IWD": 0.25, "GLD": 0.25, "IEF": 0.25, laa_safe: 0.25}
    meta = {"safe": laa_safe, "prev_safe": prev_pick}
    return weights, meta


# ======================
# 결과 표시(UI 정리 버전)
# ======================
def show_result(result: dict, current_holdings: dict, layout: str = "side"):
    rate = float(result["meta"]["usdkrw_rate"])
    price_map = result["meta"]["prices_adj_close"]

    vaa = result["VAA"]
    laa = result["LAA"]
    tdm = result["TDM"]

    vaa_h = vaa["holdings"]
    laa_h = laa["holdings"]
    tdm_h = tdm["holdings"]

    vaa_cash = float(vaa.get("cash_usd", 0.0))
    laa_cash = float(laa.get("cash_usd", 0.0))
    tdm_cash = float(tdm.get("cash_usd", 0.0))

    def holdings_value_usd(h):
        return sum(float(price_map[t]) * int(q) for t, q in h.items() if t in price_map)

    total_holdings_usd = holdings_value_usd(vaa_h) + holdings_value_usd(laa_h) + holdings_value_usd(tdm_h)
    total_cash_usd = vaa_cash + laa_cash + tdm_cash
    total_usd = total_holdings_usd + total_cash_usd

    total_krw = total_usd * rate
    cash_krw = total_cash_usd * rate

    a, b, c = st.columns(3)
    a.metric("총자산(₩)", f"₩{total_krw:,.0f}")
    b.metric("현금(₩)", f"₩{cash_krw:,.0f}")
    c.metric("달러환율(₩/$)", f"₩{rate:,.2f}")

    all_target = merge_holdings(vaa_h, laa_h, tdm_h)

    def render_target_clean():
        st.subheader("목표 보유자산")
        items = [(t, int(q)) for t, q in all_target.items() if int(q) != 0 and t != "BIL"]
        items.sort(key=lambda x: x[0])

        if not items:
            st.write("-")
            return

        cols = st.columns(5)
        for i, (t, q) in enumerate(items):
            with cols[i % 5]:
                st.metric(t, f"{q}주")

    def render_trades_clean():
        st.subheader("매도/매수")
        rows = []
        for t in sorted(set(current_holdings.keys()) | set(all_target.keys())):
            if t == "BIL":
                continue
            cur = int(current_holdings.get(t, 0))
            tar = int(all_target.get(t, 0))
            delta = tar - cur
            if delta != 0:
                rows.append((t, delta))

        if not rows:
            st.write("-")
            return

        rows.sort(key=lambda x: (abs(x[1]), x[0]), reverse=True)

        sells = [(t, -d) for t, d in rows if d < 0]
        buys = [(t, d) for t, d in rows if d > 0]

        left, right = st.columns(2)
        with left:
            st.markdown("**매도**")
            if not sells:
                st.write("-")
            else:
                for t, q in sells:
                    st.write(f"{t} {q}주 매도")
        with right:
            st.markdown("**매입**")
            if not buys:
                st.write("-")
            else:
                for t, q in buys:
                    st.write(f"{t} {q}주 매입")

    def render_scores_bar():
        st.subheader("모멘텀스코어")
        df = vaa_scores_df(vaa)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Ticker:N", sort=alt.SortField(field="Momentum Score", order="descending")),
                y=alt.Y("Momentum Score:Q"),
                tooltip=["Ticker:N", alt.Tooltip("Momentum Score:Q", format=".4f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

    def render_portfolio_pie():
        st.subheader("포트폴리오 현황")
        cat_map = {
            "주식": {"SPY", "EFA", "EEM", "IWD"},
            "채권": {"IEF", "SHY"},
            "금": {"GLD"},
            "현금": {"BIL"},
        }

        def add_value(values: dict, t: str, q: int):
            if t not in price_map:
                return
            v = float(price_map[t]) * int(q)
            for cat, tickers in cat_map.items():
                if t in tickers:
                    values[cat] = values.get(cat, 0.0) + v
                    return

        values = {k: 0.0 for k in cat_map.keys()}
        for t, q in all_target.items():
            add_value(values, t, int(q))

        cash_total = float(total_cash_usd)
        values["현금"] = values.get("현금", 0.0) + cash_total

        total = sum(values.values())
        if total <= 0:
            st.write("-")
            return

        rows = []
        for cat, v in values.items():
            pct = int(round((v / total) * 100))
            rows.append(
                {"Category": cat, "Value": float(v), "Pct": pct, "Label": f"{cat}\n{pct}%"}
            )

        df = pd.DataFrame(rows)
        outer_radius = 120
        chart = (
            alt.Chart(df)
            .mark_arc(outerRadius=outer_radius)
            .encode(
                theta=alt.Theta("Value:Q"),
                color=alt.Color("Category:N", legend=alt.Legend(title="분류")),
                tooltip=["Category:N", alt.Tooltip("Value:Q", format=",.2f"), "Label:N"],
            )
            .properties(height=280, width=280)
        )
        st.altair_chart(chart, use_container_width=True)

    if layout == "side":
        left, right = st.columns([2, 1], gap="large")
        with right:
            render_scores_bar()
            render_portfolio_pie()
        with left:
            render_target_clean()
            render_trades_clean()
    else:
        render_scores_bar()
        render_portfolio_pie()
        render_target_clean()
        render_trades_clean()


# ======================
# 실행본 편집 + 저장(JSON: ETF만)
# ======================
def _clear_keys_with_prefix(prefix: str):
    for k in list(st.session_state.keys()):
        if k.startswith(prefix):
            del st.session_state[k]


def export_holdings_only(executed: dict, timestamp: str) -> dict:
    """
    ✅ 저장 JSON에는 현금 제외, ETF holdings만 저장
    executed = {"VAA":{...}, "LAA":{...}, "TDM":{...}} (각각 holdings dict)
    """
    payload = {
        "timestamp": timestamp,
        "schema_version": "holdings_only_v1",
        "VAA": {"holdings": {t: int(q) for t, q in executed["VAA"]["holdings"].items() if int(q) != 0}},
        "LAA": {"holdings": {t: int(q) for t, q in executed["LAA"]["holdings"].items() if int(q) != 0}},
        "TDM": {"holdings": {t: int(q) for t, q in executed["TDM"]["holdings"].items() if int(q) != 0}},
    }
    return payload


def render_execution_editor(result: dict, editor_prefix: str):
    st.subheader("실제 보유자산")

    strategy_labels = {
        "VAA": "Vigilant Asset Allocation",
        "LAA": "Lethargic Asset Allocation",
        "TDM": "Tactical Dual Momentum",
    }
    executed = {"VAA": {"holdings": {}}, "LAA": {"holdings": {}}, "TDM": {"holdings": {}}}

    for strat in ["VAA", "LAA", "TDM"]:
        rec = result[strat]["holdings"]

        with st.expander(strategy_labels.get(strat, strat), expanded=(strat == "VAA")):
            cols = st.columns(5)
            for i, t in enumerate(STRATEGY_TICKERS.get(strat, [])):
                default_q = int(rec.get(t, 0))
                key = f"{editor_prefix}{strat}_{t}"
                with cols[i % 5]:
                    q = st.number_input(t, min_value=0, value=default_q, step=1, key=key)

                if int(q) != 0:
                    executed[strat]["holdings"][t] = int(q)

    return executed


# ======================
# 날짜 기준
# ======================
today = datetime.today()
today_naive = today.replace(tzinfo=None)
d_1m, d_3m, d_6m, d_12m = [today_naive - timedelta(days=d) for d in [30, 90, 180, 365]]


# ======================
# 시장 데이터 로드
# ======================
with st.spinner("가격/환율 불러오는 중..."):
    prices = {t: last_adj_close(t) for t in TICKER_LIST}
    usdkrw_rate = fx_usdkrw()


# ======================
# 실행 함수
# ======================
def compute_vaa_scores(prices: dict) -> dict:
    return {t: float(momentum_score(t, prices, d_1m, d_3m, d_6m, d_12m)) for t in VAA_UNIVERSE}


def run_year(amounts: dict, cash_usd: float):
    total_usd = sum(float(amounts.get(t, 0.0)) * float(prices[t]) for t in INPUT_TICKERS) + float(cash_usd)
    budgets = {k: float(total_usd) * float(w) for k, w in STRATEGY_WEIGHTS.items()}

    scores = compute_vaa_scores(prices)
    vaa_weights, vaa_meta = vaa_target_weights(scores)
    vaa_hold, vaa_cash_usd = buy_weighted_min_cash(vaa_weights, budgets["VAA"], prices)

    laa_weights, laa_meta = laa_target_weights(today, prices, prev_pick=None)
    laa_hold, laa_cash_usd = buy_weighted_min_cash(laa_weights, budgets["LAA"], prices)

    tdm_weights, tdm_meta = tdm_target_weights(prices, d_12m, prev_pick=None)
    tdm_hold, tdm_cash_usd = buy_weighted_min_cash(tdm_weights, budgets["TDM"], prices)

    return {
        "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "usdkrw_rate": float(usdkrw_rate),
            "prices_adj_close": {t: float(prices[t]) for t in TICKER_LIST},
            "input_cash_usd": float(cash_usd),
            "cash_rule": "Annual: total_usd includes input cash; budget split by strategy weights. (Cash is NOT saved to JSON.)",
        },
        "VAA": {
            "holdings": vaa_hold,
            "cash_usd": float(vaa_cash_usd),
            "picked": vaa_meta.get("picked"),
            "risk": vaa_meta.get("risk"),
            "scores": scores,
        },
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_meta.get("safe")},
        "TDM": {"holdings": tdm_hold, "cash_usd": float(tdm_cash_usd), "picked": tdm_meta.get("picked")},
    }


def run_month(prev: dict, cash_usd: float):
    cash_total = float(cash_usd)

    vaa_prev_hold = prev["VAA"]["holdings"]
    laa_prev_hold = prev["LAA"]["holdings"]
    tdm_prev_hold = prev["TDM"]["holdings"]

    values = {
        "VAA": strategy_value_holdings(vaa_prev_hold, prices),
        "LAA": strategy_value_holdings(laa_prev_hold, prices),
        "TDM": strategy_value_holdings(tdm_prev_hold, prices),
    }
    total_holdings = float(values["VAA"] + values["LAA"] + values["TDM"])

    rebalance_all = total_holdings <= 0
    if total_holdings > 0:
        for k, w in STRATEGY_WEIGHTS.items():
            cur_w = float(values[k]) / total_holdings
            if abs(cur_w - float(w)) >= REBALANCE_BAND:
                rebalance_all = True
                break

    total_portfolio = float(total_holdings + cash_total)
    if rebalance_all:
        budgets = {k: total_portfolio * float(w) for k, w in STRATEGY_WEIGHTS.items()}
    else:
        budgets = {k: float(values[k]) + cash_total * float(w) for k, w in STRATEGY_WEIGHTS.items()}

    scores = compute_vaa_scores(prices)
    vaa_weights, vaa_meta = vaa_target_weights(scores)
    vaa_current_weights = holdings_weights(vaa_prev_hold, prices)
    vaa_internal_reb = needs_rebalance(vaa_current_weights, vaa_weights, REBALANCE_BAND) or not vaa_prev_hold
    vaa_full = rebalance_all or vaa_internal_reb
    if vaa_full:
        vaa_hold, vaa_cash_usd = buy_weighted_min_cash(vaa_weights, budgets["VAA"], prices)
    else:
        add_cash = float(budgets["VAA"] - values["VAA"])
        vaa_hold, vaa_cash_usd = add_cash_by_weights(vaa_prev_hold, add_cash, vaa_weights, prices)

    laa_prev_pick = infer_prev_pick(laa_prev_hold, ["QQQ", "SHY"])
    laa_weights, laa_meta = laa_target_weights(today, prices, prev_pick=laa_prev_pick)
    laa_current_weights = holdings_weights(laa_prev_hold, prices)
    laa_internal_reb = needs_rebalance(laa_current_weights, laa_weights, REBALANCE_BAND) or not laa_prev_hold
    laa_force = laa_prev_pick is not None and laa_meta.get("safe") != laa_prev_pick
    laa_full = rebalance_all or laa_internal_reb or laa_force
    if laa_full:
        laa_hold, laa_cash_usd = buy_weighted_min_cash(laa_weights, budgets["LAA"], prices)
    else:
        add_cash = float(budgets["LAA"] - values["LAA"])
        laa_hold, laa_cash_usd = add_cash_by_weights(laa_prev_hold, add_cash, laa_weights, prices)

    tdm_prev_pick = infer_prev_pick(tdm_prev_hold, ["SPY", "EFA", "AGG", "SHY", "IEF"])
    tdm_weights, tdm_meta = tdm_target_weights(prices, d_12m, prev_pick=tdm_prev_pick)
    tdm_current_weights = holdings_weights(tdm_prev_hold, prices)
    tdm_internal_reb = needs_rebalance(tdm_current_weights, tdm_weights, REBALANCE_BAND) or not tdm_prev_hold
    tdm_full = rebalance_all or tdm_internal_reb
    if tdm_full:
        tdm_hold, tdm_cash_usd = buy_weighted_min_cash(tdm_weights, budgets["TDM"], prices)
    else:
        add_cash = float(budgets["TDM"] - values["TDM"])
        tdm_hold, tdm_cash_usd = add_cash_by_weights(tdm_prev_hold, add_cash, tdm_weights, prices)

    return {
        "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "usdkrw_rate": float(usdkrw_rate),
            "prices_adj_close": {t: float(prices[t]) for t in TICKER_LIST},
            "input_cash_usd": float(cash_usd),
            "cash_rule": "Monthly: cash split by strategy weights; rebalance only when drift >= 10%p. (Cash is NOT saved to JSON.)",
        },
        "VAA": {
            "holdings": vaa_hold,
            "cash_usd": float(vaa_cash_usd),
            "picked": vaa_meta.get("picked"),
            "risk": vaa_meta.get("risk"),
            "scores": scores,
        },
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_meta.get("safe")},
        "TDM": {"holdings": tdm_hold, "cash_usd": float(tdm_cash_usd), "picked": tdm_meta.get("picked")},
    }



# ======================
# 화면: Annual / Monthly
# ======================
if mode == "Annual":
    st.header("Annual Rebalancing")

    st.subheader("Assets")
    amounts = {}

    # ✅ 10개 티커 + 현금 = 11개 → 6칸 그리드
    fields = INPUT_TICKERS + ["현금($)"]
    cols = st.columns(6)

    cash_usd = 0.0
    for i, f in enumerate(fields):
        with cols[i % 6]:
            if f == "현금($)":
                cash_usd = money_input("현금($)", key="y_cash_usd", default=0, allow_decimal=True)
            else:
                amounts[f] = st.number_input(f, min_value=0, value=0, step=1, key=f"y_amt_{f}")

    run_btn = st.button("리밸런싱", type="primary")
    if run_btn:
        try:
            with st.spinner("계산 중..."):
                result = run_year(amounts, cash_usd)
            st.session_state["annual_result"] = result
            _clear_keys_with_prefix("exec_annual_")  # 실행본 편집 키 초기화
            st.success("Completed")
        except Exception as e:
            st.error(str(e))

    if "annual_result" in st.session_state:
        result = st.session_state["annual_result"]
        current_holdings = {t: int(amounts.get(t, 0)) for t in INPUT_TICKERS}

        show_result(result, current_holdings, layout="side")
        st.divider()

        executed = render_execution_editor(result, editor_prefix="exec_annual_")
        payload = export_holdings_only(executed, timestamp=result["timestamp"])

        st.download_button(
            label="저장",
            data=json.dumps(payload, indent=2),
            file_name=f"rebalance_exec_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
            mime="application/json",
            use_container_width=False,
        )

if st.sidebar.button("새로고침", use_container_width=True):
    st.cache_data.clear()
    # 실행 결과/편집 상태도 같이 초기화
    for k in list(st.session_state.keys()):
        if k.startswith(
            (
                "annual_result",
                "monthly_result",
                "exec_annual_",
                "exec_monthly_",
                "monthly_file_sig",
            )
        ):
            del st.session_state[k]
    st.rerun()

elif mode == "Monthly":
    st.header("자산배분 포트폴리오")

    uploaded = st.file_uploader("File Upload", type=["json"])
    cash_usd = money_input("현금($)", key="m_cash_usd", default=0, allow_decimal=True)

    prev = {"VAA": {"holdings": {}}, "LAA": {"holdings": {}}, "TDM": {"holdings": {}}}
    if uploaded:
        raw_bytes = uploaded.getvalue()
        file_sig = hashlib.md5(raw_bytes).hexdigest()

        if st.session_state.get("monthly_file_sig") != file_sig:
            st.session_state["monthly_file_sig"] = file_sig
            if "monthly_result" in st.session_state:
                del st.session_state["monthly_result"]
            _clear_keys_with_prefix("exec_monthly_")

        try:
            prev_raw = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            st.error("업로드 파일이 JSON 파싱에 실패했어. (파일 깨짐/형식 오류)")
            st.stop()

        if "TDM" not in prev_raw and "ODM" in prev_raw:
            prev_raw["TDM"] = prev_raw.get("ODM", {})
        for k in ["VAA", "LAA", "TDM"]:
            if k not in prev_raw or "holdings" not in prev_raw[k]:
                st.error("이 JSON은 예상 형식이 아니야. (VAA/LAA/TDM 안에 holdings가 필요)")
                st.stop()

        prev = json.loads(json.dumps(prev_raw))  # deep copy

        st.subheader("현재 보유자산")
        merged_prev = merge_holdings(prev["VAA"]["holdings"], prev["LAA"]["holdings"], prev["TDM"]["holdings"])
        items = [(t, int(q)) for t, q in merged_prev.items() if int(q) != 0]
        items.sort(key=lambda x: x[0])
        if not items:
            st.write("-")
        else:
            cols = st.columns(5)
            for i, (t, q) in enumerate(items):
                with cols[i % 5]:
                    st.metric(t, f"{q}주")

    run_btn = st.button("리밸런싱", type="primary")
    if run_btn:
        try:
            with st.spinner("Calculating..."):
                result = run_month(prev, cash_usd)
            st.session_state["monthly_result"] = result
            _clear_keys_with_prefix("exec_monthly_")
            st.success("Completed")
        except Exception as e:
            st.error(str(e))

    if "monthly_result" in st.session_state:
        result = st.session_state["monthly_result"]
        current_holdings = merge_holdings(prev["VAA"]["holdings"], prev["LAA"]["holdings"], prev["TDM"]["holdings"])

        show_result(result, current_holdings, layout="side")
        st.divider()

        executed = render_execution_editor(result, editor_prefix="exec_monthly_")
        payload = export_holdings_only(executed, timestamp=result["timestamp"])

        st.download_button(
            label="저장",
            data=json.dumps(payload, indent=2),
            file_name=f"rebalance_exec_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
            mime="application/json",
            use_container_width=False,
        )
