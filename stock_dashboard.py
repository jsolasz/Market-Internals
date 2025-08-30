#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 14:37:14 2025

@author: jacksolasz

"""


import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os, json, time, pathlib, warnings
import numpy as np

# SciPy is optional for KDE but provides better results.
try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Market Internals Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Constants and Configuration ---
TICKERS_TTL_HOURS = 6       # Cache S&P list for 6 hours
CACHE_DIR = ".dashboard_cache"    # Local cache folder
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SECTOR_ETF_MAP = {
    'Information Technology': 'XLK',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB'
}


# --- Custom CSS for Theming ---
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        h1, h2, h3, .st-emotion-cache-16txtl3 {
            color: #FAFAFA;
        }
        .st-emotion-cache-10trblm {
            color: #A0AEC0;
        }
    </style>
""", unsafe_allow_html=True)


# --- Robust Data Fetching and Caching ---
_cache_path = pathlib.Path(CACHE_DIR); _cache_path.mkdir(parents=True, exist_ok=True)

def _tickers_cache_file(): return _cache_path / "sp500_tickers.json"

def _load_cached_tickers():
    f = _tickers_cache_file()
    if not f.exists(): return None
    try:
        obj = json.loads(f.read_text())
        if time.time() - obj.get("_ts", 0) > TICKERS_TTL_HOURS * 3600:
            return None
        return obj.get("tickers", None)
    except Exception:
        return None

def _save_cached_tickers(list_of_dicts):
    try:
        payload = {"_ts": time.time(), "tickers": list_of_dicts}
        _tickers_cache_file().write_text(json.dumps(payload))
    except Exception:
        pass

@st.cache_data(ttl=TICKERS_TTL_HOURS * 3600)
def get_sp500_tickers():
    """Robustly fetches S&P 500 tickers and sectors using multiple fallbacks."""
    cached = _load_cached_tickers()
    if cached:
        return pd.DataFrame(cached)

    # 1. Primary Source: Wikipedia
    try:
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(wiki_url, storage_options={"User-Agent": UA})
        constituents_table = next(tbl for tbl in tables if 'Symbol' in tbl.columns and 'GICS Sector' in tbl.columns)
        df = constituents_table[['Symbol', 'GICS Sector']].copy()
        df['Symbol'] = df['Symbol'].str.replace(".", "-", regex=False).str.strip()
        # FIX: Do NOT drop duplicates or filter out GOOG/GOOGL. The S&P 500 includes multiple
        # share classes for some companies, and all are needed for accurate volume.
        _save_cached_tickers(df.to_dict('records'))
        return df
    except Exception:
        pass

    # 2. Fallback Source: Slickcharts
    try:
        resp = requests.get("https://www.slickcharts.com/sp500", headers={"User-Agent": UA}, timeout=15)
        resp.raise_for_status()
        tbl = pd.read_html(resp.text)[0]
        df = tbl[['Symbol']].copy()
        df['GICS Sector'] = 'N/A'
        df['Symbol'] = df['Symbol'].str.replace(".", "-", regex=False).str.strip()
        _save_cached_tickers(df.to_dict('records'))
        return df
    except Exception as e:
        st.error(f"Could not download S&P 500 tickers from any source. Last error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_data(tickers):
    """Fetches historical and intraday data for a list of tickers."""
    ticker_str = " ".join(tickers)
    data_daily = yf.download(ticker_str, period="1y", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
    data_intraday = yf.download(ticker_str, period="1d", interval="5m", group_by='ticker', auto_adjust=True, progress=False)
    return data_daily, data_intraday

@st.cache_data(ttl=3600*6) # Cache seasonality data for 6 hours
def get_seasonality_data():
    """Fetches and calculates SPX seasonality for the last 50 years."""
    start_date = (datetime.now() - pd.DateOffset(years=51)).strftime('%Y-%m-%d')
    spx_hist = yf.download('^GSPC', start=start_date, auto_adjust=True, progress=False)
    
    if spx_hist.empty:
        return None, None

    spx_hist['Daily % Change'] = spx_hist['Close'].pct_change()
    spx_hist['DayOfYear'] = spx_hist.index.dayofyear
    spx_hist['Year'] = spx_hist.index.year
    
    current_year = datetime.now().year
    historical_data = spx_hist[spx_hist['Year'] < current_year]

    avg_daily_returns = historical_data.groupby('DayOfYear')['Daily % Change'].mean()
    average_seasonality = (1 + avg_daily_returns).cumprod() - 1
    average_seasonality *= 100

    current_year_data = spx_hist[spx_hist['Year'] == current_year].copy()
    current_year_data['Cumulative YTD'] = ((1 + current_year_data['Daily % Change']).cumprod() - 1) * 100
    current_ytd = current_year_data[['DayOfYear', 'Cumulative YTD']].set_index('DayOfYear')
    
    return average_seasonality, current_ytd

# --- Helper functions for Distribution Plot ---
def weighted_stats(vals, wts):
    v = np.asarray(vals, float); w = np.asarray(wts, float)
    avg = np.sum(w * v) / np.sum(w) if np.sum(w) > 0 else np.nan
    pos_mask = v > 0
    avg_gain = np.sum(w[pos_mask] * v[pos_mask]) / np.sum(w[pos_mask]) if pos_mask.any() else np.nan
    neg_mask = v < 0
    avg_decline = np.sum(w[neg_mask] * v[neg_mask]) / np.sum(w[neg_mask]) if neg_mask.any() else np.nan
    return avg, avg_gain, avg_decline

def build_distribution(vals, wts, grid, bw=0.60):
    """Unified interface to produce a smooth density on `grid`."""
    vals = np.asarray(vals, float); wts = np.asarray(wts, float)
    if gaussian_kde:
        try:
            kde = gaussian_kde(vals, weights=wts)
            kde.set_bandwidth(kde.factor * bw)
            return kde(grid)
        except Exception: pass
    hist, edges = np.histogram(vals, bins=120, range=(grid.min(), grid.max()), weights=wts, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = np.interp(grid, centers, hist)
    y = np.convolve(y, np.ones(5) / 5, mode="same")
    return y

def run_dashboard():
    st.title("Market Internals Dashboard")
    st.markdown("An overview of S&P 500 market health and performance, powered by real-time data.")

    with st.spinner('Loading S&P 500 component data and market prices...'):
        sp500_df = get_sp500_tickers()
        if sp500_df.empty:
            st.error("Failed to retrieve S&P 500 ticker list. Dashboard cannot be loaded.")
            return

        tickers = sp500_df['Symbol'].tolist()
        sector_etfs = list(SECTOR_ETF_MAP.values())
        tickers_with_etfs = tickers + ['SPY', 'RSP', '^GSPC'] + sector_etfs
        daily_data, intraday_data = get_stock_data(tickers_with_etfs)
        
        if daily_data.empty or len(daily_data) < 201:
            st.error("Failed to load sufficient historical market data. Please try again later.")
            return

    # --- SPX Performance Metrics ---
    st.header("S&P 500 (SPX) Performance")
    spx_col = ('^GSPC', 'Close') if isinstance(daily_data.columns, pd.MultiIndex) else 'Close'
    if spx_col not in daily_data.columns:
        st.warning("Could not load S&P 500 index data for performance metrics.")
    else:
        spx_data = daily_data[spx_col].dropna()
        periods = {"1 Day": 1, "1 Week": 5, "1 Month": 21, "3 Months": 63, "YTD": None}
        cols = st.columns(5)
        for i, (period_name, period_days) in enumerate(periods.items()):
            change = 0.0
            if period_name == "YTD":
                ytd_start_date = spx_data.index[spx_data.index.year == datetime.now().year].min()
                start_price = spx_data.loc[ytd_start_date]
                if start_price > 0: change = ((spx_data.iloc[-1] - start_price) / start_price) * 100
            elif len(spx_data) > period_days:
                change = (spx_data.pct_change(period_days).iloc[-1]) * 100
            cols[i].metric(label=period_name, value=f"{change:.2f}%", delta_color="normal" if change >= 0 else "inverse")

    st.divider()
    
    # --- S&P 500 Volume Analysis ---
    st.header("S&P 500 Volume Analysis")
    valid_tickers_daily = [t for t in tickers if (t, 'Volume') in daily_data.columns]
    total_daily_sp500_volume = daily_data.xs('Volume', level=1, axis=1)[valid_tickers_daily].sum(axis=1)
    avg_vol_90d = total_daily_sp500_volume.tail(30).mean()

    spx_change_daily = daily_data[('^GSPC', 'Close')].diff()
    vol_and_spx_change = pd.concat([total_daily_sp500_volume, spx_change_daily], axis=1).dropna()
    vol_and_spx_change.columns = ['TotalVolume', 'SPXChange']
    last_90_days_vol = vol_and_spx_change.tail(30)
    
    up_days_volume = last_90_days_vol[last_90_days_vol['SPXChange'] > 0]['TotalVolume'].mean()
    down_days_volume = last_90_days_vol[last_90_days_vol['SPXChange'] < 0]['TotalVolume'].mean()
    
    def format_volume(vol):
        if pd.isna(vol): return "N/A"
        if vol > 1_000_000_000: return f"{vol / 1_000_000_000:.2f}B"
        if vol > 1_000_000: return f"{vol / 1_000_000:.2f}M"
        if vol > 1_000: return f"{vol / 1_000:.2f}K"
        return f"{vol}"

    vol_cols = st.columns(4)
    if not intraday_data.empty:
        valid_tickers_intraday = [t for t in tickers if (t, 'Volume') in intraday_data.columns]
        current_total_sp500_volume = intraday_data.xs('Volume', level=1, axis=1)[valid_tickers_intraday].sum().sum()
        vol_cols[0].metric("Current Daily Volume", format_volume(current_total_sp500_volume))
    else:
        vol_cols[0].metric("Current Daily Volume", "N/A")
    vol_cols[1].metric("Avg Volume (30-day)", format_volume(avg_vol_90d))
    vol_cols[2].metric("Avg Volume (Up Days)", format_volume(up_days_volume))
    vol_cols[3].metric("Avg Volume (Down Days)", format_volume(down_days_volume))

    st.divider()

    if intraday_data.empty or len(intraday_data) < 2:
        st.warning("Intraday data is not available (market may be closed).")
    else:
        try:
            intraday_data.index = intraday_data.index.tz_convert('America/New_York')
        except TypeError:
            intraday_data.index = intraday_data.index.tz_localize('UTC').tz_convert('America/New_York')
    
        valid_tickers = [t for t in tickers if (t, 'Close') in daily_data.columns and (t, 'Close') in intraday_data.columns]
        last_close = daily_data.xs('Close', level=1, axis=1)[valid_tickers].iloc[-2]
        current_price = intraday_data.xs('Close', level=1, axis=1)[valid_tickers].iloc[-1]
        price_change = current_price - last_close
        
        advancing = (price_change > 0).sum()
        declining = (price_change < 0).sum()
        
        intraday_volumes = intraday_data.xs('Volume', level=1, axis=1)[valid_tickers]
        total_share_volume = intraday_volumes.sum()
        adv_share_volume = total_share_volume[price_change > 0].sum()
        dec_share_volume = total_share_volume[price_change < 0].sum()
        
        intraday_closes = intraday_data.xs('Close', level=1, axis=1)[valid_tickers]
        dollar_volume_per_bar = intraday_closes * intraday_volumes
        total_dollar_volume = dollar_volume_per_bar.sum()

        up_dollar_volume = total_dollar_volume[price_change > 0].sum()
        down_dollar_volume = total_dollar_volume[price_change < 0].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("S&P 500 Breadth (Advancers/Decliners)")
            st.markdown(f"Advancing: <span style='color:green;'>{advancing}</span> | Declining: <span style='color:red;'>{declining}</span>", unsafe_allow_html=True)
            if (advancing + declining) > 0: st.progress(advancing / (advancing + declining))
        with col2:
            st.subheader("Volume Breadth")
            up_vol_str = f"{adv_share_volume/1e9:.2f}B"
            down_vol_str = f"{dec_share_volume/1e9:.2f}B"
            st.markdown(f"Up Vol (Shares): <span style='color:green;'>{up_vol_str}</span> | Down Vol (Shares): <span style='color:red;'>{down_vol_str}</span>", unsafe_allow_html=True)
            if (adv_share_volume + dec_share_volume) > 0: st.progress(adv_share_volume / (adv_share_volume + dec_share_volume))
            up_dollar_str = f"{up_dollar_volume/1e9:.2f}B"
            down_dollar_str = f"{down_dollar_volume/1e9:.2f}B"
            st.markdown(f"Up Vol (USD): <font color='green'>{up_dollar_str}</font> | Down Vol (USD): <font color='red'>{down_dollar_str}</font>", unsafe_allow_html=True)
            if (up_dollar_volume + down_dollar_volume) > 0: st.progress(up_dollar_volume / (up_dollar_volume + down_dollar_volume))

        st.divider()

        spy_daily = daily_data['SPY'].dropna()

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Intraday Performance")
            spy_intra = (intraday_data[('SPY', 'Close')].pct_change().cumsum() * 100).fillna(0)
            rsp_intra = (intraday_data[('RSP', 'Close')].pct_change().cumsum() * 100).fillna(0)
            intraday_prices = intraday_data.xs('Close', level=1, axis=1)[valid_tickers].ffill()
            adv_dec_line = (intraday_prices > last_close).sum(axis=1) - (intraday_prices < last_close).sum(axis=1)

            fig_spy_rsp = go.Figure()
            fig_spy_rsp.add_trace(go.Scatter(x=spy_intra.index.time, y=spy_intra, mode='lines', name='SPY (Market Cap Weighted)', line=dict(color='#636EFA')))
            fig_spy_rsp.add_trace(go.Scatter(x=rsp_intra.index.time, y=rsp_intra, mode='lines', name='RSP (Equal Weighted)', line=dict(color='#FFA15A')))
            fig_spy_rsp.update_layout(title_text='SPY vs. RSP Intraday', yaxis_title='% Change', legend=dict(x=0.01, y=0.99), plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white')
            st.plotly_chart(fig_spy_rsp, use_container_width=True)
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                spy_intraday_full = intraday_data['SPY']
                typical_price = (spy_intraday_full['High'] + spy_intraday_full['Low'] + spy_intraday_full['Close']) / 3
                tp_volume = typical_price * spy_intraday_full['Volume']
                spy_intraday_full['VWAP'] = tp_volume.cumsum() / spy_intraday_full['Volume'].cumsum()
                spy_10d_sma_val = spy_daily['Close'].rolling(window=10).mean().iloc[-1]
                spy_20d_sma_val = spy_daily['Close'].rolling(window=20).mean().iloc[-1]

                fig_vwap = go.Figure()
                fig_vwap.add_trace(go.Scatter(x=spy_intraday_full.index.time, y=spy_intraday_full['Close'], mode='lines', name='SPY Price', line=dict(color='#636EFA')))
                fig_vwap.add_trace(go.Scatter(x=spy_intraday_full.index.time, y=spy_intraday_full['VWAP'], mode='lines', name='Intraday VWAP', line=dict(color='#FFA15A', dash='dash')))
                fig_vwap.add_hline(y=spy_10d_sma_val, line_width=1, line_dash="dash", line_color="cyan", annotation_text=f"10-d SMA: {spy_10d_sma_val:.2f}")
                fig_vwap.add_hline(y=spy_20d_sma_val, line_width=1, line_dash="dash", line_color="magenta", annotation_text=f"20-d SMA: {spy_20d_sma_val:.2f}")
                fig_vwap.update_layout(title='SPY Intraday Analysis', yaxis_title='Price', plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white', legend=dict(x=0.01, y=0.99))
                st.plotly_chart(fig_vwap, use_container_width=True)

            with sub_col2:
                spy_daily_monthly = spy_daily.tail(30).copy()
                spy_daily_monthly['SMA10'] = spy_daily_monthly['Close'].rolling(window=10).mean()
                spy_daily_monthly['SMA20'] = spy_daily_monthly['Close'].rolling(window=20).mean()
                fig_daily_spy = go.Figure()
                fig_daily_spy.add_trace(go.Candlestick(x=spy_daily_monthly.index, open=spy_daily_monthly['Open'], high=spy_daily_monthly['High'], low=spy_daily_monthly['Low'], close=spy_daily_monthly['Close'], name='SPY Daily'))
                fig_daily_spy.add_trace(go.Scatter(x=spy_daily_monthly.index, y=spy_daily_monthly['SMA10'], mode='lines', name='10-Day SMA', line=dict(color='cyan', width=1)))
                fig_daily_spy.add_trace(go.Scatter(x=spy_daily_monthly.index, y=spy_daily_monthly['SMA20'], mode='lines', name='20-Day SMA', line=dict(color='magenta', width=1)))
                fig_daily_spy.update_layout(title='Last 30 Days', yaxis_title='Price', plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white', legend=dict(x=0.01, y=0.99), margin=dict(l=20, r=20, t=40, b=20), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_daily_spy, use_container_width=True)



        with col2:
            st.subheader("% Stocks Above MA")
            close_prices = daily_data.xs('Close', level=1, axis=1)[valid_tickers]
            ma_periods = [10, 20, 50, 200]
            for ma in ma_periods:
                if len(close_prices) >= ma:
                    sma = close_prices.rolling(window=ma).mean().iloc[-1]
                    above_ma = (close_prices.iloc[-1] > sma).sum()
                    below_ma = len(valid_tickers) - above_ma
                    metric_col, chart_col = st.columns([1, 1])
                    with metric_col:
                        st.metric(label=f"{ma}-Day MA", value=f"{above_ma / len(valid_tickers) * 100:.1f}%")
                    with chart_col:
                        fig_pie = go.Figure(data=[go.Pie(labels=['Above', 'Below'], values=[above_ma, below_ma], marker_colors=['#10b981', '#ef4444'], hole=.4, textinfo='none')])
                        fig_pie.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=80, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_pie, use_container_width=True, key=f"ma_pie_{ma}")
            
            st.subheader("SPY Daily Chart")
        
            fig_adv_dec = go.Figure()
            fig_adv_dec.add_trace(go.Scatter(x=adv_dec_line.index.time, y=adv_dec_line, mode='lines', name='Net A/D', line=dict(color='white')))
            fig_adv_dec.add_trace(go.Scatter(x=adv_dec_line.index.time, y=adv_dec_line.where(adv_dec_line >= 0), fill='tozeroy', mode='none', fillcolor='rgba(16, 185, 129, 0.5)'))
            fig_adv_dec.add_trace(go.Scatter(x=adv_dec_line.index.time, y=adv_dec_line.where(adv_dec_line <= 0), fill='tozeroy', mode='none', fillcolor='rgba(239, 68, 68, 0.5)'))
            fig_adv_dec.update_layout(title='Advancing - Declining Stocks (Intraday Net)', yaxis_range=[-505, 505], yaxis_title='Net Advancing Stocks', plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white', showlegend=False)
            st.plotly_chart(fig_adv_dec, use_container_width=True)



        st.divider()
        st.header("SPX Seasonality")
        avg_seasonality, current_ytd_seasonality = get_seasonality_data()
        if avg_seasonality is not None and current_ytd_seasonality is not None:
            fig_season = go.Figure()
            fig_season.add_trace(go.Scatter(x=avg_seasonality.index, y=avg_seasonality.values, mode='lines', name='50-Year Average', line=dict(color='#636EFA')))
            fig_season.add_trace(go.Scatter(x=current_ytd_seasonality.index, y=current_ytd_seasonality['Cumulative YTD'], mode='lines', name=f'{datetime.now().year} Performance', line=dict(color='#FFA15A', width=3)))
            tickvals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig_season.update_layout(title='Current YTD Performance vs. 50-Year Average', xaxis_title='Time of Year', yaxis_title='Cumulative Performance (%)', plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white', legend=dict(x=0.01, y=0.99), xaxis=dict(tickvals=tickvals, ticktext=ticktext, gridcolor='rgba(255, 255, 255, 0.1)'), yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'), height=800, yaxis_range=[-20, 20])
            st.plotly_chart(fig_season, use_container_width=True)
        else:
            st.warning("Could not load long-term data for seasonality chart.")
            
        st.divider()
        st.subheader("Sector Performance vs. SPY")
        sector_etf_tickers = list(SECTOR_ETF_MAP.values())
        etfs_for_perf = sector_etf_tickers + ['SPY']
        daily_closes = daily_data.xs('Close', level=1, axis=1)[etfs_for_perf]
        ytd_start_date = daily_closes.index[daily_closes.index.year == datetime.now().year].min()
        ytd_start_prices = daily_closes.loc[ytd_start_date]
        perf_ytd_all = ((daily_closes.iloc[-1] - ytd_start_prices) / ytd_start_prices) * 100
        spy_perf_ytd = perf_ytd_all['SPY']
        sector_perf_ytd = perf_ytd_all[sector_etf_tickers]
        relative_perf_ytd = (sector_perf_ytd - spy_perf_ytd).dropna()
        relative_perf_ytd_df = pd.DataFrame({'Sector': [k for k, v in SECTOR_ETF_MAP.items() if v in relative_perf_ytd.index], 'Relative Performance': relative_perf_ytd.values}).sort_values('Relative Performance', ascending=False)
        perf_1m_all = daily_closes.pct_change(periods=21).iloc[-1] * 100
        spy_perf_1m = perf_1m_all['SPY']
        sector_perf_1m = perf_1m_all[sector_etf_tickers]
        relative_perf_1m = (sector_perf_1m - spy_perf_1m).dropna()
        relative_perf_1m_df = pd.DataFrame({'Sector': [k for k, v in SECTOR_ETF_MAP.items() if v in relative_perf_1m.index], 'Relative Performance': relative_perf_1m.values}).sort_values('Relative Performance', ascending=False)
        last_close_etfs_all = daily_closes.iloc[-2]
        current_price_etfs_all = intraday_data.xs('Close', level=1, axis=1)[etfs_for_perf].iloc[-1]
        pct_change_etfs_all = ((current_price_etfs_all - last_close_etfs_all) / last_close_etfs_all * 100).dropna()
        spy_perf_daily = pct_change_etfs_all['SPY']
        sector_perf_daily = pct_change_etfs_all[sector_etf_tickers]
        relative_perf_daily = (sector_perf_daily - spy_perf_daily).dropna()
        relative_perf_daily_df = pd.DataFrame({'Sector': [k for k, v in SECTOR_ETF_MAP.items() if v in relative_perf_daily.index], 'Relative Performance': relative_perf_daily.values}).sort_values('Relative Performance', ascending=False)
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        with chart_col1:
            st.markdown("##### Daily Relative Performance")
            colors_daily = ['#10b981' if x >= 0 else '#ef4444' for x in relative_perf_daily_df['Relative Performance']]
            fig_daily = go.Figure(go.Bar(x=relative_perf_daily_df['Relative Performance'], y=relative_perf_daily_df['Sector'], orientation='h', marker_color=colors_daily))
            fig_daily.update_layout(title="vs. SPY (Today)", xaxis_title="Outperformance (%)", yaxis_title="", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white', yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_daily, use_container_width=True)
        with chart_col2:
            st.markdown("##### 1-Month Relative Performance")
            colors_1m = ['#10b981' if x >= 0 else '#ef4444' for x in relative_perf_1m_df['Relative Performance']]
            fig_1m = go.Figure(go.Bar(x=relative_perf_1m_df['Relative Performance'], y=relative_perf_1m_df['Sector'], orientation='h', marker_color=colors_1m))
            fig_1m.update_layout(title="vs. SPY (Last 21 Days)", xaxis_title="Outperformance (%)", yaxis_title="", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white', yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_1m, use_container_width=True)
        with chart_col3:
            st.markdown("##### YTD Relative Performance")
            colors_ytd = ['#10b981' if x >= 0 else '#ef4444' for x in relative_perf_ytd_df['Relative Performance']]
            fig_ytd = go.Figure(go.Bar(x=relative_perf_ytd_df['Relative Performance'], y=relative_perf_ytd_df['Sector'], orientation='h', marker_color=colors_ytd))
            fig_ytd.update_layout(title="vs. SPY (Year-to-Date)", xaxis_title="Outperformance (%)", yaxis_title="", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white', yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ytd, use_container_width=True)
        st.divider()
        with st.spinner('Fetching market caps for analysis...'):
            tickers_obj = yf.Tickers(valid_tickers)
            market_caps_info = {}
            for ticker_symbol in tickers_obj.tickers:
                try:
                    market_caps_info[ticker_symbol] = tickers_obj.tickers[ticker_symbol].info
                except Exception: pass
        valid_caps = {t: market_caps_info[t]['marketCap'] for t in valid_tickers if t in market_caps_info and market_caps_info.get(t) and market_caps_info[t].get('marketCap')}
        mcap_df = pd.DataFrame.from_dict(valid_caps, orient='index', columns=['Market Cap'])
        perf_df = pd.concat([(price_change / last_close * 100).rename('% Change'), mcap_df], axis=1).dropna()
        perf_df.index.name = 'Symbol'
        st.subheader("S&P 500 Daily Return Distribution")
        if perf_df.empty or perf_df['% Change'].isnull().all():
            st.warning("Could not generate the return distribution chart. Data is incomplete.")
        else:
            w_cap = perf_df["Market Cap"].values / perf_df["Market Cap"].sum()
            w_eq  = np.ones(len(perf_df)) / len(perf_df)
            avg_e, ag_e, ad_e = weighted_stats(perf_df["% Change"].values, w_eq)
            avg_w, _, _ = weighted_stats(perf_df["% Change"].values, w_cap)
            x_lo = perf_df['% Change'].quantile(0.01)
            x_hi = perf_df['% Change'].quantile(0.99)
            grid = np.linspace(x_lo, x_hi, 500)
            den_eq  = build_distribution(perf_df["% Change"].values, w_eq,  grid, bw=0.6)
            den_cap = build_distribution(perf_df["% Change"].values, w_cap, grid, bw=1.4)
            den_eq  /= max(den_eq.max(), 1e-12)
            den_cap /= max(den_cap.max(), 1e-12)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=grid, y=den_cap, mode='lines', line=dict(color='white', width=1, dash='dash'), name='Cap-Weighted'))
            zero_idx = np.searchsorted(grid, 0.0)
            fig_dist.add_trace(go.Scatter(x=grid[:zero_idx], y=den_eq[:zero_idx], fill='tozeroy', mode='none', fillcolor='rgba(239, 68, 68, 0.4)', name='Negative Returns'))
            fig_dist.add_trace(go.Scatter(x=grid[zero_idx:], y=den_eq[zero_idx:], fill='tozeroy', mode='none', fillcolor='rgba(16, 185, 129, 0.4)', name='Positive Returns'))
            fig_dist.add_trace(go.Scatter(x=grid, y=den_eq, mode='lines', line=dict(color='#f1c40f', width=2), name='Equal-Weighted'))
            fig_dist.add_vline(x=avg_e, line_width=1, line_dash="dash", line_color="#f1c40f", annotation_text=f"Average: {avg_e:+.2f}%", annotation_position="top left")
            if not pd.isna(ag_e):
                fig_dist.add_vline(x=ag_e, line_width=1, line_dash="dash", line_color="green", annotation_text=f"Avg Gain: {ag_e:+.2f}%", annotation_position="bottom right")
            if not pd.isna(ad_e):
                fig_dist.add_vline(x=ad_e, line_width=1, line_dash="dash", line_color="red", annotation_text=f"Avg Decline: {ad_e:+.2f}%", annotation_position="bottom left")
            fig_dist.update_layout(title="Equal-Weighted vs. Cap-Weighted Return Distribution", xaxis_title="Individual Stock Daily Return (%)", yaxis_title="Density (Scaled)", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white', legend=dict(x=0.01, y=0.99), height=600, yaxis_range=[0, 1.2])
            st.plotly_chart(fig_dist, use_container_width=True)
        st.divider()
        st.subheader("Daily Movers & Shakers")
        def format_movers_table(df, title, daily_data_df, current_prices_series):
            html = f"<div><b>{title}</b><table style='width:100%; color: white; border-collapse: collapse;'>"
            html += "<tr><th style='text-align: left; padding: 4px;'>Symbol</th><th style='text-align: right; padding: 4px;'>% Change</th></tr>"
            for symbol, row in df.iterrows():
                change_str = row['% Change']
                color = 'green' if change_str.startswith('+') else 'red'
                stock_daily = daily_data_df.xs(symbol, level=0, axis=1)
                low_52wk = stock_daily['Low'].rolling(window=252, min_periods=1).min().iloc[-1]
                high_52wk = stock_daily['High'].rolling(window=252, min_periods=1).max().iloc[-1]
                current = current_prices_series.get(symbol, 0)
                position_pct = 0
                if high_52wk > low_52wk:
                    position_pct = max(0, min(100, ((current - low_52wk) / (high_52wk - low_52wk)) * 100))
                html += f"<tr style='border-top: 1px solid #4A5568;'><td style='padding-top: 8px;'>{symbol}</td><td style='text-align: right; color: {color}; font-weight: bold; padding-top: 8px;'>{change_str}</td></tr>"
                html += f"""
                <tr>
                    <td colspan='2' style='padding-bottom: 8px;'>
                        <div style='display: flex; align-items: center; font-size: 0.75rem; color: #A0AEC0; width: 100%;'>
                            <span style='margin-right: 5px;'>{low_52wk:.2f}</span>
                            <div style='position: relative; flex-grow: 1; height: 4px; background-color: #2D3748; border-radius: 2px;'>
                                <div style='position: absolute; top: -3px; left: {position_pct}%; width: 10px; height: 10px; background-color: #636EFA; border-radius: 50%;' title='Current: {current:.2f}'></div>
                            </div>
                            <span style='margin-left: 5px;'>{high_52wk:.2f}</span>
                        </div>
                    </td>
                </tr>
                """
            html += "</table></div>"
            return html
        col1, col2 = st.columns(2)
        if 'perf_df' in locals() and not perf_df.empty:
            top_performers = perf_df.sort_values('% Change', ascending=False).head(25)[['% Change']]
            top_performers['% Change'] = top_performers['% Change'].map('{:+.2f}%'.format)
            bottom_performers = perf_df.sort_values('% Change').head(25)[['% Change']]
            bottom_performers['% Change'] = bottom_performers['% Change'].map('{:+.2f}%'.format)
            with col1:
                st.markdown(format_movers_table(top_performers, "Top 25 Performers", daily_data, current_price), unsafe_allow_html=True)
            with col2:
                st.markdown(format_movers_table(bottom_performers, "Bottom 25 Performers", daily_data, current_price), unsafe_allow_html=True)
        else:
            st.warning("Top/Bottom performer data is unavailable.")

if __name__ == "__main__":
    run_dashboard()

