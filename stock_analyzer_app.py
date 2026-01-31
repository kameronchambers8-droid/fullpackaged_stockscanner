"""
Complete Stock Analysis Platform
==================================
Professional stock analysis with technical indicators, fundamentals,
news sentiment, and trade recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Stock Analysis Pro",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .metric-card {
        background: #1a1d26;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .signal-buy { background: #2ecc71; color: white; padding: 0.4rem 1rem; border-radius: 20px; }
    .signal-sell { background: #e74c3c; color: white; padding: 0.4rem 1rem; border-radius: 20px; }
    .signal-hold { background: #f39c12; color: white; padding: 0.4rem 1rem; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# API Keys
ALPHA_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "demo")
FMP_KEY = st.secrets.get("FMP_KEY", "demo")

# API Functions
def fetch_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch daily stock data"""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": ALPHA_KEY,
            "outputsize": "full"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "Error Message" in data:
            st.error(f"Invalid ticker: {symbol}")
            return None
        
        if "Note" in data:
            st.warning("API limit reached. Try again tomorrow or upgrade.")
            return None
        
        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            return None
        
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def fetch_company_overview(symbol: str) -> Optional[Dict]:
    """Fetch company fundamentals"""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": ALPHA_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except:
        return None

def fetch_news(symbol: str) -> List[Dict]:
    """Fetch news with sentiment"""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": ALPHA_KEY,
            "limit": 10
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data.get('feed', [])
    except:
        return []

# Technical Indicators
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(prices):
    exp1 = prices.ewm(span=12).mean()
    exp2 = prices.ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd, signal, macd - signal

def calc_bb(prices, period=20):
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return sma + (2*std), sma, sma - (2*std)

def calc_ma(prices):
    return {
        'sma20': prices.rolling(20).mean(),
        'sma50': prices.rolling(50).mean(),
        'sma200': prices.rolling(200).mean()
    }

def get_signals(df):
    """Generate buy/sell signals"""
    signals = {}
    price = df['close'].iloc[-1]
    
    # RSI
    rsi = calc_rsi(df['close']).iloc[-1]
    if rsi < 30:
        signals['RSI'] = ('BUY', f'{rsi:.1f}', 'Oversold')
    elif rsi > 70:
        signals['RSI'] = ('SELL', f'{rsi:.1f}', 'Overbought')
    else:
        signals['RSI'] = ('HOLD', f'{rsi:.1f}', 'Neutral')
    
    # MACD
    macd, signal, hist = calc_macd(df['close'])
    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
        signals['MACD'] = ('BUY', f'{macd.iloc[-1]:.2f}', 'Bullish cross')
    elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
        signals['MACD'] = ('SELL', f'{macd.iloc[-1]:.2f}', 'Bearish cross')
    else:
        signals['MACD'] = ('HOLD', f'{macd.iloc[-1]:.2f}', 'No cross')
    
    # BB
    upper, mid, lower = calc_bb(df['close'])
    if price <= lower.iloc[-1]:
        signals['BB'] = ('BUY', f'${price:.2f}', 'At lower band')
    elif price >= upper.iloc[-1]:
        signals['BB'] = ('SELL', f'${price:.2f}', 'At upper band')
    else:
        signals['BB'] = ('HOLD', f'${price:.2f}', 'In bands')
    
    # MA
    ma = calc_ma(df['close'])
    sma50 = ma['sma50'].iloc[-1]
    sma200 = ma['sma200'].iloc[-1]
    if pd.notna(sma50) and pd.notna(sma200):
        if price > sma50 > sma200:
            signals['MA'] = ('BUY', f'${sma50:.2f}', 'Golden cross')
        elif price < sma50 < sma200:
            signals['MA'] = ('SELL', f'${sma50:.2f}', 'Death cross')
        else:
            signals['MA'] = ('HOLD', f'${sma50:.2f}', 'Mixed')
    
    return signals

def get_recommendation(signals):
    """Overall recommendation"""
    buy_count = sum(1 for s in signals.values() if s[0] == 'BUY')
    sell_count = sum(1 for s in signals.values() if s[0] == 'SELL')
    total = len(signals)
    
    if buy_count / total >= 0.6:
        return "STRONG BUY"
    elif buy_count / total >= 0.4:
        return "BUY"
    elif sell_count / total >= 0.6:
        return "STRONG SELL"
    elif sell_count / total >= 0.4:
        return "SELL"
    return "HOLD"

# Main App
st.markdown('<h1 class="main-header">📈 Stock Analysis Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#7f8c8d;">Professional stock analysis for smarter trading</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Stock Search")
    symbol = st.text_input("Enter Ticker", "AAPL").upper()
    st.markdown("---")
    st.markdown("**Quick Links**")
    st.markdown("- [Webull](https://webull.com)")
    st.markdown("- [Yahoo Finance](https://finance.yahoo.com)")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Technical",
    "📰 News & Fundamentals", 
    "💰 P/L Calculator",
    "🎯 Recommendation",
    "📔 Trade Journal"
])

# TAB 1: Technical Analysis
with tab1:
    if symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            df = fetch_stock_data(symbol)
            
            if df is not None and not df.empty:
                price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-2]
                change = price - prev_price
                change_pct = (change / prev_price) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${price:.2f}", f"{change_pct:+.2f}%")
                col2.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}")
                col3.metric("High", f"${df['high'].iloc[-1]:.2f}")
                col4.metric("Low", f"${df['low'].iloc[-1]:.2f}")
                
                st.subheader("📈 Technical Indicators")
                
                # Calculate indicators
                rsi = calc_rsi(df['close'])
                macd, signal_line, hist = calc_macd(df['close'])
                bb_upper, bb_mid, bb_lower = calc_bb(df['close'])
                ma = calc_ma(df['close'])
                
                # Create chart
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=('Price & Indicators', 'RSI', 'MACD')
                )
                
                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ),
                    row=1, col=1
                )
                
                # Bollinger Bands
                fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name='BB Upper', line=dict(dash='dash', color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=bb_mid, name='BB Mid', line=dict(dash='dash', color='gray')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name='BB Lower', line=dict(dash='dash', color='green')), row=1, col=1)
                
                # Moving Averages
                fig.add_trace(go.Scatter(x=df.index, y=ma['sma50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=ma['sma200'], name='SMA 200', line=dict(color='red')), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal', line=dict(color='orange')), row=3, col=1)
                fig.add_trace(go.Bar(x=df.index, y=hist, name='Histogram', marker_color='gray'), row=3, col=1)
                
                fig.update_layout(height=900, template='plotly_dark', showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Current values
                st.subheader("📊 Current Indicators")
                ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                
                with ind_col1:
                    rsi_val = rsi.iloc[-1]
                    st.metric("RSI (14)", f"{rsi_val:.1f}", 
                             "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral")
                
                with ind_col2:
                    st.metric("MACD", f"{macd.iloc[-1]:.3f}", 
                             f"Signal: {signal_line.iloc[-1]:.3f}")
                
                with ind_col3:
                    st.metric("SMA 50", f"${ma['sma50'].iloc[-1]:.2f}" if pd.notna(ma['sma50'].iloc[-1]) else "N/A")
                
                with ind_col4:
                    st.metric("SMA 200", f"${ma['sma200'].iloc[-1]:.2f}" if pd.notna(ma['sma200'].iloc[-1]) else "N/A")

# TAB 2: News & Fundamentals
with tab2:
    if symbol:
        st.subheader("📰 Latest News")
        news = fetch_news(symbol)
        
        if news:
            for article in news[:5]:
                sentiment_score = float(article.get('overall_sentiment_score', 0))
                sentiment_class = "positive-sentiment" if sentiment_score > 0.15 else "negative-sentiment" if sentiment_score < -0.15 else "neutral-sentiment"
                
                st.markdown(f"""
                <div class="news-card {sentiment_class}">
                    <h4>{article.get('title', 'No title')}</h4>
                    <p><small>{article.get('source', 'Unknown')} | {article.get('time_published', '')[:10]}</small></p>
                    <p>Sentiment: {sentiment_score:.2f} {'🟢' if sentiment_score > 0.15 else '🔴' if sentiment_score < -0.15 else '⚪'}</p>
                    <a href="{article.get('url', '#')}" target="_blank">Read more →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent news available")
        
        st.markdown("---")
        st.subheader("📊 Company Fundamentals")
        overview = fetch_company_overview(symbol)
        
        if overview and 'Symbol' in overview:
            fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
            
            with fund_col1:
                st.metric("Market Cap", overview.get('MarketCapitalization', 'N/A'))
                st.metric("P/E Ratio", overview.get('PERatio', 'N/A'))
            
            with fund_col2:
                st.metric("EPS", overview.get('EPS', 'N/A'))
                st.metric("Dividend Yield", overview.get('DividendYield', 'N/A'))
            
            with fund_col3:
                st.metric("52W High", overview.get('52WeekHigh', 'N/A'))
                st.metric("52W Low", overview.get('52WeekLow', 'N/A'))
            
            with fund_col4:
                st.metric("Beta", overview.get('Beta', 'N/A'))
                st.metric("Sector", overview.get('Sector', 'N/A'))

# TAB 3: P/L Calculator
with tab3:
    st.subheader("💰 Profit/Loss Calculator")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        st.markdown("### Entry Details")
        entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=100.0, step=0.01)
        shares = st.number_input("Shares", min_value=1, value=100, step=1)
        entry_fees = st.number_input("Entry Fees ($)", min_value=0.0, value=0.0, step=0.01)
    
    with calc_col2:
        st.markdown("### Exit Details")
        exit_price = st.number_input("Exit Price ($)", min_value=0.01, value=110.0, step=0.01)
        exit_fees = st.number_input("Exit Fees ($)", min_value=0.0, value=0.0, step=0.01)
    
    entry_cost = (entry_price * shares) + entry_fees
    exit_value = (exit_price * shares) - exit_fees
    profit_loss = exit_value - entry_cost
    profit_loss_pct = (profit_loss / entry_cost) * 100
    
    st.markdown("---")
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric("Entry Cost", f"${entry_cost:,.2f}")
    
    with result_col2:
        st.metric("Exit Value", f"${exit_value:,.2f}")
    
    with result_col3:
        st.metric("Profit/Loss", f"${profit_loss:,.2f}", f"{profit_loss_pct:+.2f}%")
    
    st.markdown("---")
    st.subheader("📏 Position Sizing")
    
    pos_col1, pos_col2 = st.columns(2)
    
    with pos_col1:
        account_size = st.number_input("Account Size ($)", min_value=100.0, value=10000.0, step=100.0)
        risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
    
    with pos_col2:
        stop_loss_pct = st.slider("Stop Loss (%)", 1.0, 20.0, 5.0, 0.5)
    
    risk_amount = account_size * (risk_percent / 100)
    position_size = risk_amount / (stop_loss_pct / 100)
    shares_to_buy = int(position_size / entry_price)
    
    st.success(f"""
    **Recommendation:**
    - Risk Amount: ${risk_amount:,.2f}
    - Position Size: ${position_size:,.2f}
    - Shares to Buy: {shares_to_buy}
    - Stop Loss Price: ${entry_price * (1 - stop_loss_pct/100):.2f}
    """)

# TAB 4: Trade Recommendation
with tab4:
    st.subheader("🎯 Trade Recommendation")
    
    if symbol:
        df = fetch_stock_data(symbol)
        
        if df is not None and not df.empty:
            signals = get_signals(df)
            recommendation = get_recommendation(signals)
            
            rec_colors = {
                'STRONG BUY': 'success',
                'BUY': 'success',
                'HOLD': 'warning',
                'SELL': 'error',
                'STRONG SELL': 'error'
            }
            
            if recommendation in ['STRONG BUY', 'BUY']:
                st.success(f"## {recommendation}")
            elif recommendation in ['SELL', 'STRONG SELL']:
                st.error(f"## {recommendation}")
            else:
                st.warning(f"## {recommendation}")
            
            st.markdown("---")
            st.subheader("📊 Signal Breakdown")
            
            sig_cols = st.columns(len(signals))
            
            for idx, (indicator, (signal, value, reason)) in enumerate(signals.items()):
                with sig_cols[idx]:
                    badge_class = f"signal-{signal.lower()}"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{indicator}</h3>
                        <span class="{badge_class}">{signal}</span>
                        <p style="margin-top:1rem;">{reason}</p>
                        <p><small>Value: {value}</small></p>
                    </div>
                    """, unsafe_allow_html=True)

# TAB 5: Trade Journal
with tab5:
    st.subheader("📔 Trade Journal")
    
    st.info("""
    **Coming Soon - Full Trade Journal Features:**
    - Log all trade details (entry/exit, P/L, strategy)
    - Track emotions and mindset
    - Document lessons learned
    - Performance analytics and win rate
    - Export to CSV/Excel
    
    For now, use the P/L Calculator tab to analyze individual trades!
    """)
    
    with st.expander("✏️ Log Trade (Preview)"):
        j_col1, j_col2 = st.columns(2)
        
        with j_col1:
            st.date_input("Trade Date")
            st.text_input("Ticker")
            st.selectbox("Type", ["Long", "Short"])
            st.number_input("Entry Price", 0.0)
            st.number_input("Exit Price", 0.0)
        
        with j_col2:
            st.selectbox("Strategy", ["Breakout", "Trend", "Mean Reversion", "News", "Other"])
            st.text_area("Entry Reason")
            st.text_area("Exit Reason")
            st.slider("Emotional State", 1, 10, 5)
        
        st.button("Save Trade (Coming Soon)", disabled=True)
