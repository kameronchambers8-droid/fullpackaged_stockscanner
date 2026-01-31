"""
PREMIUM Stock Analysis Platform
================================
Features:
- Live trade tracking with real-time updates
- 1-year news archive (Urgent vs Regular)
- Smart entry/exit recommendations based on holding duration
- Portfolio performance tracking
- Live price monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time

# Page config
st.set_page_config(
    page_title="Stock Analysis Pro - PREMIUM",
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
    .live-price {
        font-size: 3rem;
        font-weight: 700;
        color: #2ecc71;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .urgent-news {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #c0392b;
        animation: urgentPulse 3s infinite;
    }
    @keyframes urgentPulse {
        0%, 100% { box-shadow: 0 0 10px rgba(231, 76, 60, 0.5); }
        50% { box-shadow: 0 0 20px rgba(231, 76, 60, 0.8); }
    }
    .regular-news {
        background: #1a1d26;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #3498db;
    }
    .trade-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #3498db;
    }
    .profit { color: #2ecc71; font-weight: bold; }
    .loss { color: #e74c3c; font-weight: bold; }
    .entry-signal {
        background: #2ecc71;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .exit-signal {
        background: #e74c3c;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .hold-signal {
        background: #f39c12;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for live trades
if 'live_trades' not in st.session_state:
    st.session_state.live_trades = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 0

# API Keys
ALPHA_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "demo")
FMP_KEY = st.secrets.get("FMP_KEY", "demo")

# ============================================================================
# LIVE DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)  # Update every minute
def get_live_quote(symbol: str) -> Optional[Dict]:
    """Get real-time quote"""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": ALPHA_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        quote = data.get("Global Quote", {})
        if quote:
            return {
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_pct': float(quote.get('10. change percent', '0').replace('%', '')),
                'volume': int(quote.get('06. volume', 0)),
                'timestamp': datetime.now()
            }
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch historical daily data"""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": ALPHA_KEY,
            "outputsize": "compact"
        }
        
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if "Error Message" in data:
            st.error(f"❌ Invalid ticker: {symbol}")
            return None
        
        if "Note" in data:
            st.error("🚨 API rate limit reached. Wait until tomorrow.")
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

@st.cache_data(ttl=1800)
def fetch_news_archive(symbol: str, lookback_months: int = 12) -> List[Dict]:
    """Fetch news from past year with sentiment"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months*30)
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": ALPHA_KEY,
            "time_from": start_date.strftime("%Y%m%dT%H%M"),
            "time_to": end_date.strftime("%Y%m%dT%H%M"),
            "limit": 50,
            "sort": "LATEST"
        }
        
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if "Note" in data:
            return []
        
        return data.get('feed', [])
    except:
        return []

def categorize_news(articles: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize news as URGENT or REGULAR"""
    urgent = []
    regular = []
    
    # Keywords that indicate urgent/price-moving news
    urgent_keywords = [
        'earnings', 'acquisition', 'merger', 'bankruptcy', 'sec', 'investigation',
        'lawsuit', 'settlement', 'fda', 'approval', 'recall', 'ceo', 'layoffs',
        'guidance', 'beat', 'miss', 'downgrade', 'upgrade', 'analyst', 'target',
        'dividend', 'split', 'buyback', 'offering', 'ipo', 'delisting'
    ]
    
    for article in articles:
        title = article.get('title', '').lower()
        summary = article.get('summary', '').lower()
        sentiment = float(article.get('overall_sentiment_score', 0))
        
        # Check if urgent
        is_urgent = False
        
        # Strong sentiment = urgent
        if abs(sentiment) > 0.25:
            is_urgent = True
        
        # Contains urgent keywords
        if any(keyword in title or keyword in summary for keyword in urgent_keywords):
            is_urgent = True
        
        if is_urgent:
            urgent.append(article)
        else:
            regular.append(article)
    
    return {'urgent': urgent, 'regular': regular}

# ============================================================================
# SMART ENTRY/EXIT CALCULATOR
# ============================================================================

def calculate_entry_exit_points(df: pd.DataFrame, duration_days: int, current_price: float) -> Dict:
    """Calculate optimal entry/exit based on holding duration"""
    
    # Calculate technical levels
    rsi = calc_rsi(df['close'])
    macd, signal_line, hist = calc_macd(df['close'])
    bb_upper, bb_mid, bb_lower = calc_bb(df['close'])
    ma = calc_ma(df['close'])
    atr = calc_atr(df)
    
    current_rsi = rsi.iloc[-1]
    current_atr = atr.iloc[-1]
    
    recommendation = {
        'action': 'WAIT',
        'entry_price': None,
        'stop_loss': None,
        'take_profit_1': None,
        'take_profit_2': None,
        'take_profit_3': None,
        'reasoning': [],
        'risk_reward': None
    }
    
    # DAY TRADING (1-3 days)
    if duration_days <= 3:
        # Look for intraday momentum
        if current_rsi < 40 and current_price <= bb_lower.iloc[-1]:
            recommendation['action'] = 'BUY NOW'
            recommendation['entry_price'] = current_price
            recommendation['stop_loss'] = current_price - (current_atr * 1.5)
            recommendation['take_profit_1'] = current_price + (current_atr * 2)
            recommendation['take_profit_2'] = current_price + (current_atr * 3)
            recommendation['take_profit_3'] = current_price + (current_atr * 4)
            recommendation['reasoning'] = [
                f"RSI oversold at {current_rsi:.1f}",
                f"Price at lower Bollinger Band",
                "Short-term bounce expected"
            ]
        elif current_rsi > 60 and current_price >= bb_upper.iloc[-1]:
            recommendation['action'] = 'WAIT FOR PULLBACK'
            recommendation['entry_price'] = current_price - (current_atr * 1)
            recommendation['reasoning'] = [
                f"RSI overbought at {current_rsi:.1f}",
                "Price at upper band - wait for dip"
            ]
        else:
            recommendation['action'] = 'WAIT'
            recommendation['reasoning'] = ["No clear day trading setup"]
    
    # SWING TRADING (4-30 days)
    elif duration_days <= 30:
        sma50 = ma['sma50'].iloc[-1]
        sma200 = ma['sma200'].iloc[-1]
        
        if current_price > sma50 > sma200 and current_rsi < 50:
            recommendation['action'] = 'BUY NOW'
            recommendation['entry_price'] = current_price
            recommendation['stop_loss'] = sma50 * 0.98  # 2% below 50 MA
            recommendation['take_profit_1'] = current_price + (current_atr * 3)
            recommendation['take_profit_2'] = current_price + (current_atr * 5)
            recommendation['take_profit_3'] = current_price + (current_atr * 8)
            recommendation['reasoning'] = [
                "Uptrend intact (price > 50MA > 200MA)",
                f"RSI pullback to {current_rsi:.1f}",
                "Good risk/reward for swing trade"
            ]
        elif current_price < sma50:
            recommendation['action'] = 'WAIT FOR SUPPORT'
            recommendation['entry_price'] = sma50
            recommendation['reasoning'] = [
                "Wait for price to return to 50MA support",
                f"Target entry: ${sma50:.2f}"
            ]
        else:
            recommendation['action'] = 'WAIT'
            recommendation['reasoning'] = ["Monitor for swing setup"]
    
    # POSITION TRADING (31+ days)
    else:
        sma200 = ma['sma200'].iloc[-1]
        
        if pd.notna(sma200):
            if current_price > sma200 * 1.05:  # 5% above 200MA
                recommendation['action'] = 'BUY ON DIP'
                recommendation['entry_price'] = sma200 * 1.02  # Near 200MA
                recommendation['stop_loss'] = sma200 * 0.95
                recommendation['take_profit_1'] = current_price * 1.10
                recommendation['take_profit_2'] = current_price * 1.25
                recommendation['take_profit_3'] = current_price * 1.50
                recommendation['reasoning'] = [
                    "Long-term uptrend confirmed",
                    f"Wait for dip to ${sma200 * 1.02:.2f}",
                    "Position trade with 10-50% upside"
                ]
            elif current_price <= sma200 * 1.02 and current_rsi < 40:
                recommendation['action'] = 'BUY NOW'
                recommendation['entry_price'] = current_price
                recommendation['stop_loss'] = current_price * 0.90
                recommendation['take_profit_1'] = sma200 * 1.10
                recommendation['take_profit_2'] = sma200 * 1.25
                recommendation['take_profit_3'] = sma200 * 1.50
                recommendation['reasoning'] = [
                    "Near 200MA support",
                    f"RSI oversold at {current_rsi:.1f}",
                    "Long-term value entry"
                ]
            else:
                recommendation['action'] = 'WAIT'
                recommendation['reasoning'] = ["Wait for better position entry"]
    
    # Calculate risk/reward
    if recommendation['entry_price'] and recommendation['stop_loss'] and recommendation['take_profit_1']:
        risk = recommendation['entry_price'] - recommendation['stop_loss']
        reward = recommendation['take_profit_1'] - recommendation['entry_price']
        if risk > 0:
            recommendation['risk_reward'] = reward / risk
    
    return recommendation

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(prices):
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
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

def calc_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ============================================================================
# TRADE MANAGEMENT
# ============================================================================

def add_live_trade(ticker, entry_price, shares, stop_loss, take_profit, duration):
    """Add a new live trade"""
    trade = {
        'id': len(st.session_state.live_trades),
        'ticker': ticker,
        'entry_price': entry_price,
        'shares': shares,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'duration_days': duration,
        'entry_date': datetime.now(),
        'status': 'OPEN',
        'current_price': entry_price,
        'current_pl': 0,
        'current_pl_pct': 0
    }
    st.session_state.live_trades.append(trade)
    return trade

def update_live_trades():
    """Update all open trades with current prices"""
    for trade in st.session_state.live_trades:
        if trade['status'] == 'OPEN':
            quote = get_live_quote(trade['ticker'])
            if quote:
                trade['current_price'] = quote['price']
                trade['current_pl'] = (quote['price'] - trade['entry_price']) * trade['shares']
                trade['current_pl_pct'] = ((quote['price'] - trade['entry_price']) / trade['entry_price']) * 100
                
                # Check stop loss / take profit
                if quote['price'] <= trade['stop_loss']:
                    trade['status'] = 'STOPPED OUT'
                    trade['exit_price'] = quote['price']
                    trade['exit_date'] = datetime.now()
                elif quote['price'] >= trade['take_profit']:
                    trade['status'] = 'TAKE PROFIT HIT'
                    trade['exit_price'] = quote['price']
                    trade['exit_date'] = datetime.now()
    
    st.session_state.last_update = datetime.now()

def close_trade(trade_id, exit_price):
    """Manually close a trade"""
    for trade in st.session_state.live_trades:
        if trade['id'] == trade_id and trade['status'] == 'OPEN':
            trade['status'] = 'CLOSED'
            trade['exit_price'] = exit_price
            trade['exit_date'] = datetime.now()
            trade['current_pl'] = (exit_price - trade['entry_price']) * trade['shares']
            trade['current_pl_pct'] = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown('<h1 class="main-header">📈 PREMIUM Stock Analysis Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#7f8c8d;">Live Trade Tracking | Smart Entry/Exit | News Intelligence</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Stock Analysis")
    symbol = st.text_input("Enter Ticker", "AAPL").upper()
    
    st.markdown("---")
    st.header("⏱️ Trading Timeframe")
    duration = st.selectbox(
        "How long will you hold?",
        ["Day Trade (1-3 days)", "Swing Trade (4-30 days)", "Position Trade (31+ days)"]
    )
    
    duration_days = 1 if "Day" in duration else 14 if "Swing" in duration else 60
    
    if st.button("🔄 Analyze", type="primary"):
        st.session_state.current_symbol = symbol
        st.session_state.current_duration = duration_days
        st.rerun()
    
    st.markdown("---")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 60sec)", value=False)
    
    if auto_refresh:
        st.info("Live prices updating...")
        time.sleep(60)
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Smart Entry/Exit",
    "📊 Live Trades",
    "📰 News Intelligence",
    "📈 Technical Analysis",
    "💰 Calculators"
])

# TAB 1: Smart Entry/Exit
with tab1:
    if 'current_symbol' in st.session_state:
        symbol = st.session_state.current_symbol
        duration_days = st.session_state.current_duration
        
        st.subheader(f"🎯 Smart Analysis: {symbol}")
        st.info(f"Trading Duration: {duration_days} days")
        
        # Get live quote
        quote = get_live_quote(symbol)
        
        if quote:
            # Display live price
            price_class = "profit" if quote['change'] > 0 else "loss"
            st.markdown(f"""
            <div class="live-price">
                <div style="font-size: 1.5rem; color: #7f8c8d;">LIVE PRICE</div>
                <div class="{price_class}">${quote['price']:.2f}</div>
                <div style="font-size: 1.2rem;">{quote['change']:+.2f} ({quote['change_pct']:+.2f}%)</div>
                <div style="font-size: 0.9rem; color: #7f8c8d;">Last Update: {quote['timestamp'].strftime('%H:%M:%S')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get historical data
            df = fetch_stock_data(symbol)
            
            if df is not None:
                # Calculate entry/exit
                recommendation = calculate_entry_exit_points(df, duration_days, quote['price'])
                
                # Display recommendation
                signal_class = "entry-signal" if "BUY" in recommendation['action'] else "exit-signal" if "SELL" in recommendation['action'] else "hold-signal"
                
                st.markdown(f"""
                <div class="{signal_class}">
                    {recommendation['action']}
                </div>
                """, unsafe_allow_html=True)
                
                # Entry details
                if recommendation['entry_price']:
                    st.markdown("### 📍 Recommended Entry")
                    entry_col1, entry_col2, entry_col3 = st.columns(3)
                    
                    with entry_col1:
                        st.metric("Entry Price", f"${recommendation['entry_price']:.2f}")
                    
                    with entry_col2:
                        if recommendation['stop_loss']:
                            st.metric("Stop Loss", f"${recommendation['stop_loss']:.2f}",
                                     delta=f"-{((recommendation['entry_price'] - recommendation['stop_loss']) / recommendation['entry_price'] * 100):.1f}%")
                    
                    with entry_col3:
                        if recommendation['risk_reward']:
                            st.metric("Risk/Reward", f"1:{recommendation['risk_reward']:.2f}")
                    
                    # Take profit levels
                    st.markdown("### 🎯 Take Profit Targets")
                    tp_col1, tp_col2, tp_col3 = st.columns(3)
                    
                    with tp_col1:
                        if recommendation['take_profit_1']:
                            gain = ((recommendation['take_profit_1'] - recommendation['entry_price']) / recommendation['entry_price'] * 100)
                            st.metric("Target 1", f"${recommendation['take_profit_1']:.2f}", delta=f"+{gain:.1f}%")
                    
                    with tp_col2:
                        if recommendation['take_profit_2']:
                            gain = ((recommendation['take_profit_2'] - recommendation['entry_price']) / recommendation['entry_price'] * 100)
                            st.metric("Target 2", f"${recommendation['take_profit_2']:.2f}", delta=f"+{gain:.1f}%")
                    
                    with tp_col3:
                        if recommendation['take_profit_3']:
                            gain = ((recommendation['take_profit_3'] - recommendation['entry_price']) / recommendation['entry_price'] * 100)
                            st.metric("Target 3", f"${recommendation['take_profit_3']:.2f}", delta=f"+{gain:.1f}%")
                
                # Reasoning
                st.markdown("### 💡 Analysis Reasoning")
                for reason in recommendation['reasoning']:
                    st.info(f"✓ {reason}")
                
                # Add to live trades
                st.markdown("---")
                st.markdown("### 📝 Add to Live Trades")
                
                with st.form("add_trade_form"):
                    trade_col1, trade_col2 = st.columns(2)
                    
                    with trade_col1:
                        entry_price_input = st.number_input("Entry Price", value=float(recommendation.get('entry_price', quote['price'])))
                        shares_input = st.number_input("Shares", min_value=1, value=10)
                    
                    with trade_col2:
                        stop_loss_input = st.number_input("Stop Loss", value=float(recommendation.get('stop_loss', quote['price'] * 0.95)))
                        take_profit_input = st.number_input("Take Profit", value=float(recommendation.get('take_profit_1', quote['price'] * 1.05)))
                    
                    submitted = st.form_submit_button("➕ Add Live Trade", type="primary")
                    
                    if submitted:
                        add_live_trade(symbol, entry_price_input, shares_input, stop_loss_input, take_profit_input, duration_days)
                        st.success(f"✅ Added {symbol} to live trades!")
                        st.rerun()
    
    else:
        st.info("👈 Enter a ticker and click 'Analyze' to get started")

# TAB 2: Live Trades
with tab2:
    st.subheader("📊 Live Trade Dashboard")
    
    # Update all trades
    update_live_trades()
    
    if st.session_state.live_trades:
        # Summary stats
        open_trades = [t for t in st.session_state.live_trades if t['status'] == 'OPEN']
        closed_trades = [t for t in st.session_state.live_trades if t['status'] != 'OPEN']
        
        total_pl = sum(t['current_pl'] for t in open_trades)
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Open Trades", len(open_trades))
        
        with stat_col2:
            st.metric("Closed Trades", len(closed_trades))
        
        with stat_col3:
            pl_class = "profit" if total_pl >= 0 else "loss"
            st.metric("Total P/L", f"${total_pl:,.2f}")
        
        with stat_col4:
            if st.session_state.last_update:
                st.metric("Last Update", st.session_state.last_update.strftime('%H:%M:%S'))
        
        st.markdown("---")
        
        # Open trades
        if open_trades:
            st.markdown("### 🟢 Open Positions")
            
            for trade in open_trades:
                pl_class = "profit" if trade['current_pl'] >= 0 else "loss"
                
                st.markdown(f"""
                <div class="trade-card">
                    <h3>{trade['ticker']} - {trade['shares']} shares</h3>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0;">
                        <div>
                            <small>Entry Price</small><br>
                            <strong>${trade['entry_price']:.2f}</strong>
                        </div>
                        <div>
                            <small>Current Price</small><br>
                            <strong>${trade['current_price']:.2f}</strong>
                        </div>
                        <div>
                            <small>P/L</small><br>
                            <strong class="{pl_class}">${trade['current_pl']:.2f} ({trade['current_pl_pct']:+.2f}%)</strong>
                        </div>
                        <div>
                            <small>Duration</small><br>
                            <strong>{trade['duration_days']} days</strong>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                        <div>
                            <small>Stop Loss: ${trade['stop_loss']:.2f}</small> | 
                            <small>Take Profit: ${trade['take_profit']:.2f}</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"❌ Close {trade['ticker']}", key=f"close_{trade['id']}"):
                    close_trade(trade['id'], trade['current_price'])
                    st.success(f"Closed {trade['ticker']} at ${trade['current_price']:.2f}")
                    st.rerun()
        
        # Closed trades
        if closed_trades:
            st.markdown("---")
            st.markdown("### ⚫ Closed Positions")
            
            for trade in closed_trades:
                pl_class = "profit" if trade['current_pl'] >= 0 else "loss"
                
                st.markdown(f"""
                <div class="trade-card" style="opacity: 0.7;">
                    <h4>{trade['ticker']} - {trade['status']}</h4>
                    <p>Entry: ${trade['entry_price']:.2f} → Exit: ${trade.get('exit_price', 0):.2f}</p>
                    <p class="{pl_class}">P/L: ${trade['current_pl']:.2f} ({trade['current_pl_pct']:+.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("No live trades yet. Add a trade from the Smart Entry/Exit tab!")

# TAB 3: News Intelligence
with tab3:
    if 'current_symbol' in st.session_state:
        symbol = st.session_state.current_symbol
        
        st.subheader(f"📰 News Intelligence: {symbol}")
        
        with st.spinner("Fetching 1-year news archive..."):
            articles = fetch_news_archive(symbol, lookback_months=12)
            categorized = categorize_news(articles)
        
        # Urgent News
        st.markdown("### 🚨 URGENT NEWS (Price-Moving)")
        
        urgent_news = categorized['urgent']
        
        if urgent_news:
            for article in urgent_news[:10]:
                sentiment = float(article.get('overall_sentiment_score', 0))
                sentiment_emoji = '🟢' if sentiment > 0.15 else '🔴' if sentiment < -0.15 else '⚪'
                
                st.markdown(f"""
                <div class="urgent-news">
                    <h3>🚨 {article.get('title', 'No title')}</h3>
                    <p><strong>{article.get('source', 'Unknown')} | {article.get('time_published', '')[:10]}</strong></p>
                    <p><strong>Sentiment: {sentiment:.2f} {sentiment_emoji}</strong></p>
                    <p>{article.get('summary', 'No summary')[:200]}...</p>
                    <a href="{article.get('url', '#')}" target="_blank" style="color: white;">📖 Read Full Article →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No urgent news in the past year")
        
        st.markdown("---")
        
        # Regular News
        st.markdown("### 📰 REGULAR NEWS")
        
        regular_news = categorized['regular']
        
        if regular_news:
            for article in regular_news[:15]:
                sentiment = float(article.get('overall_sentiment_score', 0))
                sentiment_emoji = '🟢' if sentiment > 0.15 else '🔴' if sentiment < -0.15 else '⚪'
                
                st.markdown(f"""
                <div class="regular-news">
                    <h4>{article.get('title', 'No title')}</h4>
                    <p><small>{article.get('source', 'Unknown')} | {article.get('time_published', '')[:10]}</small></p>
                    <p>Sentiment: {sentiment:.2f} {sentiment_emoji}</p>
                    <a href="{article.get('url', '#')}" target="_blank">Read more →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No regular news available")
    
    else:
        st.info("👈 Analyze a stock first")

# TAB 4: Technical Analysis (Same as before)
with tab4:
    if 'current_symbol' in st.session_state:
        symbol = st.session_state.current_symbol
        df = fetch_stock_data(symbol)
        
        if df is not None:
            st.subheader(f"📈 Technical Analysis: {symbol}")
            
            # Price chart
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            )])
            
            fig.update_layout(title=f"{symbol} Price Chart", template='plotly_dark', height=600)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Analyze a stock first")

# TAB 5: Calculators (P/L, Position Sizing)
with tab5:
    st.subheader("💰 Trading Calculators")
    
    st.markdown("### P/L Calculator")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        entry = st.number_input("Entry Price", value=100.0)
        shares = st.number_input("Shares", value=10)
    
    with calc_col2:
        exit = st.number_input("Exit Price", value=110.0)
    
    pl = (exit - entry) * shares
    pl_pct = ((exit - entry) / entry) * 100
    
    pl_class = "profit" if pl >= 0 else "loss"
    
    st.markdown(f"""
    <div class="trade-card">
        <h3>Result</h3>
        <p class="{pl_class}">P/L: ${pl:.2f} ({pl_pct:+.2f}%)</p>
    </div>
    """, unsafe_allow_html=True)
