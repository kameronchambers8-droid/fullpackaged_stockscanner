"""
Personal Stock Analysis Platform
==================================
A comprehensive tool for stock analysis, trading decisions, and journal tracking.

Created for: Personal trading analysis and decision-making
Features: Technical analysis, fundamental analysis, P/L calculator, trade journal
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ecc71;
        --danger-color: #e74c3c;
        --warning-color: #f39c12;
        --bg-dark: #0e1117;
        --bg-secondary: #262730;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #262730 0%, #1a1d26 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-card .positive {
        color: #2ecc71;
    }
    
    .metric-card .negative {
        color: #e74c3c;
    }
    
    .metric-card .neutral {
        color: #1f77b4;
    }
    
    /* Signal badges */
    .signal-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    
    .signal-buy {
        background: #2ecc71;
        color: white;
    }
    
    .signal-sell {
        background: #e74c3c;
        color: white;
    }
    
    .signal-hold {
        background: #f39c12;
        color: white;
    }
    
    .signal-neutral {
        background: #7f8c8d;
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #262730;
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    /* Info boxes */
    .info-box {
        background: #1a1d26;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #3498db;
        margin: 1rem 0;
    }
    
    .success-box {
        border-left-color: #2ecc71;
        background: rgba(46, 204, 113, 0.1);
    }
    
    .warning-box {
        border-left-color: #f39c12;
        background: rgba(243, 156, 18, 0.1);
    }
    
    .danger-box {
        border-left-color: #e74c3c;
        background: rgba(231, 76, 60, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

# API Keys (User will add their own)
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "demo")
FMP_KEY = st.secrets.get("FMP_KEY", "demo")

# API Endpoints
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
FMP_BASE = "https://financialmodelingprep.com/api/v3"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fetch_stock_data(symbol: str, interval: str = "daily") -> Optional[pd.DataFrame]:
    """Fetch stock price data from Alpha Vantage"""
    try:
        function_map = {
            "daily": "TIME_SERIES_DAILY",
            "intraday": "TIME_SERIES_INTRADAY",
            "weekly": "TIME_SERIES_WEEKLY"
        }
        
        params = {
            "function": function_map.get(interval, "TIME_SERIES_DAILY"),
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_KEY,
            "outputsize": "full"
        }
        
        if interval == "intraday":
            params["interval"] = "5min"
        
        response = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=10)
        data = response.json()
        
        # Find the time series key
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return {
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }

def calculate_moving_averages(prices: pd.Series) -> Dict:
    """Calculate various moving averages"""
    return {
        'sma_20': prices.rolling(window=20).mean(),
        'sma_50': prices.rolling(window=50).mean(),
        'sma_200': prices.rolling(window=200).mean(),
        'ema_12': prices.ewm(span=12, adjust=False).mean(),
        'ema_26': prices.ewm(span=26, adjust=False).mean()
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume"""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

def get_technical_signals(df: pd.DataFrame) -> Dict:
    """Generate buy/sell/hold signals from technical indicators"""
    signals = {}
    current_price = df['close'].iloc[-1]
    
    # RSI Signal
    rsi = calculate_rsi(df['close'])
    current_rsi = rsi.iloc[-1]
    if current_rsi < 30:
        signals['RSI'] = {'signal': 'BUY', 'value': current_rsi, 'reason': 'Oversold'}
    elif current_rsi > 70:
        signals['RSI'] = {'signal': 'SELL', 'value': current_rsi, 'reason': 'Overbought'}
    else:
        signals['RSI'] = {'signal': 'HOLD', 'value': current_rsi, 'reason': 'Neutral'}
    
    # MACD Signal
    macd_data = calculate_macd(df['close'])
    current_macd = macd_data['macd'].iloc[-1]
    current_signal = macd_data['signal'].iloc[-1]
    prev_macd = macd_data['macd'].iloc[-2]
    prev_signal = macd_data['signal'].iloc[-2]
    
    if current_macd > current_signal and prev_macd <= prev_signal:
        signals['MACD'] = {'signal': 'BUY', 'value': current_macd, 'reason': 'Bullish crossover'}
    elif current_macd < current_signal and prev_macd >= prev_signal:
        signals['MACD'] = {'signal': 'SELL', 'value': current_macd, 'reason': 'Bearish crossover'}
    else:
        signals['MACD'] = {'signal': 'HOLD', 'value': current_macd, 'reason': 'No crossover'}
    
    # Bollinger Bands Signal
    bb = calculate_bollinger_bands(df['close'])
    current_upper = bb['upper'].iloc[-1]
    current_lower = bb['lower'].iloc[-1]
    
    if current_price <= current_lower:
        signals['Bollinger'] = {'signal': 'BUY', 'value': current_price, 'reason': 'Below lower band'}
    elif current_price >= current_upper:
        signals['Bollinger'] = {'signal': 'SELL', 'value': current_price, 'reason': 'Above upper band'}
    else:
        signals['Bollinger'] = {'signal': 'HOLD', 'value': current_price, 'reason': 'Within bands'}
    
    # Moving Average Signal
    ma = calculate_moving_averages(df['close'])
    sma_50 = ma['sma_50'].iloc[-1]
    sma_200 = ma['sma_200'].iloc[-1]
    
    if pd.notna(sma_50) and pd.notna(sma_200):
        if current_price > sma_50 > sma_200:
            signals['MA'] = {'signal': 'BUY', 'value': sma_50, 'reason': 'Golden cross (50>200)'}
        elif current_price < sma_50 < sma_200:
            signals['MA'] = {'signal': 'SELL', 'value': sma_50, 'reason': 'Death cross (50<200)'}
        else:
            signals['MA'] = {'signal': 'HOLD', 'value': sma_50, 'reason': 'Mixed signals'}
    
    return signals

def generate_recommendation(signals: Dict) -> str:
    """Generate overall buy/sell/hold recommendation"""
    buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
    sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
    total = len(signals)
    
    buy_ratio = buy_count / total if total > 0 else 0
    sell_ratio = sell_count / total if total > 0 else 0
    
    if buy_ratio >= 0.6:
        return "STRONG BUY"
    elif buy_ratio >= 0.4:
        return "BUY"
    elif sell_ratio >= 0.6:
        return "STRONG SELL"
    elif sell_ratio >= 0.4:
        return "SELL"
    else:
        return "HOLD"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional-grade stock analysis for smarter trading decisions</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Stock Search")
        symbol = st.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, TSLA, INDO").upper()
        
        st.markdown("---")
        st.header("⚙️ Settings")
        analysis_period = st.selectbox("Analysis Period", ["Daily", "Intraday", "Weekly"], index=0)
        
        st.markdown("---")
        st.markdown("### 📚 Quick Links")
        st.markdown("- [Webull](https://www.webull.com)")
        st.markdown("- [Yahoo Finance](https://finance.yahoo.com)")
        st.markdown("- [TradingView](https://www.tradingview.com)")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Technical Analysis",
        "📰 Fundamentals & News",
        "💰 P/L Calculator",
        "🎯 Trade Recommendation",
        "📔 Trade Journal",
        "📅 Earnings Calendar"
    ])
    
    # ========================================================================
    # TAB 1: TECHNICAL ANALYSIS
    # ========================================================================
    with tab1:
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                df = fetch_stock_data(symbol, analysis_period.lower())
                
                if df is not None and not df.empty:
                    # Current price and change
                    current_price = df['close'].iloc[-1]
                    prev_price = df['close'].iloc[-2]
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100
                    
                    # Display current metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Current Price",
                            value=f"${current_price:.2f}",
                            delta=f"{price_change_pct:+.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            label="Volume",
                            value=f"{df['volume'].iloc[-1]:,.0f}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Day High",
                            value=f"${df['high'].iloc[-1]:.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            label="Day Low",
                            value=f"${df['low'].iloc[-1]:.2f}"
                        )
                    
                    # Technical Indicators
                    st.subheader("📈 Technical Indicators")
                    
                    # Calculate all indicators
                    rsi = calculate_rsi(df['close'])
                    macd_data = calculate_macd(df['close'])
                    bb = calculate_bollinger_bands(df['close'])
                    ma = calculate_moving_averages(df['close'])
                    atr = calculate_atr(df)
                    obv = calculate_obv(df)
                    
                    # Create main price chart with indicators
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Volume')
                    )
                    
                    # Candlestick chart
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
                    fig.add_trace(
                        go.Scatter(x=df.index, y=bb['upper'], name='BB Upper',
                                 line=dict(color='rgba(250, 128, 114, 0.5)', dash='dash')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df.index, y=bb['middle'], name='BB Middle',
                                 line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df.index, y=bb['lower'], name='BB Lower',
                                 line=dict(color='rgba(250, 128, 114, 0.5)', dash='dash')),
                        row=1, col=1
                    )
                    
                    # Moving Averages
                    fig.add_trace(
                        go.Scatter(x=df.index, y=ma['sma_50'], name='SMA 50',
                                 line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df.index, y=ma['sma_200'], name='SMA 200',
                                 line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    # RSI
                    fig.add_trace(
                        go.Scatter(x=df.index, y=rsi, name='RSI',
                                 line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    # MACD
                    fig.add_trace(
                        go.Scatter(x=df.index, y=macd_data['macd'], name='MACD',
                                 line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df.index, y=macd_data['signal'], name='Signal',
                                 line=dict(color='orange')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Bar(x=df.index, y=macd_data['histogram'], name='Histogram',
                              marker_color='gray'),
                        row=3, col=1
                    )
                    
                    # Volume
                    colors = ['red' if close < open else 'green' 
                             for close, open in zip(df['close'], df['open'])]
                    fig.add_trace(
                        go.Bar(x=df.index, y=df['volume'], name='Volume',
                              marker_color=colors),
                        row=4, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=1000,
                        showlegend=True,
                        template='plotly_dark',
                        hovermode='x unified',
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Indicator values
                    st.subheader("📊 Current Indicator Values")
                    
                    ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                    
                    with ind_col1:
                        current_rsi = rsi.iloc[-1]
                        rsi_color = "positive" if 30 <= current_rsi <= 70 else "negative"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>RSI (14)</h3>
                            <div class="value {rsi_color}">{current_rsi:.2f}</div>
                            <small>{'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ind_col2:
                        current_macd = macd_data['macd'].iloc[-1]
                        current_signal = macd_data['signal'].iloc[-1]
                        macd_color = "positive" if current_macd > current_signal else "negative"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>MACD</h3>
                            <div class="value {macd_color}">{current_macd:.4f}</div>
                            <small>Signal: {current_signal:.4f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ind_col3:
                        current_atr = atr.iloc[-1]
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>ATR (14)</h3>
                            <div class="value neutral">${current_atr:.2f}</div>
                            <small>Volatility measure</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ind_col4:
                        current_obv = obv.iloc[-1]
                        prev_obv = obv.iloc[-2]
                        obv_color = "positive" if current_obv > prev_obv else "negative"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>OBV</h3>
                            <div class="value {obv_color}">{current_obv:,.0f}</div>
                            <small>On-Balance Volume</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                else:
                    st.error("❌ Could not fetch stock data. Please check the ticker symbol or try again later.")
        else:
            st.info("👈 Enter a stock ticker in the sidebar to begin analysis")
    
    # ========================================================================
    # TAB 2: FUNDAMENTALS & NEWS
    # ========================================================================
    with tab2:
        st.subheader("📰 Fundamentals & News Analysis")
        st.info("🚧 Coming in Phase 2: Company financials, earnings data, news sentiment analysis")
        
        # Placeholder for fundamental data
        st.markdown("""
        ### Planned Features:
        - **Company Profile**: Market cap, P/E ratio, EPS, dividend yield
        - **Financial Statements**: Income statement, balance sheet, cash flow
        - **News Sentiment**: AI-powered sentiment analysis of recent news
        - **Analyst Ratings**: Buy/sell/hold recommendations from analysts
        - **Insider Trading**: Recent insider buying/selling activity
        """)
    
    # ========================================================================
    # TAB 3: P/L CALCULATOR
    # ========================================================================
    with tab3:
        st.subheader("💰 Profit/Loss Calculator")
        
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            st.markdown("### 📥 Entry Details")
            entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=100.0, step=0.01)
            shares = st.number_input("Number of Shares", min_value=1, value=100, step=1)
            entry_fees = st.number_input("Entry Fees/Commission ($)", min_value=0.0, value=0.0, step=0.01)
        
        with calc_col2:
            st.markdown("### 📤 Exit Details")
            exit_price = st.number_input("Exit Price ($)", min_value=0.01, value=110.0, step=0.01)
            exit_fees = st.number_input("Exit Fees/Commission ($)", min_value=0.0, value=0.0, step=0.01)
        
        # Calculations
        entry_cost = (entry_price * shares) + entry_fees
        exit_value = (exit_price * shares) - exit_fees
        profit_loss = exit_value - entry_cost
        profit_loss_pct = (profit_loss / entry_cost) * 100
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Results")
        
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        
        with result_col1:
            st.metric("Total Entry Cost", f"${entry_cost:,.2f}")
        
        with result_col2:
            st.metric("Total Exit Value", f"${exit_value:,.2f}")
        
        with result_col3:
            st.metric("Profit/Loss", f"${profit_loss:,.2f}", 
                     delta=f"{profit_loss_pct:+.2f}%",
                     delta_color="normal")
        
        with result_col4:
            risk_reward = abs((exit_price - entry_price) / (entry_price - (entry_price * 0.95)))
            st.metric("Risk/Reward Ratio", f"1:{risk_reward:.2f}")
        
        # Visual representation
        fig_pl = go.Figure()
        
        fig_pl.add_trace(go.Bar(
            x=['Entry Cost', 'Exit Value', 'Profit/Loss'],
            y=[entry_cost, exit_value, profit_loss],
            marker_color=['lightblue', 'lightgreen' if profit_loss > 0 else 'lightcoral', 
                         'green' if profit_loss > 0 else 'red'],
            text=[f'${entry_cost:,.2f}', f'${exit_value:,.2f}', f'${profit_loss:,.2f}'],
            textposition='auto',
        ))
        
        fig_pl.update_layout(
            title="Trade P/L Breakdown",
            template='plotly_dark',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_pl, use_container_width=True)
        
        # Position sizing calculator
        st.markdown("---")
        st.subheader("📏 Position Sizing Calculator")
        
        pos_col1, pos_col2 = st.columns(2)
        
        with pos_col1:
            account_size = st.number_input("Account Size ($)", min_value=100.0, value=10000.0, step=100.0)
            risk_percent = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        
        with pos_col2:
            stop_loss_pct = st.slider("Stop Loss (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
            
        risk_amount = account_size * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_pct / 100)
        shares_to_buy = int(position_size / entry_price)
        
        st.info(f"""
        **Position Sizing Recommendation:**
        - Risk Amount: ${risk_amount:,.2f}
        - Position Size: ${position_size:,.2f}
        - Shares to Buy: {shares_to_buy}
        - Stop Loss Price: ${entry_price * (1 - stop_loss_pct/100):.2f}
        """)
    
    # ========================================================================
    # TAB 4: TRADE RECOMMENDATION
    # ========================================================================
    with tab4:
        st.subheader("🎯 AI-Powered Trade Recommendation")
        
        if symbol:
            df = fetch_stock_data(symbol, analysis_period.lower())
            
            if df is not None and not df.empty:
                # Get all technical signals
                signals = get_technical_signals(df)
                overall_rec = generate_recommendation(signals)
                
                # Display overall recommendation
                rec_color = {
                    'STRONG BUY': 'success',
                    'BUY': 'success',
                    'HOLD': 'warning',
                    'SELL': 'danger',
                    'STRONG SELL': 'danger'
                }
                
                st.markdown(f"""
                <div class="{rec_color.get(overall_rec, 'warning')}-box">
                    <h2 style="margin: 0;">Overall Recommendation: {overall_rec}</h2>
                    <p>Based on {len(signals)} technical indicators</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("📊 Signal Breakdown")
                
                # Display individual signals
                signal_cols = st.columns(len(signals))
                
                for idx, (indicator, data) in enumerate(signals.items()):
                    with signal_cols[idx]:
                        signal_class = f"signal-{data['signal'].lower()}"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{indicator}</h3>
                            <span class="signal-badge {signal_class}">{data['signal']}</span>
                            <p style="margin-top: 1rem; font-size: 0.9rem;">{data['reason']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Action plan
                st.markdown("---")
                st.subheader("📝 Suggested Action Plan")
                
                current_price = df['close'].iloc[-1]
                atr = calculate_atr(df).iloc[-1]
                
                if overall_rec in ['STRONG BUY', 'BUY']:
                    entry_suggestion = current_price
                    stop_loss = entry_suggestion - (2 * atr)
                    take_profit_1 = entry_suggestion + (2 * atr)
                    take_profit_2 = entry_suggestion + (4 * atr)
                    
                    st.success(f"""
                    **BUY PLAN:**
                    - Entry Price: ${entry_suggestion:.2f}
                    - Stop Loss: ${stop_loss:.2f} (-{((entry_suggestion - stop_loss) / entry_suggestion * 100):.1f}%)
                    - Take Profit 1: ${take_profit_1:.2f} (+{((take_profit_1 - entry_suggestion) / entry_suggestion * 100):.1f}%)
                    - Take Profit 2: ${take_profit_2:.2f} (+{((take_profit_2 - entry_suggestion) / entry_suggestion * 100):.1f}%)
                    
                    **Risk/Reward Ratio:** 1:{((take_profit_1 - entry_suggestion) / (entry_suggestion - stop_loss)):.2f}
                    """)
                
                elif overall_rec in ['SELL', 'STRONG SELL']:
                    st.warning(f"""
                    **SELL/AVOID PLAN:**
                    - Consider exiting existing positions
                    - Wait for better entry (if planning to buy)
                    - Set alerts for reversal signals
                    - Current price: ${current_price:.2f}
                    """)
                
                else:
                    st.info(f"""
                    **HOLD/WAIT PLAN:**
                    - Mixed signals - wait for clearer trend
                    - Monitor for breakout above ${current_price + atr:.2f}
                    - Or breakdown below ${current_price - atr:.2f}
                    - Current price: ${current_price:.2f}
                    """)
        else:
            st.info("👈 Enter a stock ticker to get trade recommendations")
    
    # ========================================================================
    # TAB 5: TRADE JOURNAL
    # ========================================================================
    with tab5:
        st.subheader("📔 Trade Journal")
        st.info("🚧 Coming in Phase 3: Full trade tracking with emotions, strategies, and performance analytics")
        
        st.markdown("""
        ### Planned Features:
        - **Trade Entry Form**: Log all trade details (entry/exit, P/L, strategy)
        - **Emotional Tracking**: Record your emotions and mindset
        - **Lessons Learned**: Document what worked and what didn't
        - **Performance Analytics**: Win rate, average P/L, best setups
        - **Trade Screenshots**: Upload chart screenshots
        - **Goals & Progress**: Track improvement over time
        - **Export Data**: Download your journal as CSV/Excel
        """)
        
        # Placeholder form
        with st.expander("✏️ Log New Trade (Preview)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.date_input("Trade Date")
                st.text_input("Ticker Symbol")
                st.selectbox("Trade Type", ["Long", "Short"])
                st.number_input("Entry Price", value=0.0)
                st.number_input("Exit Price", value=0.0)
                st.number_input("Shares/Contracts", value=1)
            
            with col2:
                st.selectbox("Strategy Used", ["Breakout", "Trend Following", "Mean Reversion", "News Play", "Other"])
                st.text_area("Entry Reason")
                st.text_area("Exit Reason")
                st.slider("Emotional State (1-10)", 1, 10, 5)
                st.text_area("Lessons Learned")
            
            st.button("Save Trade (Not functional yet)", disabled=True)
    
    # ========================================================================
    # TAB 6: EARNINGS CALENDAR
    # ========================================================================
    with tab6:
        st.subheader("📅 Earnings Calendar")
        st.info("🚧 Coming in Phase 2: Upcoming earnings dates, estimates, and historical performance")
        
        st.markdown("""
        ### Planned Features:
        - **This Week's Earnings**: Companies reporting earnings this week
        - **Earnings Estimates**: Expected vs actual EPS and revenue
        - **Historical Performance**: How stock moved after past earnings
        - **Earnings Surprises**: Track beats and misses
        - **Pre-Earnings Analysis**: Technical setup before earnings
        - **Alerts**: Get notified before key earnings reports
        """)

if __name__ == "__main__":
    main()
