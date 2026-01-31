"""
Stock Analysis Platform - DIAGNOSTIC VERSION
This version will show you if your API keys are working
"""

import streamlit as st
import requests
import pandas as pd

st.title("🔍 API Diagnostic Tool")

# Show what keys are loaded
st.subheader("Step 1: Check API Keys")

try:
    alpha_key = st.secrets.get("ALPHA_VANTAGE_KEY", "NOT_FOUND")
    fmp_key = st.secrets.get("FMP_KEY", "NOT_FOUND")
    
    st.write(f"**Alpha Vantage Key Found:** {alpha_key[:4]}...{alpha_key[-4:] if len(alpha_key) > 8 else 'TOO SHORT'}")
    st.write(f"**FMP Key Found:** {fmp_key[:4]}...{fmp_key[-4:] if len(fmp_key) > 8 else 'TOO SHORT'}")
    
    if alpha_key == "NOT_FOUND":
        st.error("❌ ALPHA_VANTAGE_KEY not found in secrets!")
        st.stop()
    
    if alpha_key == "demo":
        st.warning("⚠️ Using demo key - limited functionality")
    
except Exception as e:
    st.error(f"❌ Error reading secrets: {e}")
    st.stop()

st.success("✅ API keys are loaded!")

# Test the API
st.subheader("Step 2: Test API Connection")

symbol = st.text_input("Enter ticker to test", "AAPL")

if st.button("Test API Call"):
    with st.spinner("Testing API..."):
        try:
            # Make API call
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": alpha_key
            }
            
            st.write(f"**Calling:** {url}")
            st.write(f"**Parameters:** function=GLOBAL_QUOTE, symbol={symbol}, apikey={alpha_key[:4]}...")
            
            response = requests.get(url, params=params, timeout=10)
            
            st.write(f"**Status Code:** {response.status_code}")
            
            # Show raw response
            data = response.json()
            st.write("**Raw API Response:**")
            st.json(data)
            
            # Check for errors
            if "Error Message" in data:
                st.error(f"❌ API Error: {data['Error Message']}")
            elif "Note" in data:
                st.error(f"⚠️ Rate Limit Hit: {data['Note']}")
            elif "Global Quote" in data:
                quote = data["Global Quote"]
                if quote:
                    st.success("✅ API Call Successful!")
                    st.metric("Price", f"${quote.get('05. price', 'N/A')}")
                    st.metric("Change", quote.get('09. change', 'N/A'))
                else:
                    st.warning("⚠️ No data returned - might be invalid ticker")
            else:
                st.warning("⚠️ Unexpected response format")
                
        except requests.exceptions.Timeout:
            st.error("❌ API request timed out")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

st.markdown("---")
st.subheader("Step 3: Check API Limits")
st.info("""
**Alpha Vantage Free Tier Limits:**
- 25 API calls per day
- 5 calls per minute

If you see a 'Note' message about rate limiting, you've hit your daily limit.
Wait until tomorrow or upgrade to a paid plan.
""")

st.markdown("---")
st.subheader("Troubleshooting Tips")
st.markdown("""
**If keys show "NOT_FOUND":**
1. Go to Settings → Secrets in Streamlit
2. Add your keys exactly like this:
```
ALPHA_VANTAGE_KEY = "your_key_here"
FMP_KEY = "your_key_here"
```
3. Save and reboot the app

**If you see "Note" in the response:**
- You've hit the 25 calls/day limit
- Wait 24 hours from your first API call today
- Or get a premium key ($50/month unlimited)

**If status code is not 200:**
- Network/API issue
- Try again in a few minutes

**If "Error Message" appears:**
- Invalid ticker symbol
- Try AAPL, MSFT, or TSLA
""")
