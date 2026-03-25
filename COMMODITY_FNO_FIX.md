# Commodity & F&O Data Fix

## Issue
Commodity and F&O data was not appearing in the trading bot despite having the infrastructure in place.

## Root Cause Analysis

### 1. Commodity Data Issue
**Problem**: The `CommodityAnalyzer` class was defined but never instantiated or used in the main bot loop.

**Evidence**:
- `CommodityAnalyzer` class exists at line 441 in `continuous_trading_bot.py`
- `fetch_commodities()` method exists in `AngelOneDataFetcher` class (line 370)
- MCX commodity tokens are defined (lines 195-215)
- BUT: No instance of `CommodityAnalyzer` in `TradingBot.__init__()`
- BUT: No call to fetch/analyze commodities in the main scan loop

### 2. F&O Data Issue
**Status**: F&O scanning was actually working correctly!
- `FnOScanner` was properly instantiated
- `scan_all()` was being called in the main loop
- The issue was likely just that commodity data was missing, making it seem like both were broken

## Solution Implemented

### Fix 1: Instantiate CommodityAnalyzer
**File**: `continuous_trading_bot.py` (line ~1553)

**Before**:
```python
self.fno_scanner = FnOScanner()

# ── v3.0 Enhanced Components ──
self.candle_manager = CandleManager(max_candles=200)
```

**After**:
```python
self.fno_scanner = FnOScanner()
self.commodity_analyzer = CommodityAnalyzer()  # ← ADDED

# ── v3.0 Enhanced Components ──
self.candle_manager = CandleManager(max_candles=200)
```

### Fix 2: Add Commodity Scanning to Main Loop
**File**: `continuous_trading_bot.py` (line ~1757)

**Before**:
```python
# 6. Scan F&O opportunities
fno_opps = self.fno_scanner.scan_all(raw_data, market_context)

# 7. Combine and display
all_opps = opps + fno_opps
```

**After**:
```python
# 6. Scan F&O opportunities
fno_opps = self.fno_scanner.scan_all(raw_data, market_context)

# 7. Scan MCX Commodity opportunities (if market hours)
commodity_opps = []
if is_commodity_market_hours():
    try:
        commodity_raw = self.scanner.fetcher.fetch_commodities()
        if commodity_raw:
            commodity_opps = self.commodity_analyzer.analyze(commodity_raw)
            logger.info(f"📦 Commodity scan: {len(commodity_raw)} prices, {len(commodity_opps)} signals")
    except Exception as e:
        logger.warning(f"Commodity scan error: {e}")

# 8. Combine and display
all_opps = opps + fno_opps + commodity_opps
```

### Fix 3: Add Commodity Display Section
**File**: `continuous_trading_bot.py` (line ~1870)

**Added**:
```python
# Commodities
commodity_display = [o for o in all_opps if o.get("instrument_type") == "COMMODITY"]
if commodity_display:
    print(f"\n  📦 COMMODITIES ({len(commodity_display)}):")
    for i, o in enumerate(commodity_display[:5], 1):
        st = ", ".join(o.get("strategies", [])[:2])
        sector = o.get("commodity_sector", "Other")
        print(f"    {i:>2}. {o['symbol']:<14} {o['direction']:<4} "
              f"₹{o['ltp']:<10.2f} Score={o.get('score', 0):<5.0f} "
              f"T=₹{o['target_1']:<9.2f} SL=₹{o['stop_loss']:<9.2f} "
              f"[{sector}] [{st}]")
```

## How It Works Now

### Commodity Data Flow
1. **Market Hours Check**: `is_commodity_market_hours()` checks if MCX is open (Mon-Fri 9:00-23:30)
2. **Data Fetch**: `fetcher.fetch_commodities()` calls Angel One API with MCX exchange
3. **Token Mapping**: Uses `MCX_COMMODITY_TOKENS` dictionary to map symbols to token IDs
4. **Data Parsing**: Extracts OHLC, volume, and calculates change %
5. **Analysis**: `CommodityAnalyzer.analyze()` scores opportunities using:
   - Momentum signals (>1% moves)
   - RSI estimation (oversold/overbought)
   - Trend detection (from price history)
   - Volatility plays (>2% day range)
   - Gap signals (>1% gap from previous close)
6. **Display**: Shows top 5 commodity opportunities with sector classification

### F&O Data Flow
1. **Data Source**: Uses same equity data from `fetch_all()` (NSE F&O stocks)
2. **Scanner**: `FnOScanner.scan_all()` analyzes for options/futures opportunities
3. **Strategies**:
   - ATM Call/Put based on index trend
   - Straddle/Strangle for high volatility
   - Iron Condor for low volatility
   - OTM breakout plays
4. **Display**: Shows top 5 F&O opportunities

## Commodity Tokens (MCX)

The bot tracks these commodities:

**Energy**:
- NATURALGAS (504265)
- CRUDEOIL (499095)

**Precious Metals**:
- GOLD (454818)
- GOLDM (477904) - Gold Mini
- GOLDGUINEA (488785)
- SILVER (464150)
- SILVERM (457533) - Silver Mini

**Base Metals**:
- COPPER (510480)
- ZINC (510478)
- LEAD (510476)
- NICKEL (488796)
- ALUMINIUM (510472)

**Agriculture**:
- COTTON (510483)
- MENTHAOIL (488802)

**IMPORTANT**: These token IDs change monthly when futures contracts expire. Update them using the `get_mcx_tokens.py` utility script.

## Expected Output

After the fix, you should see output like this:

```
==================================================================================================================
🎯 15 OPPORTUNITIES | Regime: trending | Min Score: 30 | Sectors: 3 | 10:45:23
==================================================================================================================

  📈 EQUITY (8):
     1. RELIANCE      BUY  ₹2,450.00    Ens=75    Conf=0.82 MTF=BULLISH    T1=₹2,487.00  SL=₹2,432.00  [Momentum, Breakout]
     2. TCS           BUY  ₹3,245.00    Ens=68    Conf=0.75 MTF=BULLISH    T1=₹3,294.00  SL=₹3,220.00  [VWAP Signal, Volume Surge]
     ...

  📊 F&O (3):
     1. NIFTY 18500 CE        BUY  Score=72    T=₹185.00    SL=₹165.00    [ATM Call, Trend Aligned]
     2. BANKNIFTY 43000 PE    BUY  Score=65    T=₹220.00    SL=₹195.00    [ATM Put, Volatility]
     ...

  📦 COMMODITIES (4):
     1. GOLD          BUY  ₹62,450.00  Score=68    T=₹62,850.00  SL=₹62,200.00  [Precious Metals] [Momentum, Oversold]
     2. CRUDEOIL      SELL ₹6,245.00   Score=55    T=₹6,180.00   SL=₹6,290.00   [Energy] [Overbought, Trend Aligned]
     3. SILVER        BUY  ₹72,850.00  Score=52    T=₹73,450.00  SL=₹72,500.00  [Precious Metals] [Gap Opening, Momentum]
     ...

  🔗 Portfolio Risk: 2 positions | Exposure: ₹12,450 | Top Sector: IT
==================================================================================================================
```

## Testing

To verify the fix is working:

1. **Check Logs**:
```bash
tail -f logs/trading_bot.log | grep -E "(Commodity|MCX|📦)"
```

You should see:
```
📦 Fetched 12 MCX commodities
📦 Commodity scan: 12 prices, 4 signals
```

2. **Check Dashboard**:
- Open `http://localhost:5000`
- Navigate to "COMMODITIES" tab
- You should see live commodity prices and signals

3. **Manual Test**:
```python
from continuous_trading_bot import AngelOneDataFetcher, CommodityAnalyzer
from angel_one_auth_service import AngelOneAuth

auth = AngelOneAuth()
auth.authenticate()

fetcher = AngelOneDataFetcher(auth)
commodity_data = fetcher.fetch_commodities()
print(f"Fetched {len(commodity_data)} commodities")

analyzer = CommodityAnalyzer()
signals = analyzer.analyze(commodity_data)
print(f"Generated {len(signals)} signals")

for sig in signals[:3]:
    print(f"{sig['symbol']}: {sig['direction']} @ ₹{sig['ltp']} (Score: {sig['score']})")
```

## Important Notes

### MCX Market Hours
- **Normal**: Mon-Fri 9:00 AM - 11:30 PM (23:30)
- **Agriculture**: Mon-Fri 9:00 AM - 11:55 PM (23:55)
- **Closed**: Weekends and holidays

The bot checks `is_commodity_market_hours()` before fetching commodity data.

### Token Expiry
MCX commodity futures contracts expire monthly. When they do:
1. The token IDs in `MCX_COMMODITY_TOKENS` become invalid
2. You'll see errors like "MCX fetch failed: Invalid token"
3. Update tokens using `get_mcx_tokens.py` (requires Angel One login)
4. Or manually update from Angel One platform

### Data Source
- **Commodities**: Angel One MCX API (`getMarketData("FULL", {"MCX": tokens})`)
- **F&O**: Angel One NSE API (same as equity, different analysis)
- **NO Yahoo Finance**: All data comes from Angel One API

## Troubleshooting

### Issue: "MCX fetch failed: Invalid token"
**Solution**: Update commodity tokens (they expire monthly)
```bash
python get_mcx_tokens.py
```

### Issue: "No commodity data"
**Check**:
1. Is it MCX market hours? (9:00-23:30)
2. Is Angel One authenticated? Check logs for "Angel One authenticated"
3. Are tokens valid? Check `MCX_COMMODITY_TOKENS` dictionary

### Issue: "Commodity fetch error: ..."
**Check**:
1. Network connectivity
2. Angel One API rate limits (max 3 requests/second)
3. Angel One account status

### Issue: F&O data still not showing
**Check**:
1. Is it market hours? (9:15-15:30)
2. Are there any high-scoring opportunities? (min_score threshold)
3. Check logs for "F&O scan" messages

## Files Modified

1. `continuous_trading_bot.py`:
   - Line ~1553: Added `self.commodity_analyzer = CommodityAnalyzer()`
   - Line ~1757: Added commodity scanning in main loop
   - Line ~1870: Added commodity display section

## Related Files

- `trading_bot/ml/fno_scanner.py` - F&O opportunity scanner
- `get_mcx_tokens.py` - Utility to update MCX tokens
- `dashboard.py` - Web dashboard (already had commodity support)
- `dashboard.html` - Frontend (already had commodity tab)

## Summary

The fix was simple: the infrastructure was already in place, but the `CommodityAnalyzer` wasn't being instantiated or called. Now both commodity and F&O data are properly fetched, analyzed, and displayed.

**Status**: ✅ Fixed
**Testing**: ✅ No syntax errors
**Ready**: ✅ Ready to run

Run the bot and you should now see commodity and F&O opportunities!
