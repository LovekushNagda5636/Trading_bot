#!/usr/bin/env python3
"""
Get current MCX commodity token IDs from Angel One API
Run this monthly to update token IDs when contracts expire
"""

from SmartApi import SmartConnect
import json

# Your credentials
API_KEY = "2lXNdnyC"
CLIENT_CODE = "L57658270"

def get_mcx_tokens():
    """Fetch current MCX commodity tokens from Angel One."""
    try:
        smart_api = SmartConnect(api_key=API_KEY)
        
        print("🔍 Searching for MCX commodity tokens...")
        print("="*70)
        
        # Common MCX commodities to search for
        commodities = [
            "GOLD", "GOLDM", "GOLDGUINEA",
            "SILVER", "SILVERM",
            "CRUDEOIL", "NATURALGAS",
            "COPPER", "ZINC", "LEAD", "NICKEL", "ALUMINIUM",
            "COTTON", "MENTHAOIL"
        ]
        
        tokens = {}
        
        for commodity in commodities:
            try:
                # Search for the symbol
                response = smart_api.searchScrip("MCX", commodity)
                
                if response and response.get('status') and response.get('data'):
                    # Get the first (most active) contract
                    for item in response['data']:
                        symbol = item.get('symbol', '')
                        token = item.get('symboltoken', '')
                        tradingsymbol = item.get('tradingsymbol', '')
                        
                        # Look for near-month contract (usually has current/next month)
                        if token and symbol:
                            tokens[commodity] = {
                                'token': token,
                                'symbol': symbol,
                                'tradingsymbol': tradingsymbol
                            }
                            print(f"✅ {commodity:15} → Token: {token:10} | {tradingsymbol}")
                            break
                            
            except Exception as e:
                print(f"⚠️  {commodity:15} → Error: {e}")
        
        print("="*70)
        print(f"\n📊 Found {len(tokens)} commodity tokens")
        
        # Generate Python code
        print("\n" + "="*70)
        print("📝 Copy this to continuous_trading_bot.py:")
        print("="*70)
        print("\nMCX_COMMODITY_TOKENS = {")
        
        # Group by category
        energy = ["CRUDEOIL", "NATURALGAS"]
        precious = ["GOLD", "GOLDM", "GOLDGUINEA", "SILVER", "SILVERM"]
        base = ["COPPER", "ZINC", "LEAD", "NICKEL", "ALUMINIUM"]
        agri = ["COTTON", "MENTHAOIL"]
        
        if any(c in tokens for c in energy):
            print("    # Energy")
            for c in energy:
                if c in tokens:
                    print(f'    "{c}": "{tokens[c]["token"]}",')
        
        if any(c in tokens for c in precious):
            print("    # Precious Metals")
            for c in precious:
                if c in tokens:
                    print(f'    "{c}": "{tokens[c]["token"]}",')
        
        if any(c in tokens for c in base):
            print("    # Base Metals")
            for c in base:
                if c in tokens:
                    print(f'    "{c}": "{tokens[c]["token"]}",')
        
        if any(c in tokens for c in agri):
            print("    # Agriculture")
            for c in agri:
                if c in tokens:
                    print(f'    "{c}": "{tokens[c]["token"]}",')
        
        print("}")
        print("\n" + "="*70)
        
        return tokens
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  This script requires Angel One login.")
        print("💡 Alternative: Check Angel One web/app for current contract tokens")
        return {}

if __name__ == "__main__":
    print("🤖 MCX Token Fetcher - Angel One API")
    print("="*70)
    print("⚠️  NOTE: This requires Angel One authentication")
    print("💡 Without login, tokens must be manually updated from Angel One platform")
    print("="*70)
    
    tokens = get_mcx_tokens()
    
    if not tokens:
        print("\n📋 Manual Update Instructions:")
        print("="*70)
        print("1. Login to Angel One web/mobile app")
        print("2. Go to MCX section")
        print("3. For each commodity (GOLD, SILVER, CRUDEOIL, etc.):")
        print("   - Note the active contract month")
        print("   - Get the symbol token from contract details")
        print("4. Update MCX_COMMODITY_TOKENS in continuous_trading_bot.py")
        print("="*70)
