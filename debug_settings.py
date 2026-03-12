from trading_bot.core.config import Settings
try:
    s = Settings()
    print("Settings loaded successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
