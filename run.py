#!/usr/bin/env python3
import subprocess
import sys
import time
import os
from pathlib import Path

def run():
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("🚀 STARTING ANGEL ONE SELF-LEARNING TRADING BOT & DASHBOARD")
    print("="*60 + "\n")

    # Open log file — dashboard output goes here to keep terminal clean
    dash_log = open("logs/dashboard.log", "a", encoding="utf-8")

    # 1. Start the Dashboard silently (output → log file only)
    print("📊 Starting Dashboard [dashboard.py] → logs/dashboard.log ...")
    dash_proc = subprocess.Popen(
        [sys.executable, "-u", "dashboard.py", "--port", "8888"],
        stdout=dash_log,   # Dashboard output → log file
        stderr=dash_log,   # SmartAPI [E] / Flask noise → log file
    )

    time.sleep(1)

    print("\n✅ Services started!")
    print("-" * 40)
    print("🔗 DASHBOARD: http://127.0.0.1:8888")
    print("📂 BOT LOGS:  logs/trading_bot.log")
    print("📂 DASH LOGS: logs/dashboard.log")
    print("-" * 40)
    print("💡 Press Ctrl+C to stop all services safely.\n")
    print("=" * 60)
    print("  BOT OUTPUT  (live below)")
    print("=" * 60 + "\n")

    # 2. Start the Trading Bot — stdout to THIS terminal, stderr to log
    bot_log = open("logs/trading_bot_err.log", "a", encoding="utf-8")
    bot_proc = subprocess.Popen(
        [sys.executable, "-u", "continuous_trading_bot.py"],
        stdout=None,       # Bot print() and logger → terminal  ✅
        stderr=bot_log,    # SmartAPI [E] noise → log file only ✅
    )

    try:
        while True:
            # Check if processes are still alive
            if bot_proc.poll() is not None:
                print("\n❌ Trading Bot stopped unexpectedly. Check logs/trading_bot.log")
                break
            if dash_proc.poll() is not None:
                print("\n❌ Dashboard stopped unexpectedly. Check logs/dashboard.log")
                break
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n🛑 Received stop signal. Shutting down...")
        bot_proc.terminate()
        dash_proc.terminate()
        bot_proc.wait()
        dash_proc.wait()
        dash_log.close()
        bot_log.close()
        print("✅ Both services stopped. Goodbye!")

if __name__ == "__main__":
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found. Credentials might be missing.")
    run()
