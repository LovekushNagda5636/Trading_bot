#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AWS EC2 Deployment Script — Angel One Trading Bot
# Target: Ubuntu 22.04 LTS
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

APP_DIR="/home/ubuntu/trading-bot"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="trading-bot"
DASHBOARD_SERVICE="trading-dashboard"

echo "═══════════════════════════════════════════════════════════"
echo "  TRADING BOT — AWS EC2 DEPLOYMENT"
echo "═══════════════════════════════════════════════════════════"

# ── 1. System Dependencies ──────────────────────────────────────
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv git

# ── 2. Application Directory ────────────────────────────────────
echo "[2/7] Setting up application directory..."
mkdir -p "$APP_DIR/logs" "$APP_DIR/data/journal" "$APP_DIR/data/models" "$APP_DIR/config"

# ── 3. Virtual Environment ──────────────────────────────────────
echo "[3/7] Creating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 4. Dependencies ────────────────────────────────────────────
echo "[4/7] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$APP_DIR/requirements.txt"

# ── 5. Environment Variables ───────────────────────────────────
echo "[5/7] Checking .env file..."
if [ ! -f "$APP_DIR/.env" ]; then
    echo "⚠️  No .env file found! Creating from template..."
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "❗ IMPORTANT: Edit $APP_DIR/.env with your real credentials!"
    echo "   Run: nano $APP_DIR/.env"
fi

# Restrict .env permissions
chmod 600 "$APP_DIR/.env"

# ── 6. Systemd Services ───────────────────────────────────────
echo "[6/7] Installing systemd services..."
sudo cp "$APP_DIR/deploy/trading-bot.service" /etc/systemd/system/
sudo cp "$APP_DIR/deploy/trading-dashboard.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl enable "$DASHBOARD_SERVICE"

# ── 7. Start Services ─────────────────────────────────────────
echo "[7/7] Starting services..."
sudo systemctl start "$SERVICE_NAME"
sleep 3
sudo systemctl start "$DASHBOARD_SERVICE"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Bot status:       sudo systemctl status $SERVICE_NAME"
echo "  Dashboard status: sudo systemctl status $DASHBOARD_SERVICE"
echo "  Bot logs:         sudo journalctl -u $SERVICE_NAME -f"
echo "  App logs:         tail -f $APP_DIR/logs/trading_bot.log"
echo "  Dashboard:        http://<your-ec2-ip>:8888"
echo ""
echo "  Stop bot:         sudo systemctl stop $SERVICE_NAME"
echo "  Restart bot:      sudo systemctl restart $SERVICE_NAME"
echo ""
echo "  ⚠️  CHECKLIST:"
echo "  [ ] Edit .env with real credentials"
echo "  [ ] Open port 8888 in EC2 Security Group"
echo "  [ ] Set DASHBOARD_TOKEN in .env"
echo "  [ ] Test with paper trading first"
echo "═══════════════════════════════════════════════════════════"
