#!/usr/bin/env python3
"""
Angel One Authentication Service
Handles login with and WITHOUT TOTP (2FA).

All credentials are loaded from environment variables (.env file).
NEVER hardcode secrets in this file or config files.
"""

import os
import pyotp
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from SmartApi import SmartConnect

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional — user can set env vars directly

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/angel_one_config.json"


def _load_config() -> Dict:
    """Load non-secret config from config file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {CONFIG_PATH}")
        return {}


class AngelOneAuth:
    """
    Handles Angel One SmartAPI authentication.
    
    Credentials are loaded from environment variables:
      - ANGEL_ONE_API_KEY
      - ANGEL_ONE_CLIENT_CODE
      - ANGEL_ONE_PASSWORD
      - ANGEL_ONE_TOTP_SECRET  (if 2FA is enabled)
      - ANGEL_ONE_TOTP_ENABLED (true/false)
    """

    def __init__(self):
        # Load credentials from environment variables ONLY
        self.api_key     = os.environ.get("ANGEL_ONE_API_KEY", "")
        self.client_code = os.environ.get("ANGEL_ONE_CLIENT_CODE", "")
        self.password    = os.environ.get("ANGEL_ONE_PASSWORD", "")
        self.totp_secret = os.environ.get("ANGEL_ONE_TOTP_SECRET", "")
        self.totp_enabled = os.environ.get("ANGEL_ONE_TOTP_ENABLED", "false").lower() == "true"

        # Validate that required credentials exist
        if not self.api_key:
            logger.warning("ANGEL_ONE_API_KEY not set in environment")
        if not self.client_code:
            logger.warning("ANGEL_ONE_CLIENT_CODE not set in environment")
        if not self.password:
            logger.warning("ANGEL_ONE_PASSWORD not set in environment")

        # Session state
        self.smart_api: Optional[SmartConnect] = None
        self.auth_token:    Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.feed_token:    Optional[str] = None
        self.is_authenticated = False
        self.session_expiry:  Optional[datetime] = None
        self.last_auth_time:  Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def login(self, password: str = None, totp_secret: str = None) -> bool:
        """
        Log in to Angel One.

        Args:
            password:    Override password (uses env var if not given)
            totp_secret: Override TOTP secret (uses env var if not given).

        Returns:
            True on success
        """
        pw = password or self.password
        ts = totp_secret if totp_secret is not None else self.totp_secret

        if not pw:
            logger.error("Password is empty. Set ANGEL_ONE_PASSWORD environment variable.")
            return False

        for attempt in range(1, 4):
            logger.info(f"🔄 Login attempt {attempt}/3 ...")
            if self._do_login(pw, ts or None):
                self.last_auth_time = datetime.now()
                self.session_expiry = datetime.now() + timedelta(hours=8)
                logger.info("✅ Logged in successfully")
                return True
            time.sleep(2 * attempt)

        logger.error("❌ Login failed after 3 attempts")
        return False

    def is_session_valid(self) -> bool:
        if not self.is_authenticated or not self.auth_token:
            return False
        if self.session_expiry and datetime.now() > self.session_expiry:
            logger.warning("Session expired")
            return False
        return True

    def ensure_authenticated(self) -> bool:
        """Return True if session is valid; try token refresh then re-login."""
        if self.is_session_valid():
            return True
        if self.refresh_token and self._refresh_session():
            return True
        # Last resort: full re-login
        logger.warning("Session invalid — attempting re-login")
        return self.login()

    def get_session_info(self) -> Dict[str, Any]:
        return {
            "is_authenticated":  self.is_authenticated,
            "client_code":       self.client_code,
            "session_expiry":    self.session_expiry.isoformat() if self.session_expiry else None,
            "last_auth_time":    self.last_auth_time.isoformat() if self.last_auth_time else None,
            "has_auth_token":    bool(self.auth_token),
            "has_refresh_token": bool(self.refresh_token),
            "has_feed_token":    bool(self.feed_token),
            "totp_enabled":      self.totp_enabled,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_login(self, password: str, totp_secret: Optional[str]) -> bool:
        try:
            self.smart_api = SmartConnect(api_key=self.api_key)

            # Generate TOTP only if secret is provided and non-empty
            totp_token = None
            if totp_secret:
                totp_token = self._generate_totp(totp_secret)
                if not totp_token:
                    logger.error("TOTP generation failed — check your TOTP secret")
                    return False
                # SECURITY: Never log the actual TOTP token
                logger.info("🔢 TOTP generated successfully")
            else:
                logger.info("ℹ️  No TOTP — logging in with password only")

            data = self.smart_api.generateSession(
                clientCode=self.client_code,
                password=password,
                totp=totp_token
            )

            if not data or not data.get("status"):
                msg = data.get("message", "Unknown error") if data else "No response"
                logger.error(f"Login rejected: {msg}")
                return False

            session = data.get("data", {})
            self.auth_token    = session.get("jwtToken")
            self.refresh_token = session.get("refreshToken")
            self.feed_token    = session.get("feedToken")

            if not self.auth_token:
                logger.error("No JWT token in response")
                return False

            self.is_authenticated = True
            return self._verify_profile()

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def _generate_totp(self, secret: str) -> Optional[str]:
        try:
            token = pyotp.TOTP(secret).now()
            if len(token) == 6 and token.isdigit():
                return token
            logger.error("Bad TOTP format")
            return None
        except Exception as e:
            logger.error(f"TOTP error: {e}")
            return None

    def _verify_profile(self) -> bool:
        try:
            profile = self.smart_api.getProfile(refreshToken=self.refresh_token)
            if profile and profile.get("status"):
                user = profile.get("data", {})
                logger.info(f"👤 {user.get('name', 'N/A')} | {user.get('email', 'N/A')}")
                return True
            logger.error(f"Profile check failed: {profile.get('message') if profile else 'No response'}")
            return False
        except Exception as e:
            logger.error(f"Profile check error: {e}")
            return False

    def _refresh_session(self) -> bool:
        try:
            data = self.smart_api.generateToken(self.refresh_token)
            if data and data.get("status"):
                session = data.get("data", {})
                self.auth_token  = session.get("jwtToken")
                self.feed_token  = session.get("feedToken")
                self.session_expiry = datetime.now() + timedelta(hours=8)
                logger.info("✅ Session refreshed")
                return True
            logger.error("Session refresh failed")
            return False
        except Exception as e:
            logger.error(f"Session refresh error: {e}")
            return False


# ---------------------------------------------------------------------------
# TOTP Setup Guide
# ---------------------------------------------------------------------------

TOTP_SETUP_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║           TOTP / 2FA — DO YOU NEED IT?                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Angel One SmartAPI requires 2FA for API access.            ║
║                                                              ║
║  HOW TO GET YOUR TOTP SECRET:                               ║
║  1. Open Angel One app on your phone                        ║
║  2. Go to: My Profile → Settings → Two-Factor Auth          ║
║  3. Enable TOTP authenticator                               ║
║  4. A QR code appears — also look for "Can't scan?"         ║
║     → Click it to see the SECRET KEY (32-character string)  ║
║  5. Copy that secret key                                    ║
║  6. Add it to your .env file:                               ║
║        ANGEL_ONE_TOTP_SECRET=your_secret_here               ║
║        ANGEL_ONE_TOTP_ENABLED=true                          ║
║                                                             ║
║  IMPORTANT: The secret key is NOT the 6-digit code.         ║
║  It is the long text like: JBSWY3DPEHPK3PXP                 ║
║                                                             ║
╚══════════════════════════════════════════════════════════════╝
"""


def test_login():
    """Quick test of Angel One login."""
    print("🔐 Angel One Login Test")
    print("=" * 50)
    print(TOTP_SETUP_GUIDE)

    auth = AngelOneAuth()

    if not auth.password:
        print("❌ ANGEL_ONE_PASSWORD not set in environment")
        print("   Create a .env file from .env.example and fill in your credentials")
        return None

    print(f"API Key:     {'***' + auth.api_key[-3:] if len(auth.api_key) > 3 else '(not set)'}")
    print(f"Client Code: {'***' + auth.client_code[-3:] if len(auth.client_code) > 3 else '(not set)'}")
    print(f"TOTP Enabled: {auth.totp_enabled}")
    print()

    ok = auth.login()
    if ok:
        print("\n✅ Login successful!")
        info = auth.get_session_info()
        for k, v in info.items():
            print(f"   {k}: {v}")
    else:
        print("\n❌ Login failed.")
        print("   Check your .env credentials and TOTP setup above.")

    return auth if ok else None


if __name__ == "__main__":
    test_login()