"""
db.py — Robust MySQL Connection Handler
---------------------------------------
✅ Supports both Local (XAMPP/WAMP/MySQL Workbench) and Cloud (Render/Railway).
✅ Automatically detects cloud environments and enables SSL as needed.
✅ Handles timeouts, credential errors, and SSL issues gracefully.
✅ Provides clear connection logs for debugging.
"""

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables from .env file (ignored in Render)
load_dotenv()


def get_connection():
    """
    Establish and return a secure, auto-configured MySQL connection.
    - Disables SSL for localhost/127.0.0.1
    - Enables SSL automatically for Render, Railway, or remote hosts
    """
    # Load database credentials with fallbacks
    host = os.getenv("MYSQLHOST", "localhost")
    user = os.getenv("MYSQLUSER", "root")
    password = os.getenv("MYSQLPASSWORD", "")
    database = os.getenv("MYSQLDATABASE", "mca_project")
    port = int(os.getenv("MYSQLPORT", 3306))

    # Determine environment type
    render_env = bool(os.getenv("RENDER")) or "render.com" in host
    railway_env = "railway" in host
    ssl_pref = os.getenv("USE_SSL", "auto").lower()  # optional manual override

    # Decide whether to use SSL
    if ssl_pref == "true":
        use_ssl = True
    elif ssl_pref == "false":
        use_ssl = False
    else:
        use_ssl = (
            render_env or railway_env or not host.startswith(("localhost", "127.0.0.1"))
        )

    try:
        # Create MySQL connection
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            ssl_disabled=not use_ssl,
            ssl_verify_identity=False,  # Render supports self-signed certs
            connection_timeout=10,
        )

        if connection.is_connected():
            env_label = "Render Cloud" if use_ssl else "Localhost"
            print(
                f"[DB SUCCESS] Connected to MySQL ({env_label}) — SSL={'ON' if use_ssl else 'OFF'}"
            )
            return connection

    except Error as e:
        # Handle known MySQL errors clearly
        msg = str(e).lower()
        if "access denied" in msg:
            print("❌ [DB ERROR] Access denied — check MYSQLUSER or MYSQLPASSWORD.")
        elif "unknown database" in msg:
            print(f"❌ [DB ERROR] Database '{database}' does not exist.")
        elif "ssl routines" in msg:
            print("❌ [DB ERROR] SSL handshake failed — likely wrong SSL mode.")
            print("💡 Tip: set USE_SSL=false in your .env for local testing.")
        elif any(err in msg for err in ["can't connect", "refused"]):
            print("❌ [DB ERROR] Cannot connect to MySQL server — verify host/port.")
        elif "timeout" in msg:
            print("⚠️ [DB WARNING] Connection timed out — server not responding.")
        else:
            print(f"❌ [DB ERROR] MySQL connection failed: {e}")
        raise

    except Exception as e:
        print(f"🔥 [DB CRITICAL] Unexpected error: {e}")
        raise


# Standalone test utility
if __name__ == "__main__":
    print("🔍 Testing MySQL connection...")
    try:
        conn = get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DATABASE();")
            print("📦 Connected to database:", cursor.fetchone()[0])
            cursor.close()
            conn.close()
    except Exception as e:
        print("❌ Connection test failed:", e)
