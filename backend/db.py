"""
db.py — Fullproof Database Connection Handler
---------------------------------------------
✅ Works on both Local (XAMPP, WAMP, MySQL Workbench) and Render Cloud.
✅ Automatically detects Render and enables SSL only in cloud.
✅ Handles timeouts, wrong credentials, and SSL version errors gracefully.
✅ Prints clear connection logs for debugging.
"""

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load local .env file if present (ignored in Render cloud)
load_dotenv()


def get_connection():
    """
    Returns a secure, auto-configured MySQL connection.
    - SSL is disabled for local environments (localhost, 127.0.0.1)
    - SSL is enabled automatically for Render, Railway, or any remote host
    """

    host = os.getenv("MYSQLHOST", "localhost")
    user = os.getenv("MYSQLUSER", "root")
    password = os.getenv("MYSQLPASSWORD", "")
    database = os.getenv("MYSQLDATABASE", "mca_project")
    port = int(os.getenv("MYSQLPORT", 3306))

    # Detect environment
    render_env = bool(os.getenv("RENDER")) or "render.com" in host
    railway_env = "railway" in host
    use_ssl_env = os.getenv("USE_SSL", "auto").lower()  # manual override if needed

    # SSL decision logic
    if use_ssl_env == "true":
        use_ssl = True
    elif use_ssl_env == "false":
        use_ssl = False
    else:
        use_ssl = (
            render_env or railway_env or not host.startswith(("localhost", "127.0.0.1"))
        )

    try:
        # Establish connection
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
        # Categorize and print meaningful messages
        error_msg = str(e).lower()

        if "access denied" in error_msg:
            print("❌ [DB ERROR] Access denied — check MYSQLUSER or MYSQLPASSWORD.")
        elif "unknown database" in error_msg:
            print(f"❌ [DB ERROR] Database '{database}' does not exist.")
        elif "ssl routines" in error_msg:
            print("❌ [DB ERROR] SSL handshake failed — likely wrong SSL mode.")
            print("💡 Tip: set USE_SSL=false in your .env for local testing.")
        elif "can't connect" in error_msg or "refused" in error_msg:
            print("❌ [DB ERROR] Cannot connect to MySQL server — verify host/port.")
        elif "timeout" in error_msg:
            print("⚠️ [DB WARNING] Connection timed out — server not responding.")
        else:
            print(f"❌ [DB ERROR] MySQL connection failed: {e}")

        # Reraise exception for higher-level handling
        raise

    except Exception as e:
        print(f"🔥 [DB CRITICAL] Unexpected error: {e}")
        raise


# Optional standalone test for developers
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
